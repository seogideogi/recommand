import time
import random
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, MaxPooling2D, Conv2D, Dropout, Lambda, Dense, Flatten, Activation, Input, Embedding, BatchNormalization
from tensorflow.keras.initializers import glorot_normal, Zeros, TruncatedNormal
from tensorflow.keras.regularizers import l2


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy


from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import math

class AutoIntLayer(Layer):
    def __init__(self, embedding_size, att_head_num, att_res):
        super(AutoIntLayer, self).__init__()
        self.att_head_num = att_head_num
        self.att_res = att_res
        self.W_Query = Dense(embedding_size, use_bias=False)
        self.W_Key = Dense(embedding_size, use_bias=False)
        self.W_Value = Dense(embedding_size, use_bias=False)

    def call(self, inputs):
        # Multi-head Self-Attention
        Q = self.W_Query(inputs)
        K = self.W_Key(inputs)
        V = self.W_Value(inputs)

        # Split heads
        Q = tf.concat(tf.split(Q, self.att_head_num, axis=-1), axis=0)
        K = tf.concat(tf.split(K, self.att_head_num, axis=-1), axis=0)
        V = tf.concat(tf.split(V, self.att_head_num, axis=-1), axis=0)

        # Attention scores
        attention = tf.matmul(Q, K, transpose_b=True)
        attention = tf.nn.softmax(attention, axis=-1)

        # Output
        output = tf.matmul(attention, V)
        output = tf.concat(tf.split(output, self.att_head_num, axis=0), axis=-1)

        if self.att_res:
            output += inputs

        return output


class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.total_dim = sum(field_dims)
        self.embed_dim = embed_dim
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        self.embedding = tf.keras.layers.Embedding(input_dim=self.total_dim, output_dim=self.embed_dim)

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.embedding.set_weights([tf.keras.initializers.GlorotUniform()(shape=self.embedding.weights[0].shape)])

    def call(self, x):
        x = x + tf.constant(self.offsets)
        return self.embedding(x)

class MultiLayerPerceptron(Layer):
    def __init__(self, input_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, init_std=0.0001, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        hidden_units = [input_dim] + list(hidden_units)
        if output_layer:
            hidden_units += [1]

        self.linears = [Dense(units, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=init_std),
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)) for units in hidden_units[1:]]
        self.activation = tf.keras.layers.Activation(activation)
        if self.use_bn:
            self.bn = [BatchNormalization() for _ in hidden_units[1:]]
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.use_bn:
                x = self.bn[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        return x

class MultiHeadSelfAttention(Layer):

    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_res:
            self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                         dtype=tf.float32,
                                         initializer=TruncatedNormal(seed=self.seed))

        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores =  tf.nn.softmax(inner_product)

        result = tf.matmul(self.normalized_att_scores, values)
        result = tf.concat(tf.split(result, self.head_num, ), axis=-1)
        result = tf.squeeze(result, axis=0) 

        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        result = tf.nn.relu(result)

        return result

    def compute_output_shape(self, input_shape):

        return (None, input_shape[1], self.att_embedding_size * self.head_num)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num
                  , 'use_res': self.use_res, 'seed': self.seed}
        base_config = super(MultiHeadSelfAttention, self).get_config()
        base_config.update(config)
        return base_config


# autoint+ 모델
class AutoIntMLP(Layer): 
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0.4, init_std=0.0001):
        super(AutoIntMLP, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size

        self.final_layer = Dense(1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=init_std))
        
        self.dnn = Sequential([
            Dense(unit, activation=dnn_activation, kernel_regularizer=l2(l2_reg_dnn)) for unit in dnn_hidden_units
        ] + [
            Dense(1, activation='sigmoid')
        ])
        if dnn_use_bn:
            self.dnn.add(BatchNormalization())
        self.dnn.add(Dropout(dnn_dropout))
        
        self.int_layers = [
            AutoIntLayer(self.embedding_size, att_head_num, att_res) for _ in range(att_layer_num)
        ]

    def call(self, inputs):
        embed_x = self.embedding(inputs)
        dnn_embed = tf.reshape(embed_x, shape=(-1, self.embedding_size * self.num_fields))

        att_input = embed_x
        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = Flatten()(att_input)
        att_output = self.final_layer(att_output)
        
        dnn_output = self.dnn(dnn_embed)
        y_pred = tf.sigmoid(att_output + dnn_output)
        
        return y_pred


class AutoIntMLPModel(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True,
                 dnn_hidden_units=(32, 32), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5,
                 dnn_use_bn=False, dnn_dropout=0.4, init_std=0.0001):
        super(AutoIntMLPModel, self).__init__()
        
        # AutoIntMLP 레이어 초기화
        self.autoint_mlp = AutoIntMLP(field_dims, embedding_size, att_layer_num, att_head_num, att_res,
                                      dnn_hidden_units, dnn_activation, l2_reg_dnn, l2_reg_embedding,
                                      dnn_use_bn, dnn_dropout, init_std)
        
        # 출력 레이어 (선택적, AutoIntMLP에서 이미 처리할 수 있음)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        # AutoIntMLP 레이어를 통과
        x = self.autoint_mlp(inputs)
        
        # 필요한 경우 추가적인 출력 레이어 적용
        # x = self.output_layer(x)
        
        return x

    def build_graph(self, input_shape):
        # 모델 그래프 빌드를 위한 메서드
        x = tf.keras.layers.Input(shape=input_shape)
        return Model(inputs=[x], outputs=self.call(x))

# 모델 컴파일 및 학습을 위한 예시 코드
def compile_and_fit(model, train_dataset, val_dataset, epochs=10):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    
    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=val_dataset,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    
    return history
    
    
def predict_model(model, pred_df):
    batch_size = 2048
    top=10
    user_pred_info = []
    total_rows = len(pred_df)
    for i in range(0, total_rows, batch_size):
        features = pred_df.iloc[i:i + batch_size, :].values
        y_pred = model.predict(features, verbose=False)
        for feature, p in zip(features, y_pred):
            u_i = feature[:2]
            user_pred_info.append((int(u_i[1]), float(p)))
    
    return sorted(user_pred_info, key=lambda s : s[1], reverse=True)[:top]