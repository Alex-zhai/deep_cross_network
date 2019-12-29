# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/3/22 10:04

from __future__ import print_function, absolute_import, division

import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model


class CrossLayer(layers.Layer):
    def __init__(self, output_dim, num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = int(input_shape[2])
        self.w = []
        self.b = []
        for i in range(self.num_layer):
            self.w.append(self.add_weight(shape=[1, self.input_dim], initializer='glorot_uniform', name='w_' + str(i),
                                          trainable=True))
            self.b.append(self.add_weight(shape=[1, self.input_dim], initializer='zeros', name='b_' + str(i),
                                          trainable=True))
        self.built = True

    def call(self, inputs):
        for i in range(self.num_layer):
            if i == 0:
                cross = layers.Lambda(lambda x: layers.Add()([
                    K.sum(self.w[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), x), axis=1, keepdims=True),
                    self.b[i], x]))(inputs)
            else:
                cross = layers.Lambda(lambda x: layers.Add()([
                    K.sum(self.w[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), inputs), axis=1,
                          keepdims=True), self.b[i], inputs]))(cross)
        return layers.Flatten()(cross)


def create_model(input_shape, out_units=1):
    inp = layers.Input(shape=input_shape)

    # deep branch
    deep = layers.Flatten()(inp)
    deep = layers.Dense(50, activation='relu')(deep)
    deep = layers.Dropout(0.5)(deep)
    deep = layers.Dense(40, activation='relu')(deep)
    deep = layers.Dropout(0.3)(deep)

    # cross branch
    cross = CrossLayer(output_dim=input_shape[1], num_layer=2, name="cross_layer")(inp)

    concat = layers.Concatenate()([deep, cross])
    out = layers.Dense(out_units, activation=None)(concat)
    return Model(inputs=inp, outputs=out)


if __name__ == '__main__':
    model = create_model(input_shape=(1, 200,))
    print(model.summary())
