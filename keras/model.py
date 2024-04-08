import numpy as np
import gym

from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.layers as l
import tensorflow.keras.backend as K
from keras.layers import *
import tensorflow as tf


def input_model(env):
    model = Sequential()
    model.add(l.Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(l.Reshape(env.observation_space.shape))
    return model

def position_model():
    model = Sequential()
    model.add(l.Lambda(lambda x: x[:, :, :, -1]))
    model.add(l.Lambda(lambda x: K.max(x, -1)))
    return model

def features_model(env):
    w, s, _ = env.observation_space.shape
    model = Sequential()
    model.add(TokenAndPositionEmbedding(w, s, 4, 100))
    model.add(l.Conv2D(100, (12, 1), data_format='channels_last', activation='relu'))
    model.add(l.Lambda(lambda x: K.squeeze(x, -3)))
    model.add(l.GlobalMaxPool1D())
    model.add(l.Dense(100, activation='relu'))
    model.add(l.Dense(100, activation='relu'))
    model.add(l.Dense(100, activation='relu'))

    return model

def fc_model(nb_actions):
    model = Sequential()
    model.add(l.Dense(200, activation='relu'))
    model.add(l.Dense(200, activation='relu'))
    model.add(l.Dense(50, activation='relu'))
    model.add(l.Dense(nb_actions))
    return model


def make_forward(inpt, model_input, model_features, model_q):
    input_out = model_input(inpt)

    l_color = l.Lambda(lambda x: K.squeeze(x[:, :, :, :1], -1))(input_out)
    l_obstacle = l.Lambda(lambda x: K.squeeze(x[:, :, :, 1:2], -1))(input_out)
    l_position = l.Lambda(lambda x: K.squeeze(x[:, :, :, 2:], -1))(input_out)

    feat_pos = model_features(l_color)
    feat_neg = model_features(l_obstacle)
    pos_out = model_features(l_position)

    fc_in = l.Concatenate()([pos_out, feat_pos, feat_neg])

    q = model_q(fc_in)
    return q

def make_model(env, nb_actions, n_masks=10, mask_regul=1., mask_regul_temp=0.05, masks_dropout=0.8):
    model_input = input_model(env)
    model_features = features_model(env)
    model_q = fc_model(nb_actions)

    input = l.Input(shape=(1, )+env.observation_space.shape)

    q_s = make_forward(input, model_input, model_features, model_q)

    if n_masks is None or int(n_masks) <= 0:
        model = Model(inputs=[input,], outputs=[q_s,])
        return model

    masks_inputs = []
    masked_q = []
    for i in range(int(n_masks)):
        mshape = input.shape[:-1]
        mask_in = tf.random.uniform(shape=mshape[1:], maxval=1, seed=0)                                 # create random binary mask
        mask = l.Lambda(lambda x: K.expand_dims(K.cast(x < masks_dropout,'float'), -1))(mask_in)

        masked_input = mask * input                                                                     # multiply mask with input
        q = make_forward(masked_input, model_input, model_features, model_q)                            # create model on the masked input
        q = l.Lambda(lambda x: K.expand_dims(x, 1))(q)

        masks_inputs += mask_in,
        masked_q += q,

    q_m = l.Concatenate(axis=1)(masked_q) if len(masked_q) > 1 else masked_q[0]
    p_m = l.Lambda(lambda x: K.softmax(x / mask_regul_temp))(q_m)
    p_m = l.Lambda(lambda x: K.mean(x, 1))(p_m)
    p_s = l.Lambda(lambda x: K.softmax(x / mask_regul_temp))(q_s)
    dkl = tf.keras.losses.KLDivergence()
    masks_loss = l.Lambda(lambda x: mask_regul * (dkl(x[0], x[1]) + dkl(x[1], x[0]))/2)([p_s, p_m])

    model = Model(inputs=[input, ], outputs=[q_s, ])

    model.add_metric(masks_loss, name='masks_loss', aggregation='mean')
    model_q.add_loss(masks_loss)
    return model