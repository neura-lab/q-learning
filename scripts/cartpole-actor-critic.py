import numpy as np
import gym
from numpy.random import choice
import random
from tensorbuilder.api import *
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

env = gym.make("CartPole-v1")


n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
learning_rate = 0.85
y = 0.9
model_name = "deep-integrated-policy-gradient-cartpole-v1.model"
model_path = "models/" + model_name

graph = tf.Graph()
gsess = tf.InteractiveSession(graph=graph)
with graph.as_default():
    with tf.device("cpu:0"):
        s = tf.placeholder(tf.float32, [n_states])
        step = tf.placeholder(tf.int32, [])
        a = tf.placeholder(tf.int32, [])
        r = tf.placeholder(tf.float32, [])


        ops = dict(trainable=True, weights_initializer=tf.random_uniform_initializer(minval=0, maxval=0.01), biases_initializer=None) #tf.random_uniform_initializer(minval=0, maxval=0.01))


        [Ps, V] = Pipe(
            s,
            T.expand_dims(0)
            .relu_layer(32, scope='relu_layer', **ops)
            .relu_layer(16, scope='relu_layer2', **ops),
            [
                T.softmax_layer(n_actions, scope='softmax_layer_actor', **ops)
                >> T[0]
            ,
                T.linear_layer(1, scope='linear_layer_critic', **ops)
                >> T[0,0]
            ]
        )
        Psa = Ps[a]

        ws = tf.trainable_variables()

        gradients_actor = tf.gradients(tf.log(Psa), ws)
        gradients_actor = [ g if g is not None else tf.zeros_like(w) for g, w in zip(gradients_actor, ws) ]

        gradients_critic = tf.gradients(V, ws)
        gradients_critic = [ g if g is not None else tf.zeros_like(w) for g, w in zip(gradients_critic, ws) ]

        dws = [ tf.placeholder(w.dtype, w.get_shape()) for w in ws ]
        update = [ tf.assign_add(w, ws) for w, ws in zip(ws, dws) ]


        writer = tf.summary.FileWriter('/logs/' +  model_name)
        saver = tf.train.Saver()


def next_action(actions, get_max=False):
    n = actions.shape[0]
    return choice(n, p=actions) if not get_max else np.argmax(actions)


import time

_s = env.reset()
done = False
with tf.Session(graph=graph) as sess:
    # saver.restore(sess, model_path)
    sess.run(tf.global_variables_initializer())

    for i in range(1000):

        _ps = sess.run(Ps, feed_dict={s: _s})
        _a = next_action(_ps, 0)


        print _s

        print "psa", [_psa for _psa in _ps]
        print "a", _a

        _s, _r, done, info = env.step(_a)
        env.render()

        print "s", _s
        print "r", _r

        print("")

        time.sleep(0.2)

        if done:
            print(_r)
            break