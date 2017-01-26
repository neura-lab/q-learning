import numpy as np
import gym
from numpy.random import choice
import random
from tensorbuilder.api import *
import tensorflow as tf

env = gym.make("FrozenLake-v0")

def next_action(actions, e):
    n = len(actions)

    if random.random() < e:
        return random.randint(0, n-1)
    else:
        return np.argmax(actions)

n_actions = env.action_space.n
n_states = env.observation_space.n
learning_rate = 0.85
y = 0.95
k = 2000.0
model_name = "shallow.model"
model_path = "models/" + model_name

graph = tf.Graph()
with graph.as_default():
    with tf.device("cpu:0"):
        s = tf.placeholder(tf.int32, (), name='s')
        a = tf.placeholder(tf.int32, (), name='a')
        step = tf.placeholder(tf.int32, (), name='step')
        r = tf.placeholder(tf.float32, (), name='r')

        ops = dict(trainable=True, weights_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.01), biases_initializer=None) #tf.random_uniform_initializer(minval=0, maxval=0.01))


        Qs = Pipe(
            s,
            T.one_hot(n_states).expand_dims(0)
            .relu_layer(32, **ops)
            .linear_layer(4, scope='linear_layer', **ops),
            T[0]
        )

        Qsa= Qs[a]

        maxQs = tf.reduce_max(Qs, 0)
        argmaxQs = tf.argmax(Qs, 0)

        ws = tf.trainable_variables()
        dws = [ tf.placeholder(w.dtype, w.get_shape()) for w in ws ]
        update = [ tf.assign_add(w, dw) for w, dw in zip(ws, dws) ]

        gradients = tf.gradients(Qsa, ws)

        writer = tf.summary.FileWriter('/logs/' +  model_name)
        saver = tf.train.Saver()

_s = env.reset()
done = False
with tf.Session(graph=graph) as sess:
    saver.restore(sess, model_path)

    for i in range(1000):
        _qs = sess.run(Qs, feed_dict={s: _s})
        _a = next_action(_qs, 0)
        _s, _r, done, info = env.step(_a)
        env.render()
        print("")

        if done:
            print(_r)
            break