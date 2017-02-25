import numpy as np
import gym
from numpy.random import choice
import random
from tensorbuilder.api import *
import tensorflow as tf



env = gym.make("CartPole-v1")

def select_columns(tensor, indexes):
    idx = tf.stack((tf.range(tf.shape(indexes)[0]), indexes), 1)
    return tf.gather_nd(tensor, idx)

def discount(rewards, y):
    r_accum = 0.0
    gains = []
    for r in reversed(list(rewards)):
        r_accum = r + y * r_accum
        gains.insert(0, r_accum)

    return gains


model_name = "policy-gradient-cartpole.model"
model_path = "models/" + model_name
n_actions = env.action_space.n
n_states_env = env.observation_space.shape[0]
n_states = n_states_env * 3

class Model(object):

    def __init__(self, y, restore=False):

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with tf.device("cpu:0"):
                s = tf.placeholder(tf.float32, [None, n_states], name='s')
                a = tf.placeholder(tf.int32, [None], name='a')
                r = tf.placeholder(tf.float32, [None], name='r')
                lr = tf.placeholder(tf.float32, [], name='lr')

                trainer = tf.train.GradientDescentOptimizer(lr)

                ops = dict(trainable=True, weights_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.01), biases_initializer=None) #tf.random_uniform_initializer(minval=0, maxval=0.01))

                with tf.variable_scope("actor"):
                    Ps = Pipe(
                        s,
                        T
                        .relu_layer(16, **ops)
                        .softmax_layer(n_actions, scope='softmax_layer', **ops)
                    )
                Psws = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "actor")

                Psa = select_columns(Ps, a)

                base = tf.Variable(0.0)

                error = r - base

                loss = -tf.reduce_sum(tf.log(Psa) * error)
                gradients = trainer.compute_gradients(loss, var_list=Psws)
                gradients = [ (tf.clip_by_value(g, -5.0, 5.0), w) for g, w in gradients ]
                update = trainer.apply_gradients(gradients)

                loss_base = Pipe(error, tf.nn.l2_loss, tf.reduce_sum)
                gradients = trainer.compute_gradients(loss_base, var_list=[base])
                gradients = [ (tf.clip_by_value(g, -5.0, 5.0), w) for g, w in gradients ]
                update_base = trainer.apply_gradients(gradients)

                self.writer = tf.summary.FileWriter('/logs/' +  model_name)
                self.saver = tf.train.Saver()

                self.variables_initializer = tf.global_variables_initializer()



            if restore:
                self.saver.restore(self.sess, model_path)
            else:
                self.sess.run(self.variables_initializer)

        self.s = s; self.a = a; self.r = r;
        self.Ps = Ps; self.Psa = Psa; self.update = update; self.update_base = update_base
        self.lr = lr

    def next_action(self, state, get_max=False):
        actions = self.sess.run(self.Ps, feed_dict={self.s: [state]})[0]
        n = len(actions)

        return choice(n, p=actions) if not get_max else np.argmax(actions)

    def train(self, s, a, r, s1, lr):
        #train
        self.train_offline([s], [a], [r], [s1], lr)

    def train_offline(self, S, A, R, S1, lr):
        #train
        self.sess.run(self.update, feed_dict={
            self.s: S, self.a: A, self.r: R,
            self.lr: lr
        })

        self.sess.run(self.update_base, feed_dict={
            self.s: S, self.a: A, self.r: R,
            self.lr: lr
        })

    def save(self, model_path):
        self.saver.save(self.sess, model_path)

    def restore(self, model_path):
        self.sess.close()
        self.sess = tf.Session(graph=self.graph)
        self.saver.restore(self.sess, model_path)

    @staticmethod
    def learning_rate(t, b, k):
        return b * k / (k + t)


y = 0.98
b = 0.5
k = 2000.0

model = Model(y, restore=True)

import time
s = env.reset()
s = np.hstack((s,s,s))

rt = 0
for i in range(10000):
    a = model.next_action(s, True)
    s1, r, done, info = env.step(a)
    s = np.hstack((s[n_states_env:], s1))
    env.render()

    rt += r
    time.sleep(0.02)

    if done:
        print(rt)
        break