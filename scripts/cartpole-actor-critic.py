import numpy as np
import gym
from numpy.random import choice
import random
from tensorbuilder.api import *
import tensorflow as tf

env = gym.make("CartPole-v1")


def next_action(actions, e=0.05):
    n = len(actions)
    return choice(n, p=actions)

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


n_actions = env.action_space.n
n_states = env.observation_space.shape[0] * 2
model_name = "actor-critic-cartpole.model"
model_path = "models/" + model_name
y = 0.98
g_max = 5.0

graph = tf.Graph()
with graph.as_default():
    with tf.device("cpu:0"):
        S = tf.placeholder(tf.float32, [None, n_states], name='s')
        A = tf.placeholder(tf.int32, [None], name='a')
        R = tf.placeholder(tf.float32, [None], name='r')
        V1 = tf.placeholder(tf.float32, [None], name='v1')
        LR = tf.placeholder(tf.float32, [], name='lr')

        ops = dict(trainable=True, weights_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.01), biases_initializer=None) #tf.random_uniform_initializer(minval=0, maxval=0.01))

        with tf.variable_scope("Actor"):
            PS = Pipe(
                S,
                T
#                 .relu_layer(32, **ops)
                .relu_layer(16, **ops)
                .softmax_layer(2, **ops)
            )
        PWs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Actor")

        with tf.variable_scope("Critic"):
            V = Pipe(
                S,
                tf.Variable(0.0, name="m")
#                 T.linear_layer(1, **ops),
#                 T[:, 0]
            )
        VWs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Critic")


        PSA = select_columns(PS, A)

        Trainer = tf.train.GradientDescentOptimizer(LR)


        G = R
        E = G - V

        LossActor = -tf.reduce_mean(tf.log(PSA) * E)
        gradients_actor = Trainer.compute_gradients(LossActor, var_list=PWs)
        gradients_actor = [ (tf.clip_by_value(g, -g_max, g_max), w) for g, w in gradients_actor ]
        UpdateActor = Trainer.apply_gradients(gradients_actor)

        LossCritic = Pipe(E, tf.nn.l2_loss, tf.reduce_mean)
        gradients_critic = Trainer.compute_gradients(LossCritic, var_list=VWs)
        gradients_critic = [ (tf.clip_by_value(g, -g_max, g_max), w) for g, w in gradients_critic ]
        UpdateCritic = Trainer.apply_gradients(gradients_critic)

        Writer = tf.summary.FileWriter('/logs/' +  model_name, graph=graph)
        Saver = tf.train.Saver()


import time
s = env.reset()
s = np.hstack((s, s))
done = False
rt = 0
with tf.Session(graph=graph) as sess:
    Saver.restore(sess, model_path)

    for i in range(20000):
        ps = sess.run(PS, feed_dict={S: [s]})[0]
        a = np.argmax(ps)
        s1, r, done, info = env.step(a)
        rt += r
        s = np.hstack((s[n_states/2:], s1))
        env.render()

        time.sleep(0.01)

        if done:
            print(r)
            break

print(rt)