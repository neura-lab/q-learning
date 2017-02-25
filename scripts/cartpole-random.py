import gym
import time



env = gym.make('CartPole-v1')
env.reset()
env.render()



for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    time.sleep(0.02)