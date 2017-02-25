import time

import gym
env = gym.make('CartPole-v1')
env.reset()
while True:
    env.render()
    s, r, done, info = env.step(env.action_space.sample())
    print s, r, done, info

    if done:
        break



    # time.sleep(0.1)
print env.__doc__
time.sleep(5)