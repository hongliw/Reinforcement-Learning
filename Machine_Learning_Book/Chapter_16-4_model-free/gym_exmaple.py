"""
简单了解强化学习库Gym，Gum库的环境'Blackjack-v1'实现了21点游戏。
"""
import gym
import numpy as np

# Example
# observation = (14, 5, True)
# env.player: [3, 1], env.dealer: [5, 2]
# action= 1:hit
# observation = (18, 5, True), reward = 0.0, done = False
# env.player: [3, 1, 4], env.dealer: [5, 2]
# action = 1:hit
# observation = (18, 5, False), reward = 0.0, done = False
# env.player: [3, 1, 4, 10], env.dealer: [5, 2]
# action = 1:hit
# observation = (28, 5, False), reward = -1.0, done = True

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    observation, _ = env.reset()
    print('observation = {}'.format(observation))

    while True:
        print('env.player: {}, env.dealer: {}'.format(env.player, env.dealer))
        action = np.random.choice(env.action_space.n)
        print('action = {}'.format(['0:stick', '1:hit'][action]))
        observation, reward, done, _, _ = env.step(action)
        print('observation = {}, reward = {}, done = {}'.format(observation, reward, done))

        if done:
            env.close()
            break