"""
评估确定性策略π：若总点数之和>=20时不再要牌，即选择action=0；在总点数<20时要牌，即选择action=1
"""
import gym
from collections import defaultdict
import numpy as np
import sys
sys.path.append('..')
from plot_utils import plot_blackjack_values

env = gym.make('Blackjack-v1')


def ob2state(observation):
    return observation[0], observation[1], int(observation[2])


def generate_episode_from_policy(env, policy):
    """
    执行以下策略π生成episode：
        若总点数之和>=20时不再要牌，即选择action=0；在总点数<20时要牌，即选择action=1
    """
    episode = []
    observation, _ = env.reset()

    while True:
        state = ob2state(observation)
        action = np.random.choice(np.arange(env.action_space.n), p=policy[state])
        observation, reward, done, _, _ = env.step(action)
        episode.append((state, action))

        if done:
            env.close()
            break

    return episode, reward


def evaluate_action_monte_carlo(env, policy, episode_num):
    """
    on-policy蒙特卡洛策略评估
    """
    # 状态动作值函数
    Q = np.zeros_like(policy)
    N = np.zeros_like(policy)

    for i_episode in range(1, episode_num + 1):
        if i_episode % 1000 == 0:
            print('\nEpisode {}/{}'.format(i_episode, episode_num))

        # 执行策略π产生轨迹
        episode, reward = generate_episode_from_policy(env, policy)

        for state, action in episode:
            N[state][action] += 1
            Q[state][action] += (reward - Q[state][action]) / N[state][action]

    return Q


def main():
    policy = np.zeros((22, 11, 2, 2))
    policy[20:, :, :, 0] = 1  # >=20时不再要牌
    policy[:20, :, :, 1] = 1  # <20时要牌
    # 状态-动作值函数
    Q = evaluate_action_monte_carlo(env, policy, episode_num=500000)

    # 从状态-动作值函数求出状态值函数
    V = (Q * policy).sum(axis=-1)

    plot_blackjack_values(V, img_name='MC_prediction_values.png')


if __name__ == '__main__':
    main()
