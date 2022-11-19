"""
off-policy最优策略求解
基于行为策略来评估目标策略，并依据评估出的动作-状态值函数迭代求解出最优策略
行为策略（behavior policy）：柔性策略π(a|s)=0.5（s ∈ S, a ∈ A）
目标策略（target policy）：最优策略
"""
import numpy as np
import gym
import sys
sys.path.append('..')
from utils import ob2state, plot_blackjack_values, plot_policy


env = gym.make('Blackjack-v1')

def generate_episode_from_behavior_policy(env, behavior_policy):
    """
    执行行为策略产生轨迹
    """
    episode = []
    observation, _ = env.reset()

    while True:
        state = ob2state(observation)
        action = np.random.choice(env.action_space.n, p=behavior_policy[state])
        episode.append((state, action))

        observation, reward, done, _, _ = env.step(action)

        if done:
            env.close()
            break

    return episode, reward


def monte_carlo_importance_resample(env, behavior_policy, policy, episode_num):
    """
    off-policy最优策略求解
    """
    Q = np.zeros_like(policy)
    N = np.zeros_like(policy)

    for i_episode in range(1, episode_num + 1):
        if i_episode % 1000 == 0:
            print('\nEpisode {}/{}'.format(i_episode, episode_num))

        episode, reward = generate_episode_from_behavior_policy(env, behavior_policy)

        # 普通重要性采样比率
        rho = 1.
        for state, action in reversed(episode):
            G = reward * rho
            N[state][action] += 1
            Q[state][action] += (G - Q[state][action]) / N[state][action]

            # 策略改进
            best_action = np.argmax(Q[state])
            policy[state] = 0.
            policy[state][best_action] = 1.

            if action != best_action:
                break

            rho *= 1. / behavior_policy[state][action]

    return policy, Q


def main():
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 0] = 1  # 初始策略
    behavior_policy = np.ones_like(policy) * 0.5  # 行为策略

    policy, Q = monte_carlo_importance_resample(env, behavior_policy, policy, episode_num=500000)

    print(policy)
    V = Q.max(axis=-1)

    # plot state-value function
    plot_blackjack_values(V, img_name='./results/MC_control_values.png')

    # plot policy
    plot_policy(policy.argmax(axis=-1), img_name='./results/MC_control_policy.png')


if __name__ == '__main__':
    main()

