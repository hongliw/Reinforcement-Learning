"""
off-policy策略评估
基于行为策略评估目标策略，采用普通重要性采样的策略评估方法
目标策略（target policy）：最优策略
行为策略（behavior policy）：epsilon-greedy策略
                          以epsilon的概率从所有动作中均匀随机选择一个，
                          以1-epsilon的概率选择当前最优动作
"""
import numpy as np
import gym
import sys
sys.path.append('..')
from utils import ob2state, plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v1')


def get_probs(Q_s, epsilon, nA):
    """
    获取epsilon-greedy策略对应的动作概率
    """
    policy_s = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q_s)
    policy_s[best_action] = 1 - epsilon + (epsilon / nA)
    return policy_s


def generate_episode_from_epsilon_greedy_policy(env, Q, epsilon, nA):
    """
    执行epsilon-greedy策略产生轨迹
    """
    episode = []

    observation, _ = env.reset()

    while True:
        state = ob2state(observation)
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA))
        episode.append((state, action))
        observation, reward, done, _, _ = env.step(action)

        if done:
            env.close()
            break

    return episode, reward


def monte_carlo_control(env, policy, nA, episode_num):
    """
    off-policy最优策略求解
    """
    # Initialize
    Q = np.zeros_like(policy)
    N = np.zeros_like(policy)

    for i_episode in range(1, episode_num + 1):
        if i_episode % 1000 == 0:
            print('\nEpisode {}/{}'.format(i_episode, episode_num))

        epsilon = 1.0 / ((i_episode / 8000) + 1.0)

        episode, reward = generate_episode_from_epsilon_greedy_policy(env, Q, epsilon, nA)

        # 重要行采样比率
        rho = 1.0
        for state, action in reversed(episode):
            # 重要性加权累积奖赏
            g = reward * rho
            N[state][action] += 1
            Q[state][action] += (g - Q[state][action]) / N[state][action]

            # 策略更新
            best_action = np.argmax(Q[state])
            policy[state] = 0.
            policy[state][best_action] = 1.

            # TODO 这里的重要性采样系数连乘与《机器学习》P386中的π(x_i, a_i)始终为1的说法不一致
            # TODO 目前不确定这种实现方式是否正确，待再深入学习几天后，再回到头来更新这块代码；
            if action != best_action:
                break

            rho *= 1. / (1 - epsilon + (epsilon / nA))

    return policy, Q


def main():
    nA = env.action_space.n
    # 初始策略
    policy = np.ones((22, 11, 2, 2)) * 0.5

    policy, Q = monte_carlo_control(env, policy, nA, episode_num=500000)

    V = np.max(Q, axis=-1)

    # plot
    plot_blackjack_values(V, img_name='./results/MC_epsilon_values.png')

    # plot
    plot_policy(np.argmax(policy, axis=-1), img_name='./results/MC_epsilon_policy.png')


if __name__ == '__main__':
    main()


