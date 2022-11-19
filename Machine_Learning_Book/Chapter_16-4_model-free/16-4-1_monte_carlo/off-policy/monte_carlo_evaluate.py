"""
off-policy策略评估
基于行为策略评估目标策略，采用普通重要性采样的策略评估方法
行为策略（behavior policy）：柔性策略π(a|s)=0.5（s ∈ S, a ∈ A）
目标策略（target policy）：若总点数之和>=20时不再要牌，即选择action=0；在总点数<20时要牌，即选择action=1
"""
import numpy as np
import gym
import sys
sys.path.append('../..')
from utils import ob2state, plot_blackjack_values

env = gym.make('Blackjack-v1')


def generate_episode_from_behavior_policy(env, behavior_policy):
    """
    执行行为策略behavior_policy产生轨迹
    """
    episode = []
    observation, _ = env.reset()

    while True:
        state = ob2state(observation)
        action = np.random.choice(env.action_space.n, p=behavior_policy[state])
        episode.append((state, action))

        observation, reward, done , _, _ = env.step(action)

        if done:
            env.close()
            break

    return episode, reward


def evaluate_monte_carlo_importance_resample(env, policy, behavior_policy, episode_num):
    """
    重要性采样策略评估
    """
    Q = np.zeros_like(policy)
    N = np.zeros_like(policy)

    for i_episode in range(1, episode_num + 1):
        if i_episode % 1000 == 0:
            print('\nEpisode {}/{}'.format(i_episode, episode_num))

        # 执行行为策略产生轨迹
        episode, reward = generate_episode_from_behavior_policy(env, behavior_policy)

        # 普通重要性采样比率
        rho = 1.0
        # 为了更有效的更新重要性采样比率，所以采用逆序更新
        for state, action in reversed(episode):
            # 重要性加权累积奖赏
            G = reward * rho
            N[state][action] += 1
            Q[state][action] += (G - Q[state][action]) / N[state][action]
            rho *= (policy[state][action] / behavior_policy[state][action])

            if rho == 0:  # 提前结束
                break

    return Q


def main():
    # 目标策略
    policy = np.zeros((22, 11, 2, 2))
    policy[20:, :, :, 0] = 1  # >= 20时不再要牌
    policy[:20, :, :, 1] = 1  # < 20时继续要牌

    # 行为策略
    behavior_policy = np.ones_like(policy) * 0.5

    # 状态-动作值函数
    Q = evaluate_monte_carlo_importance_resample(env, policy, behavior_policy, episode_num=500000)

    # 状态值函数
    V = (Q * policy).sum(axis=-1)

    plot_blackjack_values(V, img_name='./results/MC_evaluate_values.png')


if __name__ == '__main__':
    main()