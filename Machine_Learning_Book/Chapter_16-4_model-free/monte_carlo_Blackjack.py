"""
策略π：若点数之和超过18，则选择action=0，即停止加牌
完成上述策略的评估，即计算该策略下的状态-动作值函数
"""
import gym
from collections import defaultdict
import numpy as np
from plot_utils import plot_blackjack_values

env = gym.make('Blackjack-v1')


def generate_episode_from_limit_stochastic(bj_env):
    """
    利用以下策略π生成episode：
        若点数之和超过18，则玩家以80%的概率选择action=0，即停止加牌；
        若点数之和小于等于18，则玩家以80%的概率选择action=1，即继续要牌
    """
    episode = []
    state, _ = bj_env.reset()

    while True:
        probs = [0.2, 0.8] if state[0] <= 18 else [0.8, 0.2]
        action = np.random.choice(np.arange(bj_env.action_space.n), p=probs)
        next_state, reward, done, _, _ = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state

        if done:
            bj_env.close()
            break

    return episode


def mc_prediction_q(bj_env, episode_num, gamma=1.0):
    """
    策略评估
    """
    # 记录累积收获
    returns_sum = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    N = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    # 状态动作值函数
    Q = defaultdict(lambda: np.zeros(bj_env.action_space.n))

    for i_episode in range(episode_num):
        if i_episode % 1000 == 0:
            print('\nEpisode {}/{}'.format(i_episode, episode_num))

        # 执行策略π产生轨迹
        episode = generate_episode_from_limit_stochastic(bj_env)

        states, actions, rewards = zip(*episode)

        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

        for i, state in enumerate(states):
            returns_sum[state][actions[i]] += sum(rewards[i:] * discounts[:-(i + 1)])
            N[state][actions[i]] += 1
            Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]

    return Q


if __name__ == '__main__':
    Q_table = mc_prediction_q(env, episode_num=50000)

    # 从状态-动作值函数求出状态值函数
    v_to_plot = defaultdict(float)
    for k, v in Q_table.items():
        if k[0] > 18:
            v_to_plot[k] = np.dot([0.8, 0.2], v)
        else:
            v_to_plot[k] = np.dot([0.2, 0.8], v)

    plot_blackjack_values(v_to_plot, img_name='MC_prediction_values.png')