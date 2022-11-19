"""
带起始探索的on-policy最优策略求解
"""
import numpy as np
import gym
import sys
sys.path.append('../..')
from utils import ob2state, plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v1')


def generate_episode_from_stochastic_start(env, policy):
    """
    带起始探索产生轨迹
    """
    # 随机选择初始起始状态和起始动作
    state = (np.random.randint(12, 22),
             np.random.randint(1, 11),
             np.random.randint(2))
    action = np.random.randint(2)
    env.reset()

    if state[2]:  # 有A
        env.player = [1, state[0] - 11]
    else:  # 没有A
        if state[0] == 21:
            env.player = [10, 9, 2]
        else:
            env.player = [10, state[0] - 10]

    env.dealer[0] = state[1]

    episode = []

    while True:
        episode.append((state, action))
        observation, reward, done, _, _ = env.step(action)

        if done:
            env.close()
            break

        state = ob2state(observation)
        action = np.random.choice(env.action_space.n, p=policy[state])

    return episode, reward


def monte_carlo_with_exploring_start(env, policy, episode_num):
    """
    求解带起始探索的on-policy最优策略
    """
    Q = np.zeros_like(policy)
    N = np.zeros_like(policy)

    for i_episode in range(1, episode_num + 1):
        if i_episode % 1000 == 0:
            print('\nEpisode {}/{}'.format(i_episode, episode_num))

        # 执行策略产生轨迹
        episode, reward = generate_episode_from_stochastic_start(env, policy)

        for state, action in episode:
            N[state][action] += 1
            # 状态-动作值函数迭代更新
            Q[state][action] += (reward - Q[state][action]) / N[state][action]
            # 策略改进
            best_action = np.argmax(Q[state])
            policy[state] = 0.
            policy[state][best_action] = 1.

    return policy, Q


def main():
    policy = np.zeros((22, 11, 2, 2))
    # 初始策略
    policy[:, :, :, 1] = 1

    # 最优策略以及最优状态-动作值函数
    policy, Q = monte_carlo_with_exploring_start(env, policy, episode_num=500000)

    # 最优值函数
    V = Q.max(axis=-1)

    # plot the optimal state-value function
    plot_blackjack_values(V, img_name='./results/MC_control_values.png')

    # plot the optimal policy
    plot_policy(policy.argmax(-1), img_name='./results/MC_control_policy.png')


if __name__ == '__main__':
    main()
