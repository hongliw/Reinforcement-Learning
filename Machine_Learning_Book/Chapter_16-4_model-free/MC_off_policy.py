import numpy as np
from collections import defaultdict
import gym
from plot_utils import plot_blackjack_values, plot_policy


env = gym.make('Blackjack-v1')

def get_probs(Q_s, epsilon, nA):
    """
    epsilon-greedy策略
    """
    policy_s = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q_s)
    policy_s[best_action] = 1 - epsilon + (epsilon / nA)
    return policy_s


def generate_episode_from_Q(env, Q, epsilon, nA):
    """
    执行epsilon-greedy策略产生轨迹
    """
    episode = []
    state = env.reset()

    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA))
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state

        if done:
            env.close()
            break

    return episode


def update_Q_GLIE(episode, Q, N, epsilon, nA, gamma):
    """
    off-policy蒙特卡洛强化学习
        行为策略：产生轨迹使用的是策略π的epsilon-greedy策略，
        目标策略：argmax Q(x,a)
    更新动作-状态值函数
    new_Q = old_Q + (R - old_Q) / (old_N + 1)
    new_N += 1
    """
    states, actions, rewards = zip(*episode)
    # generate the importance sampling coefficient
    coeffs = []
    for i, state in enumerate(states):
        if actions[i] == np.argmax(Q[state]):
            p_i = 1.0 / (1 - epsilon + (epsilon / nA))
        else:
            p_i = 1.0 / (epsilon / nA)
        coeffs.append(p_i)
        coeffs[:i] *= p_i

    # prepare for discounting
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

    for i, state in enumerate(states):
        R = sum(rewards[i:] * discounts[:-(i + 1)] * coeffs[i:])
        old_Q = Q[state][actions[i]]
        old_N = N[state][actions[i]]
        Q[state][actions[i]] = old_Q + (R - old_Q) / (old_N + 1)
        N[state][actions[i]] += 1

    return Q, N


def mc_control_GLIE_off_policy(env, episode_num, gamma=1.0):
    """

    """
    nA = env.action_space.n

    # Initialize
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.zeros(nA))

    for i_episode in range(1, episode_num + 1):
        if i_episode % 1000 == 0:
            print('\nEpisode {}/{}'.format(i_episode, episode_num))

        # set the value of epsilon
        epsilon = 1.0 / ((i_episode/8000) + 1)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # update the action-value function
        Q, N = update_Q_GLIE(episode, Q, N, epsilon, nA, gamma)

    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k, np.argmax(v)) for k, v in Q.items())

    return policy, Q


if __name__ == '__main__':
    policy_off_policy, Q_off_policy = mc_control_GLIE_off_policy(env, episode_num=500000)

    # obtain the state-value function
    V_off_policy = dict((k, np.max(v)) for k, v in Q_off_policy.items())

    # plot the optimal state-value function
    plot_blackjack_values(V_off_policy, img_name='MC_off_policy.png')

    # plot the optimal policy
    plot_policy(policy_off_policy, img_name='off_policy.png')


