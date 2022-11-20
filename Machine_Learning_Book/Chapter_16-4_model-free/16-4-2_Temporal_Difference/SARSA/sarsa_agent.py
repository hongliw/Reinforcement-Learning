"""
SARSA求解最优策略算法在通用格子世界的应用
"""
import numpy as np
from collections import defaultdict
from gym import Env
import sys
sys.path.append('..')
from gridworld import SimpleGridWorld


class SarsaAgent(object):
    def __init__(self, env: Env):
        self.env = env
        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def _get_probs(self, Q_s, epsilon):
        """
        epsilon-greedy策略产生各动作的概率
        """
        policy_s = np.ones(self.nA) * epsilon / self.nA
        best_action = np.argmax(Q_s)
        policy_s[best_action] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def perform_epsilon_greedy_policy(self, state, epsilon):
        """
        epsilon-greedy策略产生的动作
        """
        action = np.random.choice(np.arange(self.nA), p=self._get_probs(self.Q[state], epsilon))
        return action

    def learning(self, episode_num, alpha, gamma, threshold=0.00001):
        """
        SARSA求解最优策略
        """
        for i_episode in range(1, episode_num + 1):
            s0 = self.env.reset()

            # show the UI
            self.env.render()

            # set the epsilon value
            epsilon = 1. / (i_episode + 1.)

            a0 = self.perform_epsilon_greedy_policy(s0, epsilon)

            time_in_episode = 0
            is_done = False
            delta = float('-inf')

            while not is_done:
                s1, r1, is_done, _ = self.env.step(a0)
                self.env.render()

                a_prime = self.perform_epsilon_greedy_policy(s1, epsilon)

                old_Q = self.Q[s0][a0]
                self.Q[s0][a0] = old_Q + alpha * (r1 + gamma * self.Q[s1][a_prime] - old_Q)
                delta = max(delta, np.abs(self.Q[s0][a0] - old_Q))

                s0, a0 = s1, a_prime

                time_in_episode += 1

            print('Episode {} takes {} steps.'.format(i_episode, time_in_episode))
            if delta < threshold:
                self.env.close()
                break

        # 最优策略
        policy = dict((k, np.argmax(v)) for k, v in self.Q.items())
        print('The optimal policy is: ')
        print(sorted(policy.items(), key=lambda d:d[0], reverse=False))


if __name__ == "__main__":
    env = SimpleGridWorld()
    agent = SarsaAgent(env)
    agent.learning(episode_num=800, alpha=0.1, gamma=0.9)



