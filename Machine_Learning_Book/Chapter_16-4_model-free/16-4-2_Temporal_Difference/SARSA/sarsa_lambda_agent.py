"""
SARSA(λ)求解最优策略算法在通用格子世界的应用
"""
import numpy as np
from collections import defaultdict
from gym import Env
import sys
sys.path.append('..')
from gridworld import SimpleGridWorld


class SarsaLambdaAgent(object):
    def __init__(self, env: Env):
        self.env = env
        self.nA = env.action_space.n
        self.nS = env.observation_space.n
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.E = defaultdict(lambda: np.zeros(self.nA))  # Eligibility Trace 效用追踪表


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


    def learning(self, episode_num, gamma,alpha, lambda_):
        """
        SARSA(λ)求解最优策略
        """
        for i_episode in range(1, episode_num):
            # reset the value of E
            self.E = defaultdict(lambda: np.zeros(self.nA))
            s0 = self.env.reset()
            self.env.render()

            epsilon = 1.0 / (i_episode + 1.0)
            a0 = self.perform_epsilon_greedy_policy(s0, epsilon)

            time_in_episode = 0
            is_done = False

            while not is_done:
                s1, r1, is_done, _ = self.env.step(a0)
                self.env.render()

                # apply epsilon-greedy policy generate a_prime
                a_prime = self.perform_epsilon_greedy_policy(s1, epsilon)

                # update the value of Q
                old_Q = self.Q[s0][a0]
                delta = r1 + gamma * self.Q[s1][a_prime] - old_Q
                self.E[s0][a0] += 1
                for state in self.E.keys():
                    for action in range(self.nA):
                        self.Q[state][action] += alpha * delta * self.E[state][action]
                        self.E[state][action] = gamma * lambda_ * self.E[state][action]

                s0, a0 = s1, a_prime
                time_in_episode += 1

            print('Episode {} takes {} steps.'.format(i_episode, time_in_episode))


if __name__ == '__main__':
    env = SimpleGridWorld()
    agent = SarsaLambdaAgent(env)
    agent.learning(env, episode_num=800, alpha=0.1, gamma=0.9, lambda_=0.01)



