"""
Implementation of small grid world example illustrated by David Silver
in his Reinforcement Learning Lecture3 - Planning by Dynamic Programming.
Author: Jessica
Data: October 27, 2022

The value function converges to:
  0.00 -14.00 -20.00 -22.00

-14.00 -18.00 -20.00 -20.00

-20.00 -20.00 -18.00 -14.00

-22.00 -20.00 -14.00   0.00
"""
import numpy as np

nS, nA = 16, 4
# 状态空间
states = [i for i in range(16)]
# 声明行为空间
actions = np.array(['n', 'e', 's', 'w'])
# 结合方格世界的布局特点，声明行为对状态的改变
ds_actions = {'n': -4, 'e': 1, 's': 4, 'w': -1}
# 声明衰减系数
gamma = 1.0


def nextState(s, a):
    """
    根据当前状态和行为确定下一状态
    """
    next_state = s
    if (s % 4 == 0 and a == 'w') or (s < 4 and a == 'n') or \
            ((s + 1) % 4 == 0 and a == 'e') or (s > 11 and a == 's'):
        pass
    else:
        ds = ds_actions[a]
        next_state = s + ds
    return next_state


def rewardOf(s):
    """
    得到某一状态的即时奖励
    """
    return 0 if s in [0, 15] else -1


def isTerminateState(s):
    """
    判断某一状态是否为终止状态
    """
    return s in [0, 15]


def get_action_name(index):
    if index == 0:
        return 'n'
    elif index == 1:
        return 'e'
    elif index == 2:
        return 's'
    else:
        return 'w'


def value_iteration():
    """
    值迭代(value iteration)
    """
    values = np.zeros(nS)
    newValues = values.copy()
    # 收敛阈值
    THETA = 0.0001
    delta = float('inf')

    round_num = 0

    while delta > THETA:
        print('\nRound Number:' + str(round_num))
        round_num += 1

        print('Expected value in value iteration:')
        printValue(values)

        delta = 0

        for s in states:
            expected_value = float('-inf')
            if not isTerminateState(s):
                for action in actions:
                    # 获取下一个状态
                    next_state = nextState(s, action)

                    value = rewardOf(s) + gamma * values[next_state]
                    expected_value = max(expected_value, value)

                delta = max(delta, np.abs(expected_value - values[s]))
                newValues[s] = expected_value

        values = newValues.copy()

    return values


def printValue(values):
    """
    输出状态价值
    """
    for s, v in enumerate(values):
        # 取2位小数，右对齐，取6位
        print('{0:>6.2f}'.format(v), end=' ')
        if (s + 1) % 4 == 0:
            print('\n')


def next_best_action(s, values):
    action_values = np.zeros(nA)

    for a, action_name in enumerate(actions):
        next_state = nextState(s, action_name)
        action_values[a] = rewardOf(s) + gamma * values[next_state]

    return np.argmax(action_values), np.max(action_values)


def optimal_policy():
    """
    最有策略
    """
    # 初始policy策略
    policy = np.tile(np.array([1 / nA for _ in range(nA)]), (nS, 1))

    values = value_iteration()

    for s in states:
        best_action, best_action_value = next_best_action(s, values)
        policy[s] = np.eye(nA)[best_action]

    return policy


def main():
    policy = optimal_policy()
    print('The optimal policy is:')
    print(policy)


if __name__ == '__main__':
    main()
