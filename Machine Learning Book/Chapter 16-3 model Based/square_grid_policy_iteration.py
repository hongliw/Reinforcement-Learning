"""
对4*4方格随机策略π利用策略迭代法(policy iteration)进行策略改进，求解最优策略
4*4方格详细说明见square_grid.txt
"""
import numpy as np
from collections import defaultdict

# nS：状态个数，nA：行为个数
nS, nA = 16, 4
# 声明状态空间
states = [i for i in range(nS)]
# 声明行为空间
actions = ['n', 'e', 's', 'w']
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


def printValue(values):
    """
    输出状态价值
    """
    for s, v in enumerate(values):
        # 取2位小数，右对齐，取6位
        print('{0:>6.2f}'.format(v), end=' ')
        if (s + 1) % 4 == 0:
            print('\n')


def updateValue(values, policy):
    """
    迭代更新状态值函数values
    """
    newValues = np.zeros(nS)
    delta = 0
    # 遍历所有的状态
    for s in states:
        expected_value = 0

        if not isTerminateState(s):
            # 即时奖励
            r = rewardOf(s)
            # 产生随机行为动作
            for a, action_prob in enumerate(policy[s]):
                # 获取下一个状态
                a_name = actions[a]
                next_state = nextState(s, a_name)
                # 计算公式参考P378 gamma折扣累积奖赏
                # gamma=1.0
                # 当前策略产生动作a的概率为π(s, a) = 1/4
                # 执行动作a转移到next_state的概率为100%，且执行动作a只能转移到next_state P(next_state|s, a) = 100%
                expected_value += action_prob * (r + gamma * values[next_state])

            # 记录迭代后值函数的最大变化量
            delta = max(delta, np.abs(expected_value - values[s]))
        # 更新newValues
        newValues[s] = expected_value

    return newValues, delta


def policy_evaluation(policy, threshold=0.0001):
    """
    策略评估
    """
    # 声明状态值函数values
    values = np.zeros(nS)

    delta = float('inf')
    # 若值函数的改变小于threshold时，则停止
    while delta >= threshold:
        # 更新迭代值函数
        values, delta = updateValue(values, policy)

    print('The value function converges to:')
    printValue(values)
    return values


def next_best_action(s, values):
    """
    获取最优动作
    """
    action_values = np.zeros(nA)

    for a, a_name in enumerate(actions):
        next_state = nextState(s, a_name)
        action_values[a] = rewardOf(s) + gamma * values[next_state]

    return np.argmax(action_values), np.max(action_values)


def main():
    # 随机策略π
    policy = np.tile(np.array([1.00 / nA for _ in range(nA)]), (nS, 1))

    is_stable = False
    round_num = 0

    while not is_stable:
        is_stable = True

        print('Round Number: ', round_num)
        round_num += 1

        print('Current policy: ')
        print(policy)

        # 策略评估
        values = policy_evaluation(policy)

        # 策略迭代
        for s in states:
            action_by_policy = np.argmax(policy[s])
            best_action, best_action_value = next_best_action(s, values)
            policy[s] = np.eye(nA)[best_action]
            if action_by_policy != best_action:
                is_stable = False

    print('The optimal policy is:')
    print(policy)


if __name__ == '__main__':
    main()