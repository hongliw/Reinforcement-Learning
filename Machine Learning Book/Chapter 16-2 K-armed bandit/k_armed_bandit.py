import numpy as np
import matplotlib.pyplot as plt
from random import random
from matplotlib.pyplot import MultipleLocator

# 摇臂数
K = 2
# 实验次数
trial_num = 100
# 尝试次数
T = 3000


def get_reward(k):
    """
    单步奖励
    摇臂0以0.4的概率返回奖赏1，以0.6的概率返回奖赏0；
    摇臂1以0.2的概率返回奖赏1，以0.8的概率返回奖赏0；
    """
    if k == 0:
        v = np.random.choice([1, 0], p=[0.4, 0.6])
    else:
        v = np.random.choice([1, 0], p=[0.2, 0.8])

    return v


def update(Q, count, k, v):
    Q[k] = (Q[k] * count[k] + v) / (count[k] + 1)
    count[k] += 1
    return Q[k], count[k]


def exploitation_only():
    """
    策略为仅利用，即选择当前最优摇臂
    """
    print('Exploitation start...')
    # 平均累计奖赏
    rewards = []

    for i_trail in range(trial_num):
        print('i_trail: ', i_trail)
        # 初始化
        Q, count = np.zeros(K), np.zeros(K)
        # 累计奖赏
        r = 0
        #
        i_rewards = []

        for t in range(1, T + 1):
            # 总是选择当前最优摇臂
            k = np.argmax(Q)
            # 单步奖赏
            v = get_reward(k)
            # 累计奖赏
            r += v
            # 更新摇臂k的平均奖赏和选中次数
            Q[k], count[k] = update(Q, count, k, v)

            # 第i_trial轮的第t次尝试的平均累积奖赏
            i_rewards.append(r/t)

        # 第i_trail轮的平均累积奖励
        rewards.append(i_rewards)
    rewards = np.array(rewards)
    return rewards


def exploration_only():
    """
    策略为仅探索，即平均分配给每个摇臂
    """
    print('Exploration start...')
    # 平均累计奖赏
    rewards = []

    for i_trail in range(trial_num):
        print('i_trail: ', i_trail)
        # 初始化
        Q, count = np.zeros(K), np.zeros(K)
        # 累计奖赏
        r = 0
        #
        i_reward = []

        for t in range(1, T + 1):
            if (t + 1) % 2 == 0:
                k = 0
            else:
                k = 1
            # 单步奖赏
            v = get_reward(k)
            # 累计奖赏
            r += v
            # 更新摇臂k的平均奖赏和选中次数
            Q[k], count[k] = update(Q, count, k, v)
            # 平均累积奖赏
            i_reward.append(r/t)

        rewards.append(i_reward)
    rewards = np.array(rewards)
    return rewards


def epsilon_greedy(epsilon=0.1):
    """
    epsilon-贪心算法
    """
    print('Epsilon-greedy start...')
    rewards = []

    for i_trail in range(trial_num):
        # Initialize
        Q, count = np.zeros(K), np.zeros(K)
        # 累积奖赏
        r = 0
        i_reward = []

        for t in range(1, T + 1):
            if random() < epsilon:
                # 随机选取
                k = np.random.choice(np.arange(K))
            else:
                k = np.argmax(Q)
            # 单步奖赏
            v = get_reward(k)
            # 累积奖赏
            r += v
            # 更新摇臂k的平均奖赏和选中次数
            Q[k], count[k] = update(Q, count, k, v)
            # 平均累积奖赏
            i_reward.append(r / t)

        rewards.append(i_reward)
    rewards = np.array(rewards)
    return rewards


def softmax(para_T=0.1):
    """
    softmax算法
    """
    rewards = []

    for i_trail in range(trial_num):
        print('i_trail: ', i_trail)
        # Initialize
        Q, count = np.zeros(K), np.zeros(K)
        # 累积奖赏
        r = 0
        i_reward = []

        for t in range(1, T + 1):
            k = np.random.choice(np.arange(K), p=get_probs(Q, para_T))
            # 单步奖赏
            v = get_reward(k)
            # 累积奖赏
            r += v
            # 更新摇臂k的平均奖赏和选中次数
            Q[k], count[k] = update(Q, count, k, v)

            i_reward.append(r / t)

        rewards.append(i_reward)
    rewards = np.array(rewards)
    return rewards


def get_probs(Q, para_T):
    """
    Boltzmann概率
    """
    probs = []
    probs_sum = 0

    for q in Q:
        probs_sum += np.exp(q / para_T)

    for k in range(K):
        prob_k = np.exp(Q[k] / para_T) / probs_sum
        probs.append(prob_k)

    return probs


def plot_result(avg_rewards):
    """
    绘制各算法在2-摇臂赌博机上的性能比较曲线
    """
    # 设置显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 仅探索
    y_values = [0]
    for i in range(T):
        y_values.append(np.mean(avg_rewards[0][:, i]))
    plt.plot(y_values, c='green', label='仅探索')

    # 仅利用
    y_values = [0]
    for i in range(T):
        y_values.append(np.mean(avg_rewards[1][:, i]))
    plt.plot(y_values, c='red', label='仅利用')

    # epsilon-greedy e=0.1
    y_values = [0]
    for i in range(T):
        y_values.append(np.mean(avg_rewards[2][:, i]))
    plt.plot(y_values, c='blue', label='epsilon-greedy(e=0.1)')

    # epsilon-greedy e=0.01
    y_values = [0]
    for i in range(T):
        y_values.append(np.mean(avg_rewards[3][:, i]))
    plt.plot(y_values, c='black', label='epsilon-greedy(e=0.01)')

    # epsilon-greedy e=0.01
    y_values = [0]
    for i in range(T):
        y_values.append(np.mean(avg_rewards[4][:, i]))
    plt.plot(y_values, c='orange', label='softmax(T=0.1)')

    # epsilon-greedy e=0.01
    y_values = [0]
    for i in range(T):
        y_values.append(np.mean(avg_rewards[5][:, i]))
    plt.plot(y_values, c='purple', label='softmax(T=0.01)')

    # 把x轴的刻度间隔设置为500
    x_major_locator = MultipleLocator(500)
    # 把y轴的刻度间隔设置为500
    y_major_locator = MultipleLocator(0.05)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # 把x轴的刻度范围设置为0到trial_num
    plt.xlim(0, T)
    # 把y轴的刻度范围设置为0.25到0.4
    plt.ylim(0.25, 0.5)
    plt.xlabel('尝试次数')
    plt.ylabel('平均累积奖赏')
    plt.title('不同算法在2-摇臂赌博机上的性能比较')
    plt.legend()
    plt.savefig('./result.png')
    # plt.show()


if __name__ == '__main__':
    avg_rewards = [exploration_only(),   # 仅探索
                   exploitation_only(),  # 仅利用
                   epsilon_greedy(),     # epsilon-greedy 0.1
                   epsilon_greedy(epsilon=0.01),  # epsilon-greedy 0.01
                   softmax(),  # softmax T=0.1
                   softmax(para_T=0.01)  # softmax T=0.01
                   ]
    # plot result
    plot_result(avg_rewards)
