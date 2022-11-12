"""
使用随机策略玩21点游戏
"""
import gym
import numpy as np

# 创建环境
env = gym.make('Blackjack-v1')


def print_observation(observation, reward=0):
    """
    打印observation的信息：
        1、玩家现有的点数
        2、是否使用usable Ace（若为1则表示Ace被算作11点，为0则表示Ace被算作1点）
        3、庄家露出的牌面点数
    """
    player_score, dealer_score, usable_ace = observation
    print('Player Score:{}(Usable Ace:{}), Dealer Score:{}, Reward:{}'.format(player_score,
                                                                              usable_ace,
                                                                              dealer_score,
                                                                              reward))


def random_policy(episode_num):
    """
    按随机策略玩21点游戏
    """
    for i_episode in range(episode_num):
        print('\nRound Number:', i_episode)
        # 初始化环境
        observation, _ = env.reset()
        reward = 0

        while True:
            print_observation(observation, reward)
            # 产生随机动作:0,表示停牌stick；1，表示继续要牌hit
            action = np.random.choice(env.action_space.n)
            print('Taking action: {}'.format(["Stick（不要）", "Hit（要）"][action]))
            # 当玩家继续加牌时，需要判读是否超21点，若没有超过的话，返回下一状态，同时reward为0，等待下一个step方法；
            # 当玩家停止叫牌时，按照庄家策略：小于17时叫牌，游戏终局时产生+1表示玩家获胜，-1表示庄家获胜。
            observation, reward, done, _, _ = env.step(action)

            if done:
                print_observation(observation, reward)
                if reward == 1:
                    result = 'Win'
                elif reward == 0:
                    result = 'Draw'
                else:
                    result = 'Loss'
                print('Game end. Reward:{}, Result:{}\n'.format(reward, result))
                env.close()
                break


if __name__ == "__main__":
    random_policy(episode_num=5)


