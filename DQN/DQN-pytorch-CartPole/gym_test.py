"""
CartPole简介见https://www.gymlibrary.dev/environments/classic_control/cart_pole/
"""
import gym

if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = gym.make(env_id, render_mode='human')
    print('env.action_space.n: {}'.format(env.action_space.n))
    print('env.observation_space.shape: {}'.format(env.observation_space.shape))

    observation, _ = env.reset()
    print('state: {}'.format(observation))

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, _, _ = env.step(action)
        print('action: {}, reward: {}'.format(action, reward))
        print('next_state: {}'.format(observation))

        if terminated:
            observation, _ = env.reset()
            print('-' * 80)
            print('Restart...')
            print('state: {}'.format(observation))

    env.close()
