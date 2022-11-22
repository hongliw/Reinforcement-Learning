import gym
from agent import DQNAgent


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for i_episode in range(1, max_episodes + 1):
        state,_ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print('Episode {}: {}'.format(i_episode, episode_reward))
                break

            state = next_state

    return episode_rewards


if __name__ == '__main__':
    MAX_EPISODES = 1000
    MAX_STEPS = 500
    BATCH_SIZE = 32

    env = gym.make('CartPole-v1', render_mode='human')
    agent = DQNAgent(env, use_conv=False)
    episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

