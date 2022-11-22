from model import DQN, ConvDQN
import torch
from buffer import BasicBuffer
import torch.nn as nn
import numpy as np

class DQNAgent(object):
    def __init__(self, env, use_conv=True, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        if use_conv:
            self.model = ConvDQN(self.env.observation_space.shape, self.env.action_space.n)
        else:
            self.model = DQN(self.env.observation_space.shape, self.env.action_space.n)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, epsilon=0.2):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        if np.random.randn() < epsilon:
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

