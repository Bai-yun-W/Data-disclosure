import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义 DQN 代理
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=64, replay_buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        # Q 网络和目标网络
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # 初始化目标网络权重
        self.target_network.eval()  # 目标网络不需要训练

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.device = device

    def act(self, state, epsilon):
        # epsilon-greedy 策略
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)  # 随机动作
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # 缓冲区不足时不训练

        # 从缓冲区中随机抽样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 当前 Q 值
        q_values = self.q_network(states).gather(1, actions)

        # 目标 Q 值
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # 计算损失并优化
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # 将当前 Q 网络的权重复制到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())

# 主程序
if __name__ == "__main__":
    # 使用 OpenAI Gym 的 CartPole 环境
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 创建 DQN 代理
    agent = DQNAgent(state_size, action_size)

    episodes = 500
    max_steps = 200
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    target_update_frequency = 10

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(max_steps):
            action = agent.act(state, epsilon)
            result = env.step(action)
            next_state, reward, done = result  # 新版本 Gym


            # 存储经验
            agent.store_experience(state, action, reward, next_state, done)

            # 训练代理
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        # 更新目标网络
        if episode % target_update_frequency == 0:
            agent.update_target_network()

        # 减少 epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

    env.close()
