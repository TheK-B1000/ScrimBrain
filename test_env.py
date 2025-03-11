import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import cv2
import os
import platform

# Define the neural network for the A3C-inspired agent
class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # CartPole state is 4D
        self.fc2 = nn.Linear(128, 64)
        self.fc_actor = nn.Linear(64, action_size)  # Action values
        self.fc_critic = nn.Linear(64, 1)  # State value

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_values = self.fc_actor(x)
        state_value = self.fc_critic(x)
        return action_values, state_value


# Define the learning agent
class LearningAgent:
    def __init__(self, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0005)  # Lowered learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Add epsilon-greedy exploration

    def act(self, state):
        if np.random.rand() < self.epsilon:  # Epsilon-greedy exploration
            return [np.random.randint(self.action_size)]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_values, _ = self.network(state)
        policy = F.softmax(action_values, dim=-1)
        action = np.random.choice(self.action_size, p=policy.detach().cpu().numpy()[0])
        return [action]

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        action_values, state_value = self.network(state)
        _, next_state_value = self.network(next_state)

        target = reward + self.gamma * next_state_value * (1 - done)
        advantage = (target - state_value).detach()

        probs = F.softmax(action_values, dim=-1)
        log_probs = F.log_softmax(action_values, dim=-1)
        action_log_prob = log_probs[0, action[0]]
        actor_loss = -action_log_prob * advantage

        critic_loss = F.mse_loss(state_value, target.detach())
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
print("Environment created successfully!")


# Function to run the agent, train it, and save a longer video
def show_video_of_model(agent, env, num_episodes=10, max_steps_per_episode=500):
    frames = []
    total_steps = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps_per_episode):
            frame = env.render()
            frame = cv2.resize(frame, (608, 400), interpolation=cv2.INTER_AREA)  # Resize for FFmpeg
            frames.append(frame)
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action[0])
            agent.train(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            if done:
                break
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Steps: {step + 1}")

    env.close()

    # Save video with FFmpeg
    writer = imageio.get_writer('video.mp4', fps=30, format='FFMPEG', codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    # Open the video with the default media player
    video_path = os.path.abspath('video.mp4')
    if platform.system() == 'Windows':
        os.startfile(video_path)
    elif platform.system() == 'Darwin':  # macOS
        os.system(f'open "{video_path}"')
    else:  # Linux
        os.system(f'xdg-open "{video_path}"')

    print(f"Total frames: {len(frames)}, Estimated video length: {len(frames) / 30:.2f} seconds")


# Create the agent and generate the video
agent = LearningAgent(action_size=2)  # CartPole has 2 actions: left (0), right (1)
show_video_of_model(agent, env, num_episodes=10, max_steps_per_episode=500)
print("Video saved as 'video.mp4' and opened in your default media player")