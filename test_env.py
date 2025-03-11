import gymnasium as gym
env = gym.make("CartPole-v1")
print("Environment created successfully!")
env.close()