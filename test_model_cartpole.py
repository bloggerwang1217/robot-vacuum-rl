import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import imageio
import os
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()

        #MLP1
        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, num_actions)
        # )
        #MLP2
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # 128 units in the first hidden layer
            nn.ReLU(),
            nn.Linear(128, 256),        # 256 units in the second hidden layer
            nn.ReLU(),
            nn.Linear(256, 256),        # 256 units in the third hidden layer
            nn.ReLU(),
            nn.Linear(256, 128),        # 128 units in the fourth hidden layer
            nn.ReLU(),
            nn.Linear(128, num_actions) # Output layer with number of actions
        )

    def forward(self, x):
        return self.network(x)

# Preprocessing for CartPole - no need for image processing
class SimplePreprocessor:
    def __init__(self):
        pass

    def reset(self, obs):
        return obs  # CartPole state is already a 4D vector

    def step(self, obs):
        return obs  # No processing needed

def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use the CartPole-v1 environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = SimplePreprocessor()  # No image preprocessing for CartPole
    num_actions = env.action_space.n

    model = DQN(4, num_actions).to(device)  # Input dimension is 4 for CartPole state
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    all_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
            frame_idx += 1

        all_rewards.append(total_reward)

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")

    # After all episodes, calculate mean and std deviation
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    print("\n========== Evaluation Summary ==========")
    print(f"Mean Reward over {args.episodes} episodes: {mean_reward:.2f}")
    print(f"Std Deviation of Reward: {std_reward:.2f}")
    print("=========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)
