# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions, input_dim):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########

        # MLP1
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

        # MLP2
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)

class DQN_CNN(nn.Module):
    def __init__(self, num_actions, input_channels):
        super(DQN_CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)

class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
    
    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ##########
        # Compute priority with the Bellman error (p_i = |δ_i| + epsilon)
        if error is None:
            priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        else:
            priority = abs(error) + 1e-10

        # Add the transition to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition  # Replace the oldest transition if buffer is full
        
        # Store the priority
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity  # Circular buffer
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        # Compute the sampling probabilities P(i) based on the priority
        priorities = self.priorities[:len(self.buffer)]  # Get priorities for the stored transitions
        probabilities = priorities ** self.alpha

        
        # Add a small epsilon to avoid division by zero
        probabilities += 1e-10  # Small epsilon to avoid zero sum

        probabilities /= probabilities.sum()  # Normalize to get probabilities

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Retrieve the sampled transitions
        batch = [self.buffer[idx] for idx in indices]
        
        # Compute importance-sampling (IS) weights
        weights = (1 / len(self.buffer)) * (1 / probabilities[indices]) ** self.beta
        weights /= weights.max()  # Normalize the weights

        return batch, indices, weights
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########

        # Update priorities based on the Bellman errors
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-10  # Recompute priority with the new error
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        

class DQNAgent:
    def __init__(self, args=None):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.env_name = args.env_name
        self.ddqn = args.ddqn
        self.per = args.per
        self.n_steps = args.n_steps
        self.n = args.n
        self.n_step_buffer = deque(maxlen=self.n) if self.n_steps else None
        self.gradient_clipping = args.gradient_clipping
        self.max_norm = args.max_norm

        print("Using device:", self.device)
        self.env = gym.make(self.env_name, render_mode="rgb_array")
        self.test_env = gym.make(self.env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        # Determine if we are using CartPole or Pong
        if "ALE/Pong-v5" in self.env_name:
            self.preprocessor = AtariPreprocessor()
            self.state_dim = 4  # For Pong, we use stacked frames of size (84x84x4)

            self.q_net = DQN_CNN(self.num_actions, self.state_dim).to(self.device)
            self.target_net = DQN_CNN(self.num_actions, self.state_dim).to(self.device)
            self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
            
        else:
            self.preprocessor = None
            self.state_dim = self.env.observation_space.shape[0]  # CartPole state space has 4 dimensions (position, velocity, angle, angular velocity)
            self.q_net = DQN(self.num_actions, self.state_dim).to(self.device)
            self.target_net = DQN(self.num_actions, self.state_dim).to(self.device)
            self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong

        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # Learning rate scheduling
        self.lr_decay = args.lr_decay
        self.initial_lr = args.lr
        self.final_lr = args.final_lr
        self.lr_decay_env_step = args.lr_decay_env_step

        if self.per:
            self.memory = PrioritizedReplayBuffer(capacity=args.memory_size, alpha=args.per_alpha, beta=args.per_beta)
            self.beta_increment = (1.0 - args.per_beta) / args.per_beta_annealing_env_step
        else:
            # Initialize the memory buffer with specified size
            self.memory = deque(maxlen=args.memory_size)  

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state): # epsilon greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            # Handle state based on environment
            if "ALE/Pong-v5" in self.env.spec.id:
                state = self.preprocessor.reset(obs)  # Preprocess Pong frames
            else:
                state = obs  # CartPole state is already a 4D vector

            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if "ALE/Pong-v5" in self.env.spec.id:
                    next_state = self.preprocessor.step(next_obs)  # Preprocess Pong frames
                else:
                    next_state = next_obs  # For CartPole, just use the next state as is
                
                ########## Enhanced DQN  ##########
                if self.n_steps:# Accumulate transitions for n-step
                    self.n_step_buffer.append((state, action, reward, next_state, done))
                    
                    if len(self.n_step_buffer) == self.n or done:
                        cum_reward = 0
                        for idx, (_, _, r, _, _) in enumerate(self.n_step_buffer):
                            cum_reward += (self.gamma ** idx) * r
                        
                        start_state, start_action, _, _, _ = self.n_step_buffer[0]
                        end_next_state, _, _, end_next_state_obs, end_done = self.n_step_buffer[-1]
                
                        if self.per: # PER uses add() method
                            self.memory.add((start_state, start_action, cum_reward, end_next_state_obs, end_done))
                        else:
                            self.memory.append((start_state, start_action, cum_reward, end_next_state_obs, end_done))
                        
                        if done:
                            self.n_step_buffer.clear()
                        else:
                            self.n_step_buffer.popleft()
                else:
                    # Single step storage:  PER uses add() method
                    if self.per:
                        self.memory.add((state, action, reward, next_state, done))
                    else:
                        self.memory.append((state, action, reward, next_state, done))
                ########## Enhanced DQN  ##########


                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                
                ########## PER ##########
                if self.per: # Follow the originla paper: increase the beta to 1 according to time
                    self.memory.beta = min(1.0, self.memory.beta + self.beta_increment)
                ########## PER ##########

                ########## Learning rate schedule ##########
                if self.lr_decay:
                    if self.env_count <= self.lr_decay_env_step:
                        lr = self.initial_lr - (self.initial_lr - self.final_lr) * (self.env_count / self.lr_decay_env_step)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                ########## Learning rate schedule ##########
                
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                ########## YOUR CODE HERE  ##########
                # Add additional wandb logs for debugging if needed
                if self.per:
                    wandb.log({"beta":self.memory.beta})
                ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward

                    model_path = os.path.join(self.save_dir, "best_model.pt")

                    if self.env_name == "ALE/Pong-v5":
                        if self.env_count <= 200000:
                            model_path = os.path.join(self.save_dir, "best_model_200000.pt")
                        elif self.env_count <= 400000:
                            model_path = os.path.join(self.save_dir, "best_model_400000.pt")
                        elif self.env_count <= 600000:
                            model_path = os.path.join(self.save_dir, "best_model_600000.pt")
                        elif self.env_count <= 800000:
                            model_path = os.path.join(self.save_dir, "best_model_800000.pt")
                        elif self.env_count <= 1000000:
                            model_path = os.path.join(self.save_dir, "best_model_1000000.pt")
                        
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        self.q_net.eval()

        obs, _ = self.test_env.reset()
        if "ALE/Pong-v5" in self.test_env.spec.id:
            state = self.preprocessor.reset(obs)  # Preprocess Pong frames
        else:
            state = obs  # For CartPole, state is already 4D
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if "ALE/Pong-v5" in self.test_env.spec.id:
                state = self.preprocessor.step(next_obs)  # Preprocess Pong frames
            else:
                state = next_obs  # For CartPole, just use the next state as is

        self.q_net.train()

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer

        if self.per:
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates

        if self.ddqn:  # If DDQN is enabled but Multi-step return is not
            # Choose the action for the next state using the current Q-network
            next_actions = self.q_net(next_states).argmax(1)  # Choose the action with max Q-value
        
            # Get Q-values for next states using the target network
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else: 
            next_q_values = self.target_net(next_states).max(1)[0]  # Get the max Q-value for next states
        
        if self.n_steps:       
            target_q_values = rewards + (self.gamma ** self.n) * next_q_values * (1 - dones)
        else:
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))  # Bellman target

        ########## PER switch ##########
        if self.per:
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            loss_unreduced = (q_values - target_q_values.detach()) ** 2
            loss = (loss_unreduced * weights).mean()

            bellman_error = abs(target_q_values - q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, bellman_error)
        else:
            loss = nn.MSELoss()(q_values, target_q_values.detach())
        ########## PER switch ##########
    
        # Backpropagate and optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_norm)

        self.optimizer.step()

        ########## End of PER ##########
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({"Loss": loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--ddqn", type=bool, default=False)
    parser.add_argument("--per", type=bool, default=False)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--per-beta-annealing-env-step", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=bool, default=False)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--gradient-clipping", type=bool, default=False)
    parser.add_argument("--max-norm", type=float, default=10.0)
    parser.add_argument("--lr-decay", type=bool, default=False)
    parser.add_argument("--final-lr", type=float, default=2e-5)
    parser.add_argument("--lr-decay-env-step", type=int, default=500000)
    args = parser.parse_args()

    if args.env_name == "CartPole-v1":
        wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    else:
        wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()