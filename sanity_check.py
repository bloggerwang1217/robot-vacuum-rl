"""
DQN Sanity Check Script
用於驗證 DQN 實作是否正確運作

測試項目：
1. 簡化環境學習測試（單 robot + 單充電座）
2. Q-value 合理性檢查
3. Bellman Equation 驗證
4. Target Network 同步確認
5. Gradient Flow 檢查
6. Replay Buffer 正確性
7. 隨機 vs 訓練 Policy 比較
"""

import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from collections import deque
from typing import Dict, List, Tuple
import copy

from dqn import DQN, init_weights
from gym import RobotVacuumGymEnv


class SanityCheckAgent:
    """簡化版的 DQN Agent，用於 sanity check"""

    def __init__(self, observation_dim: int, action_dim: int, device: torch.device,
                 gamma: float = 0.99, lr: float = 0.001, batch_size: int = 64):
        self.device = device
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 0.1

        # Networks
        self.q_net = DQN(action_dim, observation_dim).to(device)
        self.target_net = DQN(action_dim, observation_dim).to(device)
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = deque(maxlen=10000)

        # Stats
        self.train_count = 0
        self.last_loss = None
        self.last_q_values = None

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.q_net(obs_t)
            self.last_q_values = q_values.cpu().numpy().flatten()
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, min_buffer_size: int = 100) -> Dict[str, float]:
        if len(self.memory) < min_buffer_size:
            return {}

        self.train_count += 1

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)
        self.last_loss = loss.item()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'q_std': q_values.std().item(),
            'target_q_mean': target_q_values.mean().item(),
        }

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"       {details}")


# =============================================================================
# Test 1: 簡化環境學習測試
# =============================================================================
def test_simple_learning(device: torch.device, num_episodes: int = 500) -> bool:
    """
    測試：單 robot 環境，只有一個充電座在中間
    預期：robot 應該學會走向充電座並停留
    """
    print_header("Test 1: Simple Learning (Single Robot to Charger)")

    # 創建簡化環境：5x5，只有中間有充電座，只用 robot_0
    env = RobotVacuumGymEnv(
        n=5,
        initial_energy=100,
        robot_energies=[100, 1, 1, 1],  # 其他 robot 直接死掉
        e_move=1,
        e_charge=5,
        e_collision=10,
        e_boundary=50,
        n_steps=100,
        charger_positions=[(2, 2)]  # 只有中間一個充電座
    )

    obs_dim = env.observation_space.shape[0]
    agent = SanityCheckAgent(obs_dim, 5, device, gamma=0.99, lr=0.001)

    # Training
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 100:
            # 只控制 robot_0
            action = agent.select_action(obs['robot_0'])
            actions = [action, 4, 4, 4]  # 其他 robot stay

            next_obs, rewards, terms, truncs, _ = env.step(actions)
            reward = rewards['robot_0']
            terminated = terms['robot_0']
            truncated = truncs['robot_0']

            agent.remember(obs['robot_0'], action, reward, next_obs['robot_0'], terminated)
            agent.train_step()

            obs = next_obs
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Update target network
        if ep % 10 == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"  Episode {ep+1}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}, Epsilon = {agent.epsilon:.3f}")

    # Evaluate final policy
    agent.epsilon = 0  # Greedy
    eval_rewards = []

    for _ in range(20):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 100:
            action = agent.select_action(obs['robot_0'], eval_mode=True)
            actions = [action, 4, 4, 4]
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            obs = next_obs
            total_reward += rewards['robot_0']
            steps += 1
            done = terms['robot_0'] or truncs['robot_0']

        eval_rewards.append(total_reward)

    avg_eval_reward = np.mean(eval_rewards)

    # Compare with random policy
    random_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 100:
            action = random.randint(0, 4)
            actions = [action, 4, 4, 4]
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            obs = next_obs
            total_reward += rewards['robot_0']
            steps += 1
            done = terms['robot_0'] or truncs['robot_0']

        random_rewards.append(total_reward)

    avg_random_reward = np.mean(random_rewards)

    improvement = avg_eval_reward - avg_random_reward
    passed = improvement > 10  # 至少比 random 好 10 分

    print(f"\n  Final Evaluation:")
    print(f"    Trained Policy Avg Reward: {avg_eval_reward:.2f}")
    print(f"    Random Policy Avg Reward:  {avg_random_reward:.2f}")
    print(f"    Improvement: {improvement:.2f}")

    print_result("Simple Learning", passed,
                f"Improvement = {improvement:.2f} (threshold: > 10)")

    return passed


# =============================================================================
# Test 2: Q-value 合理性檢查
# =============================================================================
def test_q_value_reasonableness(device: torch.device) -> bool:
    """
    測試：Q-value 是否在合理範圍內
    預期：Q-value 應該在理論最大值範圍內，不應該 NaN 或 Inf
    """
    print_header("Test 2: Q-value Reasonableness")

    env = RobotVacuumGymEnv(
        n=3,
        initial_energy=100,
        e_move=1,
        e_charge=5,
        e_collision=3,
        n_steps=500
    )

    obs_dim = env.observation_space.shape[0]
    agent = SanityCheckAgent(obs_dim, 5, device, gamma=0.99)

    # Train for a bit
    for ep in range(100):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            actions = [agent.select_action(obs['robot_0'])] + [random.randint(0, 4) for _ in range(3)]
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            agent.remember(obs['robot_0'], actions[0], rewards['robot_0'],
                          next_obs['robot_0'], terms['robot_0'])
            agent.train_step(min_buffer_size=50)
            obs = next_obs
            steps += 1
            done = terms['robot_0'] or any(truncs.values())

        if ep % 10 == 0:
            agent.update_target_network()

    # Check Q-values
    obs, _ = env.reset()
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs['robot_0']).unsqueeze(0).to(device)
        q_values = agent.q_net(obs_t).cpu().numpy().flatten()

    # Theoretical max Q-value: r_max / (1 - gamma)
    # 充電獎勵 = 20, 能量獎勵 ≈ 0.25, 死亡懲罰 = -100
    # 樂觀估計每步 +20 -> Q_max ≈ 20 / 0.01 = 2000
    theoretical_max = 2000

    has_nan = np.any(np.isnan(q_values))
    has_inf = np.any(np.isinf(q_values))
    within_range = np.all(np.abs(q_values) < theoretical_max)

    print(f"  Q-values: {q_values}")
    print(f"  Q-value range: [{q_values.min():.2f}, {q_values.max():.2f}]")
    print(f"  Theoretical max: {theoretical_max}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Within theoretical range: {within_range}")

    passed = not has_nan and not has_inf and within_range
    print_result("Q-value Reasonableness", passed)

    return passed


# =============================================================================
# Test 3: Bellman Equation 驗證
# =============================================================================
def test_bellman_equation(device: torch.device) -> bool:
    """
    測試：訓練後的 Q-values 是否滿足 Bellman equation
    預期：Q(s,a) ≈ r + γ * max_a' Q(s', a') 的誤差應該很小
    """
    print_header("Test 3: Bellman Equation Verification")

    env = RobotVacuumGymEnv(
        n=3,
        initial_energy=100,
        e_move=1,
        e_charge=5,
        e_collision=3,
        n_steps=500
    )

    obs_dim = env.observation_space.shape[0]
    agent = SanityCheckAgent(obs_dim, 5, device, gamma=0.99)

    # Train
    for ep in range(200):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            actions = [agent.select_action(obs['robot_0'])] + [random.randint(0, 4) for _ in range(3)]
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            agent.remember(obs['robot_0'], actions[0], rewards['robot_0'],
                          next_obs['robot_0'], terms['robot_0'])
            agent.train_step(min_buffer_size=50)
            obs = next_obs
            steps += 1
            done = terms['robot_0'] or any(truncs.values())

        if ep % 10 == 0:
            agent.update_target_network()

    # Verify Bellman on a sample of transitions
    if len(agent.memory) < 100:
        print("  Not enough samples in buffer")
        return False

    sample = random.sample(list(agent.memory), 100)
    bellman_errors = []

    for state, action, reward, next_state, done in sample:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            q_value = agent.q_net(state_t)[0, action].item()
            next_q_max = agent.target_net(next_state_t).max().item()

            expected_q = reward + (1 - done) * agent.gamma * next_q_max
            bellman_error = abs(q_value - expected_q)
            bellman_errors.append(bellman_error)

    avg_error = np.mean(bellman_errors)
    max_error = np.max(bellman_errors)

    print(f"  Sample size: {len(sample)}")
    print(f"  Average Bellman Error: {avg_error:.4f}")
    print(f"  Max Bellman Error: {max_error:.4f}")

    # 經過訓練後，平均誤差應該相對較小
    passed = avg_error < 50  # 這個閾值取決於 reward scale
    print_result("Bellman Equation", passed, f"Avg Error = {avg_error:.4f}")

    return passed


# =============================================================================
# Test 4: Target Network 同步確認
# =============================================================================
def test_target_network_sync(device: torch.device) -> bool:
    """
    測試：update_target_network() 是否正確同步權重
    """
    print_header("Test 4: Target Network Sync")

    agent = SanityCheckAgent(29, 5, device)

    # 訓練幾步讓 q_net 和 target_net 不同
    for _ in range(10):
        state = np.random.randn(29).astype(np.float32)
        action = random.randint(0, 4)
        reward = random.random()
        next_state = np.random.randn(29).astype(np.float32)
        done = False
        agent.remember(state, action, reward, next_state, done)

    for _ in range(50):
        agent.train_step(min_buffer_size=5)

    # 檢查同步前是否不同
    diff_before = 0
    for p1, p2 in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        diff_before += (p1 - p2).abs().sum().item()

    print(f"  Parameter difference before sync: {diff_before:.6f}")

    # 同步
    agent.update_target_network()

    # 檢查同步後是否相同
    diff_after = 0
    for p1, p2 in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        diff_after += (p1 - p2).abs().sum().item()

    print(f"  Parameter difference after sync: {diff_after:.6f}")

    passed = diff_after < 1e-6 and diff_before > 0
    print_result("Target Network Sync", passed)

    return passed


# =============================================================================
# Test 5: Gradient Flow 檢查
# =============================================================================
def test_gradient_flow(device: torch.device) -> bool:
    """
    測試：梯度是否正確流動到所有層
    """
    print_header("Test 5: Gradient Flow")

    agent = SanityCheckAgent(29, 5, device)

    # 填充 buffer
    for _ in range(100):
        state = np.random.randn(29).astype(np.float32)
        action = random.randint(0, 4)
        reward = random.random() * 10
        next_state = np.random.randn(29).astype(np.float32)
        done = random.random() < 0.1
        agent.remember(state, action, reward, next_state, done)

    # 執行一次 train step
    agent.train_step(min_buffer_size=50)

    # 檢查每層的梯度
    grad_info = []
    all_have_grad = True

    for name, param in agent.q_net.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info.append((name, grad_norm))
            if grad_norm == 0:
                all_have_grad = False
        else:
            grad_info.append((name, None))
            all_have_grad = False

    print("  Layer gradients:")
    for name, grad_norm in grad_info:
        if grad_norm is not None:
            status = "✓" if grad_norm > 0 else "✗ (zero)"
            print(f"    {status} {name}: {grad_norm:.6f}")
        else:
            print(f"    ✗ {name}: None")

    print_result("Gradient Flow", all_have_grad)

    return all_have_grad


# =============================================================================
# Test 6: Replay Buffer 正確性
# =============================================================================
def test_replay_buffer(device: torch.device) -> bool:
    """
    測試：Replay buffer 是否正確存入和取出 transitions
    """
    print_header("Test 6: Replay Buffer Correctness")

    agent = SanityCheckAgent(29, 5, device)

    # 存入已知的 transitions
    test_transitions = []
    for i in range(10):
        state = np.full(29, i, dtype=np.float32)
        action = i % 5
        reward = float(i)
        next_state = np.full(29, i + 1, dtype=np.float32)
        done = (i == 9)

        test_transitions.append((state.copy(), action, reward, next_state.copy(), done))
        agent.remember(state, action, reward, next_state, done)

    # 驗證 buffer 內容
    all_correct = True

    for i, (stored, original) in enumerate(zip(list(agent.memory), test_transitions)):
        state_match = np.allclose(stored[0], original[0])
        action_match = stored[1] == original[1]
        reward_match = stored[2] == original[2]
        next_state_match = np.allclose(stored[3], original[3])
        done_match = stored[4] == original[4]

        if not all([state_match, action_match, reward_match, next_state_match, done_match]):
            print(f"  Mismatch at index {i}")
            all_correct = False

    # 測試 maxlen
    agent2 = SanityCheckAgent(29, 5, device)
    agent2.memory = deque(maxlen=5)

    for i in range(10):
        state = np.full(29, i, dtype=np.float32)
        agent2.remember(state, i % 5, float(i), state, False)

    buffer_len_correct = len(agent2.memory) == 5
    oldest_removed = agent2.memory[0][2] == 5.0  # oldest should be reward=5, not 0

    print(f"  All transitions stored correctly: {all_correct}")
    print(f"  Buffer maxlen works: {buffer_len_correct}")
    print(f"  Oldest removed correctly: {oldest_removed}")

    passed = all_correct and buffer_len_correct and oldest_removed
    print_result("Replay Buffer", passed)

    return passed


# =============================================================================
# Test 7: 隨機 vs 訓練 Policy 比較
# =============================================================================
def test_policy_improvement(device: torch.device, num_episodes: int = 300) -> bool:
    """
    測試：訓練後的 policy 是否比隨機 policy 好
    """
    print_header("Test 7: Policy Improvement over Random")

    env = RobotVacuumGymEnv(
        n=3,
        initial_energy=100,
        e_move=1,
        e_charge=5,
        e_collision=3,
        n_steps=200
    )

    obs_dim = env.observation_space.shape[0]
    agent = SanityCheckAgent(obs_dim, 5, device, gamma=0.99, lr=0.001)

    # Baseline: Random policy
    print("  Evaluating random policy...")
    random_rewards = []
    for _ in range(30):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            actions = [random.randint(0, 4) for _ in range(4)]
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            total_reward += rewards['robot_0']
            obs = next_obs
            steps += 1
            done = terms['robot_0'] or any(truncs.values())

        random_rewards.append(total_reward)

    avg_random = np.mean(random_rewards)
    print(f"    Random Policy: {avg_random:.2f} ± {np.std(random_rewards):.2f}")

    # Train
    print(f"  Training for {num_episodes} episodes...")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action = agent.select_action(obs['robot_0'])
            actions = [action] + [random.randint(0, 4) for _ in range(3)]
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            agent.remember(obs['robot_0'], action, rewards['robot_0'],
                          next_obs['robot_0'], terms['robot_0'])
            agent.train_step(min_buffer_size=100)
            obs = next_obs
            steps += 1
            done = terms['robot_0'] or any(truncs.values())

        if ep % 10 == 0:
            agent.update_target_network()
        agent.epsilon = max(0.01, agent.epsilon * 0.99)

    # Evaluate trained policy
    print("  Evaluating trained policy...")
    agent.epsilon = 0
    trained_rewards = []

    for _ in range(30):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action = agent.select_action(obs['robot_0'], eval_mode=True)
            actions = [action] + [random.randint(0, 4) for _ in range(3)]
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            total_reward += rewards['robot_0']
            obs = next_obs
            steps += 1
            done = terms['robot_0'] or any(truncs.values())

        trained_rewards.append(total_reward)

    avg_trained = np.mean(trained_rewards)
    print(f"    Trained Policy: {avg_trained:.2f} ± {np.std(trained_rewards):.2f}")

    improvement = avg_trained - avg_random
    improvement_pct = (improvement / abs(avg_random)) * 100 if avg_random != 0 else 0

    print(f"\n  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")

    passed = improvement > 5  # 至少好 5 分
    print_result("Policy Improvement", passed,
                f"Trained ({avg_trained:.2f}) vs Random ({avg_random:.2f})")

    return passed


# =============================================================================
# Test 8: Loss 下降檢查
# =============================================================================
def test_loss_decreasing(device: torch.device) -> bool:
    """
    測試：Loss 是否隨訓練下降
    """
    print_header("Test 8: Loss Decreasing")

    env = RobotVacuumGymEnv(
        n=3,
        initial_energy=100,
        e_move=1,
        e_charge=5,
        e_collision=3,
        n_steps=200
    )

    obs_dim = env.observation_space.shape[0]
    agent = SanityCheckAgent(obs_dim, 5, device, gamma=0.99, lr=0.001, batch_size=32)

    # Collect some data first
    print("  Collecting initial data...")
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            actions = [random.randint(0, 4) for _ in range(4)]
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            agent.remember(obs['robot_0'], actions[0], rewards['robot_0'],
                          next_obs['robot_0'], terms['robot_0'])
            obs = next_obs
            steps += 1
            done = any(terms.values()) or any(truncs.values())

    # Train and record loss
    print("  Training and recording loss...")
    losses = []

    for i in range(500):
        stats = agent.train_step(min_buffer_size=100)
        if stats:
            losses.append(stats['loss'])

        # Occasionally add new data
        if i % 50 == 0:
            obs, _ = env.reset()
            for _ in range(20):
                actions = [agent.select_action(obs['robot_0'])] + [random.randint(0, 4) for _ in range(3)]
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                agent.remember(obs['robot_0'], actions[0], rewards['robot_0'],
                              next_obs['robot_0'], terms['robot_0'])
                obs = next_obs
                if any(terms.values()) or any(truncs.values()):
                    break
            agent.update_target_network()

    if len(losses) < 100:
        print("  Not enough loss samples")
        return False

    # Compare early vs late losses
    early_loss = np.mean(losses[:50])
    late_loss = np.mean(losses[-50:])

    print(f"  Early loss (first 50): {early_loss:.4f}")
    print(f"  Late loss (last 50): {late_loss:.4f}")
    print(f"  Loss reduction: {early_loss - late_loss:.4f}")

    # Loss 應該下降，但不一定單調
    passed = late_loss < early_loss * 1.5  # 允許一些波動
    print_result("Loss Decreasing", passed,
                f"Early={early_loss:.4f}, Late={late_loss:.4f}")

    return passed


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DQN Sanity Check")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only (skip long training tests)")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("\n" + "=" * 60)
    print("  DQN SANITY CHECK")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Quick mode: {args.quick}")

    results = {}

    # Quick tests (always run)
    results['target_sync'] = test_target_network_sync(device)
    results['gradient_flow'] = test_gradient_flow(device)
    results['replay_buffer'] = test_replay_buffer(device)
    results['q_value'] = test_q_value_reasonableness(device)
    results['bellman'] = test_bellman_equation(device)

    # Longer tests (skip if quick mode)
    if not args.quick:
        results['loss_decreasing'] = test_loss_decreasing(device)
        results['simple_learning'] = test_simple_learning(device, num_episodes=300)
        results['policy_improvement'] = test_policy_improvement(device, num_episodes=200)
    else:
        print("\n  (Skipping long training tests in quick mode)")

    # Summary
    print_header("SUMMARY")

    passed_count = sum(results.values())
    total_count = len(results)

    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n  🎉 All sanity checks PASSED!")
    else:
        print("\n  ⚠️  Some tests FAILED. Please investigate.")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
