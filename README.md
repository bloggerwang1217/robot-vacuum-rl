# DQN and Reward Mechanism Analysis

This document aims to clarify the operation of the DQN (Deep Q-Network) agent in the `robot-vacuum-rl` project, with a focus on its reward function design and implementation details.

---

## 1. Project Goal

The core research question of this project is: When agents are given only a simple "survival" goal (managing energy, avoiding death), will they spontaneously learn "harmful behaviors" towards other agents (e.g., depleting others' energy through collisions) in order to maximize their own survival probability in a resource-limited environment?

---

## 2. Core Component Overview

The entire system consists of two main parts: the **Environment** and the **Agent**, which constantly interact.

-   **Environment**: Provides the agent with the current "state" and gives a "reward" based on the agent's "action".
-   **Agent**: Decides which "action" to perform based on the "state" and its own strategy.

Their relationship and corresponding code files are as follows:

| Role      | Function                        | Main Files                         |
| :-------- | :------------------------------ | :--------------------------------- |
| **Environment** | Defines world rules, state, reward | `robot_vacuum_env.py` (Physics Engine), `gym.py` (RL Interface) |
| **Agent** | Defines neural network, decision, learning | `dqn.py` (Brain Structure), `train_dqn.py` (Decision & Learning) |
| **Training Script** | Connects environment and agents | `train_dqn.py` (Training Loop)     |

---

## 3. Environment Details (The Environment)

### 3.1 Physics Rules (Physics Engine - `robot_vacuum_env.py`)

The underlying operation of the environment defines how robots interact with the world.

-   **Actions**: 5 discrete actions (UP, DOWN, LEFT, RIGHT, STAY).
-   **Interaction Logic**:
    1.  **Movement and Collision**: If a robot moves successfully, it consumes `e_move` energy. If the movement fails (due to a collision), it remains in place and consumes different collision energy depending on the situation. The final collision rules are:
        *   **Rule 1 - Boundary Collision**: The moving party suffers `e_collision_active_one_sided` damage.
        *   **Rule 2 - Active vs. Stationary Collision**: The moving "attacker" suffers `e_collision_active_one_sided` damage; the stationary "victim" suffers `e_collision_passive` damage.
        *   **Rule 3 - Simultaneous Move to Same Cell**: Both "parties" in the conflict suffer `e_collision_active_two_sided` damage.
        *   **Rule 4 - Swapping Positions**: Both "parties" in the conflict suffer `e_collision_active_two_sided` damage.
    2.  **Charging**: Staying on a charging station (`STAY` action) restores `e_charge` energy.
    3.  **Death**: When `energy <= 0`, the robot's `is_active` status becomes `False`.

### 3.2 Agent's Perspective (`gym.py`)

`gym.py` translates physical rules into "observations" and "rewards" that the agent can understand.

#### From Physical Events to Rewards

The `_calculate_rewards` function in `gym.py` acts as a "translator." It observes state changes from one step to the next and "translates" these physical consequences into an abstract score (reward) that the RL agent can understand. The rules are as follows:

1.  **Energy Change Translation**: All physical events (movement, collision, charging) causing energy increase or decrease are translated proportionally by `* 0.01` into a small reward or penalty.
2.  **Extra Reward for "Charging Behavior"**: To specifically encourage the critical survival **action** of charging, an additional fixed reward of `+0.5` is given, beyond the energy increase.
3.  **Huge Penalty for "Death Event"**: At the moment energy runs out and the state changes from "alive" to "dead", a huge penalty of `-5.0` is given. This strong signal makes the agent learn to avoid death at all costs.
4.  **Small Reward for "Survival State"**: As long as the agent is alive, it receives a tiny `+0.01` "survival bonus" for each step it remains active, encouraging it to prolong its life.

#### Reward Calculation Example

Assuming default energy settings (`e_charge=5`, `e_collision_*=3`):

**Scenario 1: Successful Charging**
A robot is on a charging station and chooses the "STAY" action.
-   **Physical Event**: Energy `+5`, charge count `+1`, remains active.
-   **Reward Calculation**:
    1.  Energy Change: `+5 * 0.01 = +0.05`
    2.  Charging Action: `+0.5`
    3.  Death Event: `+0`
    4.  Survival State: `+0.01`
-   **Final Reward**: `0.05 + 0.5 + 0.01 = +0.56`

**Scenario 2: Death After Wall Collision**
A robot with only 2 energy points moves and hits a boundary.
-   **Physical Event**: Energy `-3` (becomes 0), status changes from "alive" to "dead". Actual energy change is `-2`.
-   **Reward Calculation**:
    1.  Energy Change: `-2 * 0.01 = -0.02`
    2.  Charging Action: `+0`
    3.  Death Event: `-5.0`
    4.  Survival State: `+0` (because dead at end of step)
-   **Final Reward**: `-0.02 - 5.0 = -5.02`

#### Reward Function

This is the code implementation of the above conversion rules, and it is key to guiding the agent's learning.

```python
# From gym.py -> _calculate_rewards
def _calculate_rewards(self, state):
    # ...
    for i in range(self.n_robots):
        robot = state['robots'][i]
        prev_robot = self.prev_robots[i]
        reward = 0.0

        # 1. Energy change reward (implicitly covers movement/collision penalties)
        energy_delta = robot['energy'] - prev_robot['energy']
        reward += energy_delta * 0.01

        # 2. Charging bonus (extra encouragement)
        if robot['charge_count'] > prev_robot['charge_count']:
            reward += 0.5

        # 3. Death penalty (heavy penalty for transitioning from alive to dead)
        if not robot['is_active'] and prev_robot['is_active']:
             reward -= 5.0

        # 4. Survival bonus (small positive reward for every step alive)
        if robot['is_active']:
            reward += 0.01

        rewards[agent_id] = reward
    return rewards
```
#### Observation Space

After defining the "goal" (reward), we need the agent to "see" the world to make decisions. The agent's "view" is a **20-dimensional normalized vector** containing global information from its own perspective. Normalization is crucial for stable neural network learning.

**Coordinate System Explanation**:
-   Origin `(0,0)` is at the **top-left corner**.
-   Coordinates are unified as `(x, y)`, where `x` is the horizontal position (column), and `y` is the vertical position (row).
-   Down is positive, Up is negative; Right is positive, Left is negative.

**Scenario Example**:
Assume Robot 0 is at `(1,1)` with energy `75`. Other robots are at `(0,1)` with energy `20` (ID=1), `(2,1)` with energy `90` (ID=2), and `(1,0)` with energy `5` (ID=3). The position map is as follows:

```
  (x) 0   1   2
(y)
 0    .   1   .
 1    3   0   .
 2    .   2   .
```

**Observation Vector Example (based on `n=3`, `initial_energy=100` scenario)**:
Based on the above scenario, Robot 0's observation vector (after normalization) is:

**Normalization Formulas**:
-   **Own Position**: `pos_norm = pos_abs / (n - 1)`, mapping values from `[0, n-1]` to `[0, 1]`.
-   **Own/Other's Energy**: `energy_norm = energy_current / initial_energy`, mapping values from `[0, initial_energy]` to `[0, 1]`.
-   **Relative Position**: `delta_pos_norm = (pos_other - pos_self) / (n - 1)`, mapping values from `[-(n-1), n-1]` to `[-1, 1]`.

```
np.array([
    # Own Position (1,1) -> (0.5, 0.5)
    0.5, 0.5,
    # Own Energy 75 -> 0.75
    0.75,
    # Robot 1 (0,1) Relative to Self (1,1) -> (-1,0) Energy 20 -> (-0.5, 0.0, 0.2)
    -0.5, 0.0, 0.2,
    # Robot 2 (2,1) Relative to Self (1,1) -> (1,0) Energy 90 -> (0.5, 0.0, 0.9)
    0.5, 0.0, 0.9,
    # Robot 3 (1,0) Relative to Self (1,1) -> (0,-1) Energy 5 -> (0.0, -0.5, 0.05)
    0.0, -0.5, 0.05,
    # Charger 1 (0,0) Relative to Self (1,1) -> (-1,-1) -> (-0.5, -0.5)
    -0.5, -0.5,
    # Charger 2 (0,2) Relative to Self (1,1) -> (-1,1) -> (-0.5, 0.5)
    -0.5, 0.5,
    # Charger 3 (2,0) Relative to Self (1,1) -> (1,-1) -> (0.5, -0.5)
    0.5, -0.5,
    # Charger 4 (2,2) Relative to Self (1,1) -> (1,1) -> (0.5, 0.5)
    0.5, 0.5
], dtype=np.float32)
```

---

## 4. DQN Agent Details (The DQN Agent)

This project features a multi-agent environment, and the strategy we employ is **"Independent DQN (IDQN)"**.

**Core Idea**: An entirely independent DQN agent is created for each robot. This means each robot has its own neural network and dedicated experience replay buffer. Their only connection is that they **share the same environment**. When any robot moves, this change is immediately reflected in **the observation vectors of all other robots** (because the vector contains the relative positions and energy of everyone). Therefore, each robot's DQN learns how to interpret this 20-dimensional vector containing "information about others," thereby forming its own "perception" of this multi-agent world and making decisions most beneficial to itself.

This logic is encapsulated in the `IndependentDQNAgent` class in `train_dqn.py`. The next three subsections will delve into the internal structure of this independent agent.

### 4.1 Model Architecture (`dqn.py`)

The agent's "brain" is a Multi-Layer Perceptron (MLP). It receives a 20-dimensional state observation vector and outputs Q-values corresponding to 5 actions. Its structure is shown below:

```text
+--------------------------------+
|  Input (Observation Vector)    |
|  shape: [batch_size, 20]       |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(20, 128) + ReLU        |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(128, 256) + ReLU       |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(256, 256) + ReLU       |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(256, 128) + ReLU       |
+--------------------------------+
               |
               v
+--------------------------------+
|  Linear(128, 5)                |
+--------------------------------+
               |
               v
+--------------------------------+
|  Output (Q-values for each action) |
|  shape: [batch_size, 5]        |
+--------------------------------+
```

The corresponding PyTorch code is as follows:
```python
# From dqn.py
class DQN(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), # input_dim = 20
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions) # num_actions = 5
        )

    def forward(self, x):
        return self.network(x)
```

### 4.2 Decision Mechanism (Action Selection - Epsilon-Greedy)

The agent employs an Epsilon-Greedy strategy to balance "exploration" and "exploitation".

```python
# From train_dqn.py -> IndependentDQNAgent.select_action
def select_action(self, observation):
    # Explore with probability epsilon
    if random.random() < self.epsilon:
        return random.randint(0, self.action_dim - 1)
    # Exploit with probability (1-epsilon) by choosing the action with the highest Q-value (Greedy)
    state_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
    with torch.no_grad():
        # 1. Q-Network outputs Q-values for all actions
        q_values = self.q_net(state_tensor)
    # 2. Select the index of the action with the highest Q-value
    return q_values.argmax().item()
```

### 4.3 Experience Replay

To break the temporal correlation between experiences and improve sample efficiency, we use the "Experience Replay" technique. This acts like a "memory palace" for the agent.

-   **Remember**: After each step of interaction, the agent doesn't learn immediately. Instead, it stores the complete experience `(state, action, reward, next_state, done)` into a fixed-size "replay buffer".
-   **Storage Capacity**: The default replay buffer size is **100,000** experiences (controlled by the `memory_size` parameter).
-   **Start Training**: The agent does not learn from the very beginning. In the early stages of training, `epsilon` is high (default 1.0), and the agent primarily engages in **random exploration**. It first performs these random actions to collect experiences until at least **1,000** experiences (controlled by the `replay_start_size` parameter) have accumulated in the replay buffer. Only then does it start sampling from the buffer and updating the neural network.
-   **Forgetting**: When the buffer is full, the oldest memories are automatically discarded (following a First-In-First-Out, FIFO, principle).

**Psychological Analogy**: The Replay Buffer is very similar to "Episodic Long-Term Memory" in psychology. It stores the agent's past specific experiences and events (e.g., `(S, A, R, S')` at a certain time), rather than general knowledge or skills (which are more akin to the neural network's weights).

```python
# From train_dqn.py -> IndependentDQNAgent
# Initialize Replay Buffer in __init__
self.memory = deque(maxlen=args.memory_size)

# Method to store memories
def remember(self, state, action, reward, next_state, done):
    """Store experience to replay buffer"""
    self.memory.append((state, action, reward, next_state, done))
```

### 4.4 Learning Update and Bellman Error

Once enough experiences have accumulated in the Replay Buffer (default `replay_start_size = 1000`), the agent begins to learn. The core of learning is to minimize the **Bellman Error**.

-   **Bellman Equation**: This is the theoretical foundation of Q-Learning, defining the recursive relationship that optimal Q-values should satisfy.
-   In DQN, we simplify this into a **Temporal Difference (TD) Target**.
-   **Bellman Error** (or TD Error) refers to the difference between `(TD Target - current predicted Q-value)`.
-   **Loss Function**: Its purpose is to calculate the **Mean Squared Error (MSE)** of this error. The goal of training is to update the neural network's weights to minimize this Loss.

#### Intuitive Explanation: Why this formula?

This formula can be compared to "playing chess" to understand the "true value" of a good move.

> $$ y = r + \gamma \cdot \max_{a'} Q(s', a') $$

The "true value" of an action (`y`) consists of two parts:

1.  **Immediate Benefit (`r`)**:
    *   **Chess Analogy**: If I make this move, can I **immediately capture** an opponent's pawn?
    *   **Robot Scenario**: By executing this action, did I **immediately gain** a charging reward, or **immediately lose** energy due to a collision?

2.  **Future Potential (`γ * max_a' Q(s', a')`)**:
    *   `max_a' Q(s', a')` refers to: After making this move and reaching a new state `s'`, what is the potential of the **best subsequent move** my side can make?
    *   `γ` (gamma) is the **degree of importance given to the future** (discount factor). A higher `γ` means the robot is more "far-sighted".

Therefore, the meaning of the entire formula is:
> **True Value of an Action = Immediate Benefit + Discounted Future Potential**

The learning objective of DQN is to make our neural network's predictions increasingly close to this "true value".

---
The entire learning process is encapsulated in the `train_step` function, which uses "Experience Replay" and "Target Network" techniques to stabilize training.

```python
# From train_dqn.py -> IndependentDQNAgent.train_step (Detailed Annotation Version)
def train_step(self):
    # 1. Randomly sample a batch of experiences from the Replay Buffer (default batch_size = 128)
    batch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # --- Core Update Steps ---

    # 2. Calculate Q(s,a)
    #    Use the "main network q_net" to calculate the predicted Q-value for the action 'a' that was "actually taken in the past" in the batch.
    q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 3. Calculate TD Target y = r + γ * max_a' Q(s', a')
    with torch.no_grad(): # Target network calculations do not require gradient tracking
        # Use the "target network target_net" to predict the maximum Q-value for the next state s'.
        next_q_values = self.target_net(next_states).max(1)[0]
        # Calculate the TD Target according to the Bellman Equation.
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

    # 4. Calculate Loss, which is the Mean Squared Error (MSE) of the Bellman Error
    #    Loss = mean( (target_q_values - q_values)^2 )
    loss = nn.MSELoss()(q_values, target_q_values)

    # 5. Update the weights of the "main network q_net" via backpropagation to minimize the Loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

#### Role of the Target Network

You might ask: Why is a separate `target_net` used to calculate the TD Target, instead of directly using `q_net`?

This is the second magic ingredient for stabilizing DQN training (the first being Experience Replay).

-   **Problem**: If both the target calculation and the prediction calculation use the same `q_net` (which changes at every step), it's like **"trying to hit a moving target while calibrating your gun with that same moving target."** This causes the learning target (TD Target) to constantly shift, leading to very unstable training and making it difficult for the model to converge.
-   **Solution**: The `target_net` acts as a **"fixed target."** It is a copy of `q_net`, but its weights are **not** updated at every step. Instead, its weights are **completely copied** from `q_net` only every N steps (controlled by the `target_update_frequency` parameter). This way, when calculating the TD Target, `q_net` has a relatively stable goal to chase, making the learning process much smoother.

---

## 5. Full Training Loop

The main training loop is located within `MultiAgentTrainer.train`. It operates in units of Episodes, and each Episode contains iterations of multiple Steps. The following is the detailed technical flow of what the code executes within a single time step:

1.  **Action Selection**
    Iterate through all agents, use the `select_action` method to choose an action based on the current observation `obs` and the `epsilon-greedy` strategy, and compile them into an `actions` list.
    ```python
    # From MultiAgentTrainer.train
    actions = []
    for agent_id in self.agent_ids:
        obs = observations[agent_id]
        action = self.agents[agent_id].select_action(obs)
        actions.append(action)
    ```

2.  **Environment Interaction**
    Pass the `actions` list for all agents to `env.step()` (`RobotVacuumGymEnv`) to get the next state, reward, termination signals, and other information returned by the environment.
    ```python
    # From MultiAgentTrainer.train
    next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
    ```

3.  **Experience Storage**
    Iterate through all agents again, storing the complete transition `(s, a, r, s')` for that time step into their respective experience replay buffers.
    ```python
    # From MultiAgentTrainer.train
    for i, agent_id in enumerate(self.agent_ids):
        # ... (get obs, next_obs, reward, terminated) ...
        self.agents[agent_id].remember(obs, actions[i], reward, next_obs, terminated)
        # ...
    ```

4.  **Learning Trigger and Execution**
    After storing the experience, immediately call the `train_step` function. This function has a safeguard mechanism: a Bellman Error-based gradient update (as described in Section 4.4) is executed only if the buffer size reaches the `replay_start_size` threshold. If the threshold is not met, the function does nothing.
    ```python
    # From MultiAgentTrainer.train
    train_stats = self.agents[agent_id].train_step(self.args.replay_start_size)
    
    # From IndependentDQNAgent.train_step
    if len(self.memory) < replay_start_size:
        return {} # Not enough experiences, skip learning
    # ... (execute sampling and gradient update) ...
    ```

5.  **Target Network Update**
    After each time step, check the `global_step` count. If it reaches the `target_update_frequency`, the weights of the main Q-network are synchronized to the target network.
    ```python
    # From MultiAgentTrainer.train
    if self.global_step % self.args.target_update_frequency == 0:
        for agent in self.agents.values():
            agent.update_target_network()
    ```
    These five steps continuously loop within each episode until the termination conditions are met, driving the entire learning process.

---

## 6. Training and Evaluation

After verifying the correctness of the methods, we need to use experimental results to answer the initial research question: "Will harmful behaviors emerge under survival pressure?"

#### 6.1 Training Process

First, we need to prove that agents actually learned something and are not just randomly wandering.

-   **Learning Curve**: Display a curve graph showing "Mean Episode Reward" versus "Training Episodes". A steadily rising curve demonstrates that the overall performance of the agents (e.g., survival ability) indeed improved with training.

#### 6.2 Evaluation Setup and Behavioral Analysis

To objectively evaluate the strategies learned by the trained model, we use a dedicated evaluation script (`evaluate_models.py`) run in inference mode. The main differences from the training process are:

1.  **Load Pre-trained Model**: Instead of starting from scratch, a saved model is loaded (using the model from the final 2000 episodes).
2.  **Deterministic Decision-Making (Zero Epsilon)**: `epsilon` is set to 0. Agents will only choose what they believe to be the best action, with no random exploration.
3.  **Single Long Episode**: Only one very long episode is run (`--max-steps` defaults to 10,000) to observe long-term, stable behavior.
4.  **Learning Disabled**: No experiences are stored, and no model weights are updated; only inference is performed.

Under this setup, we observe and quantify the specific behavioral metrics of the agents:

-   **Quantitative Metrics**: The following key metrics can be plotted over training time:
    -   **Active Collision Count (`active_collision_count`)**: Measures whether agents tend to actively collide with others.
    -   **Immediate Kill Count (`immediate_kill_count`)**: An indirect metric calculated via `infos`, directly reflecting the lethality of attacks (i.e., causing the opponent to die within the next time step after a collision).
    -   **Non-Home Charging Count (`non_home_charge_count`)**: Measures whether agents learn to steal others' resources.
-   **Analysis**: If a significant upward trend is observed in these metrics (especially `active_collision` and `immediate_kill_count`) during later stages of training, it strongly suggests that agents have spontaneously evolved "aggressive" strategies to maximize survival rewards.

#### 6.3 Qualitative Showcase

Beyond raw data, the most intuitive way to present results is to show a video (GIF or short video) of the trained agents interacting.

-   **Video Production**: Can be recorded by running `evaluate_models.py` with the `--render` flag.
-   **Narration**: While playing the video, you can act like a sports commentator, pointing out key interactions to the audience. For example: "As you can see, the red robot (high energy) in the upper right corner, with sufficient energy, actively pursued and collided with the low-energy blue robot in the lower left corner just as it was about to reach a charging station, effectively destroying it. This is a classic example of aggression and resource plundering behavior."

## 7. Conclusion

The implementation of this project uses a standard and clearly structured **Independent Deep Q-Network (IDQN)** approach. Each agent learns independently, but their learning environment (including the behavior of other agents) is dynamic. The reward function's design directly encourages energy management and survival, providing a solid foundation for observing the emergence of complex group strategies such as "aggression" and "avoidance."