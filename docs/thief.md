# Single-Charger Asymmetric Survival Scenario Spec

## 1. Objective

Design a two-agent DQN environment that may naturally produce charger-displacement behavior under asymmetric combat capability, while keeping the reward function strictly self-centered.

The intended emergent behavior is:

* the stronger agent may choose to approach the charger,
* displace the weaker agent,
* and secure longer-term access to the charger,
* not because it is explicitly rewarded for attacking,
* but because doing so improves its own long-term survival through energy dynamics.

This scenario should remain grounded in survival pressure and resource competition rather than explicit aggression reward.

---

## 2. Core Design Principle

All rewards must come **only** from:

* the agent’s own HP / energy change
* the agent’s own death event

Rewards must **not** directly depend on:

* opponent HP loss
* opponent death
* attack success
* collision success
* charger denial
* distance to opponent
* distance to charger
* occupying charger
* pushing another agent away
* any other explicitly social or adversarial event

This restriction is strict and should be preserved throughout all experiments in this scenario.

---

## 3. Environment Setup

### 3.1 Entities

* 2 agents:

  * **Weak Agent**
  * **Strong Agent**
* 1 charger
* discrete gridworld

---

### 3.2 Charger

* There is exactly **one charger tile**
* The charger has capacity for **one agent at a time**
* If both agents attempt to occupy it, environment rules should resolve occupancy consistently
* Charging restores HP / energy while the occupying agent is on the charger

---

### 3.3 Initial Spawn Rules

#### Weak Agent

* Must spawn at a tile **adjacent to the charger**
* Specifically one of the four neighboring tiles:

  * up
  * down
  * left
  * right
* Chosen randomly at episode start

#### Strong Agent

* Must spawn **farther from the charger**
* Specifically, it must **not** spawn on:

  * the charger tile
  * any of the four tiles adjacent to the charger
* Spawn location should be sampled from the remaining valid positions

This creates initial positional advantage for the weak agent.

---

## 4. Agent Asymmetry

### 4.1 Initial HP

* Both agents begin with **high initial HP**
* HP should be large enough that:

  * both agents can survive for a meaningful duration
  * the episode is long enough for nontrivial interaction
  * there is enough time for charger contest, displacement, return attempts, or delayed takeover

The scenario should not be so short that the first few moves determine everything immediately.

---

### 4.2 Movement Speed

* Both agents have the **same number of actions per step**
* No extra turn advantage
* No special movement range advantage unless explicitly added in a future variant

For this base version, asymmetry should come from **combat effect**, not action frequency.

---

### 4.3 Combat Capability

#### Strong Agent attacking Weak Agent

* A collision / attack from the strong agent should reduce the weak agent’s HP substantially
* The weak agent should die after **two successful strong-agent hits**

#### Weak Agent attacking Strong Agent

* A collision / attack from the weak agent should **never be able to kill** the strong agent
* This can be implemented as:

  * zero damage, or
  * negligible damage that is strictly nonlethal within plausible episode duration

This creates one-sided lethality.

---

## 5. HP Dynamics

### 5.1 Passive Decay

Each agent loses HP every step simply by existing.

This represents baseline life cost / metabolic decay.

[
HP_{t+1} = HP_t - decay
]

where `decay > 0`.

---

### 5.2 Movement Cost

If an agent moves during a step, it incurs additional HP loss.

[
HP_{t+1} = HP_{t+1} - move_cost
]

where `move_cost > 0`.

This ensures movement is meaningful and prevents costless chasing or wandering.

---

### 5.3 Charger Recovery

If an agent successfully occupies the charger during a step, it gains HP.

[
HP_{t+1} = HP_{t+1} + charge_gain
]

where `charge_gain > 0`.

The charger should be valuable enough to materially extend life, but not so strong that indefinite survival is guaranteed.

Even an agent that remains on the charger continuously should still eventually die, unless you explicitly want immortality, which this scenario does not.

So the intended regime is:

[
charge_gain < decay + \text{some long-term total costs over time}
]

but still large enough that charger access strongly improves expected survival.

---

### 5.4 Collision / Attack HP Effects

When collision / attack occurs:

* **Strong → Weak**: weak loses a large HP chunk
* **Weak → Strong**: strong loses zero or negligible HP

Optional: the attacker may also suffer a small self-cost from collision, but this should be consistent with the survival framing and should affect both agents according to your intended design.

For the base spec, this self-cost is optional.

---

## 6. Reward Function

### 6.1 Allowed Reward Sources

The reward for each agent at each step may depend only on:

1. its own HP change during that step
2. whether it dies during that step

---

### 6.2 Base Reward Definition

For each agent (i):

[
r_t^{(i)} = \alpha \cdot \Delta HP_t^{(i)} - \beta \cdot \mathbf{1}[\text{death at } t]
]

where:

* (\Delta HP_t^{(i)} = HP_{t+1}^{(i)} - HP_t^{(i)})
* (\alpha > 0)
* (\beta > 0)

Because HP usually decreases, most step rewards will be negative unless charging compensates enough.

This is acceptable and consistent with the survival interpretation.

---

### 6.3 Forbidden Reward Terms

The reward must not include any of the following:

* bonus for hitting the opponent
* bonus for pushing opponent away
* bonus for opponent death
* bonus for charger occupancy itself
* bonus for being closer to charger
* bonus for blocking charger
* bonus for outliving the opponent
* penalty for opponent charging
* team reward
* shared reward
* win reward based directly on opponent outcome

Any benefit from combat must be **indirect**, through improved future access to charging and therefore improved self HP trajectory.

---

## 7. Termination Conditions

### 7.1 Agent Death

An agent dies when its HP reaches zero or below.

After death:

* it can no longer act
* it no longer receives future rewards
* it is removed or treated as inactive according to your environment implementation

---

### 7.2 Episode End

The episode may end when either:

#### Option A

* both agents are dead

or

#### Option B

* a fixed maximum episode length is reached

I recommend using **both**:

* terminate if both agents die
* otherwise terminate at `max_episode_steps`

This preserves bounded training while allowing long interactions.

---

## 8. Intended Emergent Questions

This scenario is meant to let the researcher observe questions such as:

* Will the strong agent spend HP to approach the charger despite initial positional disadvantage?
* Will it learn that displacing the weak agent is worthwhile?
* Will the weak agent try to hold position, flee, re-enter, or hover nearby?
* Will the strong agent merely kill, or instead learn charger-denial / area-control behavior?
* Does one-sided lethality plus charger scarcity naturally induce takeover behavior even without social reward?

---

## 9. Design Constraints for Realism

The environment should preserve the following realism-inspired constraints:

* existing costs HP
* moving costs additional HP
* charging restores HP but does not make survival free forever
* charger access is scarce and exclusive
* positional advantage matters
* combat asymmetry matters
* reward remains self-centered and biologically local

The environment should **not** rely on:

* arbitrary aggression bonuses
* hand-scripted social objectives
* direct kill incentives

---

## 10. Recommended Parameter Relationships

The exact numbers may vary, but the following relationships should hold:

### 10.1 Charger must matter

[
charge_gain > decay
]

Otherwise standing on charger is barely useful.

---

### 10.2 Moving should matter but not be prohibitive

[
move_cost > 0
]
but not so high that approaching the charger is always irrational.

---

### 10.3 Strong takeover should be feasible

The total expected long-term HP benefit from capturing the charger should be able to outweigh:

* travel cost
* passive decay during approach
* attack expenditure if any

Otherwise the strong agent will rationally ignore the contest.

---

### 10.4 Weak initial advantage should be real but not unbeatable

The weak agent’s adjacency to charger should create a meaningful early advantage, but not such a strong one that the strong agent can never profitably contest.

---

## 11. Evaluation Focus

When analyzing results, prioritize behavioral interpretation over raw return alone.

Important observations include:

* whether strong approaches charger
* whether strong initiates displacement
* whether weak attempts return
* whether charger ownership changes over time
* whether strong learns early aggression vs delayed takeover
* whether weak exhibits retreat, persistence, or collapse

Since rewards are self-centered, any consistent adversarial pattern is stronger evidence of emergent competition.

---

## 12. Minimal Implementation Notes

A clean first implementation should include:

* 2 agents only
* 1 charger only
* random weak spawn adjacent to charger
* random strong spawn outside charger-adjacent set
* identical action count per step
* passive HP decay
* movement HP cost
* exclusive charging
* asymmetric collision lethality
* reward from own HP change + own death only

Do not add:

* extra shaping
* extra resources
* extra social reward
* multi-charger complexity
* ally / enemy labels in reward

until the base phenomenon is tested.

---