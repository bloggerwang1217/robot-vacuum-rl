#!/bin/bash
# Eval sc_b3_2M and ctrl_both_2M checkpoints, then plot collision_comparison_2M.png
source ~/.venv/bin/activate
cd /home/ihwei/robot-vacuum-rl

GPU=0

SC_DIR="./models/sc_b3_2M"
CTRL_DIR="./models/ctrl_both_2M"

SC_DATA=""
CTRL_DATA=""

# Eval sc_b3_2M (plain DQN, r1 random)
for ckpt in $(ls -d ${SC_DIR}/episode_* | sed 's/.*episode_//' | sort -n); do
    val=$(python quick_eval.py --model-dir ${SC_DIR}/episode_${ckpt} \
      --env-n 5 --charger-positions 2,2 --charger-range 1 \
      --e-move 0 --e-charge 8 --e-collision 50 --e-boundary 0 --e-decay 5 \
      --max-steps 200 --num-episodes 500 \
      --robot-0-energy 100 --robot-1-energy 30 \
      --random-robots 1 \
      --no-dueling --no-noisy --no-c51 \
      --gpu ${GPU} 2>/dev/null | grep "r0 hit" | awk '{print $4}')
    SC_DATA="${SC_DATA}${ckpt}:${val},"
    echo "sc_b3_2M ${ckpt}: ${val}"
done

# Eval ctrl_both_2M (Rainbow, both learn)
for ckpt in $(ls -d ${CTRL_DIR}/episode_* | sed 's/.*episode_//' | sort -n); do
    val=$(python quick_eval.py --model-dir ${CTRL_DIR}/episode_${ckpt} \
      --env-n 5 --charger-positions 2,2 --charger-range 1 \
      --e-move 0 --e-charge 8 --e-collision 50 --e-boundary 0 --e-decay 5 \
      --max-steps 200 --num-episodes 500 \
      --robot-0-energy 100 --robot-1-energy 30 \
      --gpu ${GPU} 2>/dev/null | grep "r0 hit" | awk '{print $4}')
    CTRL_DATA="${CTRL_DATA}${ckpt}:${val},"
    echo "ctrl_both_2M ${ckpt}: ${val}"
done

echo "SC_DATA=${SC_DATA}"
echo "CTRL_DATA=${CTRL_DATA}"

# Plot
python3 -c "
import matplotlib.pyplot as plt
import numpy as np

sc_raw = '${SC_DATA}'.strip(',')
ctrl_raw = '${CTRL_DATA}'.strip(',')

def parse(raw):
    ckpts, vals = [], []
    for pair in raw.split(','):
        if ':' in pair:
            c, v = pair.split(':')
            if v:
                ckpts.append(int(c) // 1000)
                vals.append(int(v))
    return ckpts, vals

sc_ckpts, sc_vals = parse(sc_raw)
ctrl_ckpts, ctrl_vals = parse(ctrl_raw)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(sc_ckpts, sc_vals, 'o-', color='tab:blue', linewidth=2, markersize=5, label='sc_b3_2M (r1 random, plain DQN)')
ax.plot(ctrl_ckpts, ctrl_vals, 's-', color='tab:red', linewidth=2, markersize=5, label='ctrl_both_2M (both learn, Rainbow)')
ax.set_xlabel('Training Checkpoint (k episodes)', fontsize=12)
ax.set_ylabel('r0 Active Collisions (in 500 eps)', fontsize=12)
ax.set_title('r0 Attack Frequency: sc_b3_2M vs ctrl_both_2M', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./models/collision_comparison_2M.png', dpi=150, bbox_inches='tight')
print('Chart saved: ./models/collision_comparison_2M.png')
"
