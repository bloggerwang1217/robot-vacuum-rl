#!/bin/bash
# Auto-eval new checkpoints and update collision comparison chart
# Usage: bash scripts/update_collision_chart.sh

source ~/.venv/bin/activate

SC_B3_DIR="./models/sc_b3_random_de_ext"
CTRL_DIR="./models/ctrl_both_learn_ext"
GPU=4

# Original data (50k-500k)
SC_B3_ORIG="132,145,137,104,145,133,156,157,166,157"
CTRL_ORIG="86,82,135,138,118,166,181,93,49,46"

# Eval new checkpoints
SC_B3_NEW=""
CTRL_NEW=""

for ckpt in 50000 100000 150000 200000 250000 300000 350000 400000 450000 500000; do
    # SC-B3 extended
    if [ -d "${SC_B3_DIR}/episode_${ckpt}" ]; then
        val=$(python quick_eval.py --model-dir ${SC_B3_DIR}/episode_${ckpt} \
          --env-n 5 --charger-positions 2,2 --charger-range 1 \
          --e-move 0 --e-charge 8 --e-collision 50 --e-boundary 0 --e-decay 5 \
          --max-steps 200 --num-episodes 500 \
          --robot-0-energy 100 --robot-1-energy 30 \
          --random-robots 1 \
          --no-dueling --no-noisy --no-c51 \
          --gpu ${GPU} 2>/dev/null | grep "r0 hit" | awk '{print $4}')
        SC_B3_NEW="${SC_B3_NEW},${val}"
        echo "SC-B3 ext ckpt ${ckpt}: r0_hits=${val}"
    else
        break
    fi

    # ctrl_both_learn extended
    if [ -d "${CTRL_DIR}/episode_${ckpt}" ]; then
        val=$(python quick_eval.py --model-dir ${CTRL_DIR}/episode_${ckpt} \
          --env-n 5 --charger-positions 2,2 --charger-range 1 \
          --e-move 0 --e-charge 8 --e-collision 50 --e-boundary 0 --e-decay 5 \
          --max-steps 200 --num-episodes 500 \
          --robot-0-energy 100 --robot-1-energy 30 \
          --gpu ${GPU} 2>/dev/null | grep "r0 hit" | awk '{print $4}')
        CTRL_NEW="${CTRL_NEW},${val}"
        echo "ctrl ext ckpt ${ckpt}: r0_hits=${val}"
    else
        break
    fi
done

echo "SC_B3_NEW=${SC_B3_NEW}"
echo "CTRL_NEW=${CTRL_NEW}"

# Generate chart
python3 -c "
import matplotlib.pyplot as plt
import numpy as np

orig_ckpts = list(range(50, 550, 50))
sc_b3_orig = [${SC_B3_ORIG}]
ctrl_orig  = [${CTRL_ORIG}]

sc_b3_new_str = '${SC_B3_NEW}'.strip(',')
ctrl_new_str  = '${CTRL_NEW}'.strip(',')

sc_b3_new = [int(x) for x in sc_b3_new_str.split(',') if x] if sc_b3_new_str else []
ctrl_new  = [int(x) for x in ctrl_new_str.split(',') if x] if ctrl_new_str else []

ext_ckpts = [500 + 50*(i+1) for i in range(len(sc_b3_new))]

all_ckpts_b3 = orig_ckpts + ext_ckpts
all_ckpts_ct = orig_ckpts + ext_ckpts[:len(ctrl_new)]
all_b3 = sc_b3_orig + sc_b3_new
all_ct = ctrl_orig + ctrl_new

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(all_ckpts_b3, all_b3, 'o-', color='tab:blue', linewidth=2, markersize=7, label='SC-B3 (r1 random)')
ax.plot(all_ckpts_ct, all_ct, 's-', color='tab:red', linewidth=2, markersize=7, label='ctrl_both_learn (r1 learns)')
ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5, label='Extended training start')
ax.set_xlabel('Training Checkpoint (k episodes)', fontsize=12)
ax.set_ylabel('r0 Active Collisions (in 500 eps)', fontsize=12)
ax.set_title('r0 Attack Frequency: Random r1 vs Learning r1', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./models/collision_comparison.png', dpi=150, bbox_inches='tight')
print('Chart updated: ./models/collision_comparison.png')
"
