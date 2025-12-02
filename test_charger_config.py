#!/usr/bin/env python3
"""
測試充電器配置功能
"""

from gym import RobotVacuumGymEnv
import numpy as np

def test_config(description, charger_positions, n=3):
    """測試特定充電器配置"""
    print(f"\n{'='*60}")
    print(f"測試: {description}")
    print(f"{'='*60}")

    try:
        env = RobotVacuumGymEnv(
            n=n,
            initial_energy=100,
            e_move=1,
            e_charge=5,
            e_collision=3,
            n_steps=500,
            charger_positions=charger_positions
        )

        print(f"✓ 環境創建成功")
        print(f"  網格大小: {n}x{n}")
        print(f"  充電器數量: {len(env.env.charger_positions)}")
        print(f"  充電器位置: {env.env.charger_positions}")
        print(f"  觀察空間維度: {env.observation_space.shape[0]}")
        print(f"  預期維度: {12 + 2 * len(env.env.charger_positions)}")

        # 測試 reset
        obs, info = env.reset()
        print(f"✓ Reset 成功")
        print(f"  觀察數量: {len(obs)}")

        # 檢查機器人初始位置
        state = env.env.get_global_state()
        print(f"  機器人初始位置:")
        for i, robot in enumerate(state['robots']):
            home = robot['home_charger']
            home_str = f"({home[0]},{home[1]})" if home else "None"
            print(f"    Robot {i}: ({robot['y']},{robot['x']}) - home_charger: {home_str}")

        # 測試一步
        actions = {f'robot_{i}': 4 for i in range(4)}  # 所有機器人都 STAY
        obs, rewards, dones, truncated, info = env.step(actions)
        print(f"✓ Step 成功")

        # 驗證觀察維度
        for agent_id, observation in obs.items():
            expected_dim = 12 + 2 * len(env.env.charger_positions)
            actual_dim = len(observation)
            if actual_dim == expected_dim:
                print(f"  {agent_id}: 觀察維度正確 ({actual_dim})")
            else:
                print(f"  ✗ {agent_id}: 觀察維度錯誤! 預期 {expected_dim}, 實際 {actual_dim}")

        return True

    except Exception as e:
        print(f"✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("充電器配置測試")
    print("="*60)

    # 測試 1: 預設配置（四個角落）
    test_config(
        "預設配置 - 四個角落",
        charger_positions=None,
        n=3
    )

    # 測試 2: 四個角落（顯式指定）
    test_config(
        "顯式指定 - 四個角落",
        charger_positions=[(0, 0), (0, 2), (2, 0), (2, 2)],
        n=3
    )

    # 測試 3: 只有兩個充電器
    test_config(
        "兩個充電器 - 對角線",
        charger_positions=[(0, 0), (2, 2)],
        n=3
    )

    # 測試 4: 只有一個充電器
    test_config(
        "一個充電器 - 中心",
        charger_positions=[(1, 1)],
        n=3
    )

    # 測試 5: 三個充電器
    test_config(
        "三個充電器",
        charger_positions=[(0, 0), (0, 2), (2, 0)],
        n=3
    )

    # 測試 6: 使用 -1,-1 禁用充電器
    test_config(
        "禁用部分充電器 - 使用 (-1,-1)",
        charger_positions=[(0, 0), (-1, -1), (2, 0), (2, 2)],
        n=3
    )

    # 測試 7: 超出邊界的座標會被過濾
    test_config(
        "過濾無效座標",
        charger_positions=[(0, 0), (0, 2), (5, 5), (2, 2)],
        n=3
    )

    print(f"\n{'='*60}")
    print("所有測試完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
