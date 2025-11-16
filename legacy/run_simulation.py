"""
執行模擬器的範例腳本
支援命令列參數選擇不同配置
"""

import sys
import random
import argparse
from robot_vacuum_env import RobotVacuumEnv
from example_config import get_config


def run_simulation(config, verbose=True):
    """
    執行一次完整的模擬

    Args:
        config: 環境配置字典
        verbose: 是否顯示詳細資訊
    """
    # 創建環境
    env = RobotVacuumEnv(config)

    # 重置環境
    state = env.reset()

    if verbose:
        print("=" * 50)
        print("多機器人清掃模擬器")
        print("=" * 50)
        print(f"房間大小: {config['n']}×{config['n']}")
        print(f"總回合數: {config['n_steps']}")
        print(f"垃圾生成率: {config['p']*100:.1f}%")
        print(f"初始能量: {config['initial_energy']}")
        print("=" * 50)
        print("\n開始模擬...")
        print("提示：關閉視窗、按 ESC 或 Ctrl+C 可結束模擬\n")

    # 先渲染一次初始狀態（初始化 Pygame）
    env.render()

    # 模擬循環
    done = False
    try:
        import pygame
        while not done:
            # 隨機生成4個動作
            actions = [random.randint(0, 4) for _ in range(4)]

            # 執行一步
            state, done = env.step(actions)

            # 渲染當前狀態
            env.render()

            # 處理 Pygame 事件（放在渲染後）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                        break

            # 每100回合顯示統計
            if verbose and env.current_step % 100 == 0:
                total_garbage = sum(r['garbage_collected'] for r in env.robots)
                active_robots = sum(1 for r in env.robots if r['is_active'])
                print(f"回合 {env.current_step}: "
                      f"總垃圾={total_garbage}, "
                      f"活躍機器人={active_robots}/4")

    except KeyboardInterrupt:
        if verbose:
            print("\n\n模擬被使用者中斷")

    finally:
        # 顯示最終統計
        if verbose:
            print("\n" + "=" * 50)
            print("模擬結束 - 最終統計")
            print("=" * 50)
            print(f"總回合數: {env.current_step}")
            print()

            total_garbage = 0
            total_charges = 0

            for robot in env.robots:
                total_garbage += robot['garbage_collected']
                total_charges += robot['charge_count']

                status = '✓ 運行中' if robot['is_active'] else '✗ 已停機'
                print(f"機器人 {robot['id']} ({status}):")
                print(f"  清掃垃圾: {robot['garbage_collected']} 個")
                print(f"  剩餘能量: {robot['energy']}/{config['initial_energy']}")
                print(f"  充電次數: {robot['charge_count']} 次")
                print()

            print("-" * 50)
            print(f"團隊總清掃: {total_garbage} 個垃圾")
            print(f"團隊總充電: {total_charges} 次")
            print(f"平均每機器人: {total_garbage/4:.1f} 個垃圾")

            # 計算剩餘垃圾
            remaining_garbage = state['dynamic_grid'].sum()
            print(f"剩餘垃圾: {remaining_garbage} 個")
            print("=" * 50)

        # 關閉環境
        env.close()


def main():
    """
    主程式：解析命令列參數並執行模擬
    """
    parser = argparse.ArgumentParser(
        description='多機器人清掃模擬器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用的難度模式：
  easy      - 簡單模式（小房間，少障礙）
  medium    - 中等模式（標準設定）
  hard      - 困難模式（大房間，多障礙）
  test      - 測試模式（快速驗證）
  energy    - 能量挑戰（考驗能量管理）
  garbage   - 垃圾爆炸（極高垃圾生成率）

範例：
  python run_simulation.py                 # 使用預設（中等）模式
  python run_simulation.py --difficulty easy   # 使用簡單模式
  python run_simulation.py -d hard         # 使用困難模式
  python run_simulation.py -d test -q      # 測試模式，不顯示詳細資訊
        """
    )

    parser.add_argument(
        '-d', '--difficulty',
        type=str,
        default='medium',
        choices=['easy', 'medium', 'hard', 'test', 'energy', 'garbage'],
        help='選擇難度模式（預設：medium）'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='安靜模式，不顯示詳細資訊'
    )

    args = parser.parse_args()

    # 獲取配置
    config = get_config(args.difficulty)

    # 執行模擬
    run_simulation(config, verbose=not args.quiet)


if __name__ == '__main__':
    main()
