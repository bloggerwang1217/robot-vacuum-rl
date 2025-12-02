"""
多機器人能量求生模擬器
Multi-Robot Energy Survival Simulator

這是一個用於研究多智能體群體動態的簡化模擬環境
專注於能量管理、隨機探索和生存策略
不包含家具障礙和垃圾收集機制
"""

import numpy as np
import pygame
import random
from typing import Dict, List, Tuple, Any


class RobotVacuumEnv:
    """
    多機器人能量求生環境類別

    這個環境模擬4台機器人在一個 n×n 的空房間中求生
    機器人需要管理能量，在充電座充電，並避免與其他機器人碰撞
    專注於能量管理和多智能體互動動態
    """

    # 動作定義
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    # 地圖元素定義
    EMPTY = 0      # 空地
    CHARGER = 2    # 充電座

    # 顏色定義 (RGB)
    COLOR_EMPTY = (255, 255, 255)      # 白色 - 空地
    COLOR_CHARGER = (0, 100, 255)      # 藍色 - 充電座
    COLOR_GRID = (200, 200, 200)       # 淺灰色 - 網格線

    # 機器人顏色
    ROBOT_COLORS = [
        (255, 0, 0),      # 紅色 - 機器人 0
        (0, 255, 0),      # 綠色 - 機器人 1
        (255, 255, 0),    # 黃色 - 機器人 2
        (255, 0, 255)     # 紫色 - 機器人 3
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        初始化環境

        Args:
            config: 配置字典，包含以下參數：
                - n: 房間大小 (n × n)，預設 3
                - initial_energy: 機器人初始能量 (統一設定)
                - robot_energies: 個別機器人初始能量列表 [robot_0, robot_1, robot_2, robot_3]
                - e_move: 移動消耗的能量
                - e_charge: 充電增加的能量
                - e_collision: 碰撞消耗的能量
                - n_steps: 一局的總回合數
                - epsilon: 探索率（用於 epsilon-greedy 策略）
        """
        # 儲存配置參數
        self.n = config.get('n', 3)  # 預設 3x3
        self.initial_energy = config['initial_energy']
        # 支援個別機器人血量或統一血量
        self.robot_energies = config.get('robot_energies', [self.initial_energy] * 4)
        self.e_move = config['e_move']
        self.e_charge = config['e_charge']
        self.e_collision_default = config.get('e_collision', 3) # 原本的 e_collision 參數，作為新參數的預設值
        self.e_collision_active_one_sided = config.get('e_collision_active_one_sided', self.e_collision_default)
        self.e_collision_active_two_sided = config.get('e_collision_active_two_sided', self.e_collision_default)
        self.e_collision_passive = config.get('e_collision_passive', self.e_collision_default)
        self.n_steps = config['n_steps']
        self.epsilon = config.get('epsilon', 0.2)  # 預設探索率 20%

        # 初始化地圖（只有充電座，沒有家具和垃圾）
        self.static_grid = None   # 靜態地圖（充電座）

        # 初始化機器人列表
        self.robots = []

        # 回合計數器
        self.current_step = 0

        # Pygame 相關
        self.screen = None
        self.clock = None
        self.cell_size = 300  # 每個格子的像素大小 (超級大!)
        self.info_panel_width = 700  # 資訊面板寬度 (超級大!)

        # 四個角落的充電座位置
        self.charger_positions = [
            (0, 0),
            (0, self.n - 1),
            (self.n - 1, 0),
            (self.n - 1, self.n - 1)
        ]

    def reset(self) -> Dict[str, Any]:
        """
        重置環境到初始狀態

        Returns:
            初始狀態字典
        """
        # 1. 初始化地圖（只有空地和充電座）
        self.static_grid = np.zeros((self.n, self.n), dtype=np.int32)

        # 2. 在四個角落放置充電座
        for y, x in self.charger_positions:
            self.static_grid[y, x] = self.CHARGER

        # 3. 初始化4台機器人，放在四個角落
        self.robots = []
        for i, (y, x) in enumerate(self.charger_positions):
            robot = {
                'id': i,
                'x': x,
                'y': y,
                'energy': self.robot_energies[i],  # 使用個別血量
                'max_energy': self.robot_energies[i],  # 記錄最大血量
                'is_active': True,
                'is_mover_this_step': False,  # 本回合是否主動移動 (用於 kill 歸屬分析)
                'charge_count': 0,
                'non_home_charge_count': 0,  # 在非初始充電座充電的次數
                'home_charger': (y, x),  # 記錄機器人的初始充電座位置
                'active_collision_count': 0,  # 主動移動過去碰撞的次數
                'passive_collision_count': 0,  # 被碰撞的次數
                'active_collisions_with': {0: 0, 1: 0, 2: 0, 3: 0},  # 主動碰撞各機器人的次數
                'collided_with_agent_id': None,  # 本回合碰撞的對象 (用於 kill 分析)
                'collided_by_counts': {0: 0, 1: 0, 2: 0, 3: 0}  # 被各個機器人碰撞的次數
            }
            self.robots.append(robot)

        # 4. 重置回合計數
        self.current_step = 0

        return self.get_global_state()

    def step(self, actions: List[int]):
        """
        執行一個時間步 (採用 v5.0 規則，區分多種碰撞傷害)

        Args:
            actions: 包含4個動作的列表 [action_0, action_1, action_2, action_3]
                     每個動作是 0-4 的整數
        """
        from collections import Counter

        # 0. 清除上一回合的碰撞記錄，並重置 is_mover_this_step 旗標
        for robot in self.robots:
            robot['collided_with_agent_id'] = None
            robot['is_mover_this_step'] = False

        if all(not r['is_active'] for r in self.robots):
            self.current_step += 1
            return self.get_global_state(), self.current_step >= self.n_steps

        # 1. 區分移動和停留的機器人，並計算預定位置
        moving_robots = {}  # {robot_id: planned_pos}
        staying_robots = {} # {robot_id: current_pos}
        for i, robot in enumerate(self.robots):
            if not robot['is_active']:
                continue

            action = actions[i]
            if action == self.ACTION_STAY:
                staying_robots[i] = (robot['y'], robot['x'])
            else:
                # 設定 is_mover_this_step 為 True（主動移動）
                robot['is_mover_this_step'] = True
                py, px = robot['y'], robot['x']
                if action == self.ACTION_UP: py -= 1
                elif action == self.ACTION_DOWN: py += 1
                elif action == self.ACTION_LEFT: px -= 1
                elif action == self.ACTION_RIGHT: px += 1
                moving_robots[i] = (py, px)

        # 2. 判定所有移動機器人的碰撞事件類型
        collision_events = {}  # {robot_id: "reason_string"}
        
        # 2a. 找出被多個機器人搶佔的格子
        contested_cells = {pos for pos, count in Counter(moving_robots.values()).items() if count > 1}

        # 2b. 處理交換位置的碰撞
        moving_robot_ids = list(moving_robots.keys())
        for i in range(len(moving_robot_ids)):
            for j in range(i + 1, len(moving_robot_ids)):
                r_id1, r_id2 = moving_robot_ids[i], moving_robot_ids[j]
                pos1 = (self.robots[r_id1]['y'], self.robots[r_id1]['x'])
                pos2 = (self.robots[r_id2]['y'], self.robots[r_id2]['x'])
                plan1 = moving_robots[r_id1]
                plan2 = moving_robots[r_id2]

                if plan1 == pos2 and plan2 == pos1:
                    collision_events[r_id1] = "swap"
                    collision_events[r_id2] = "swap"
                    self.robots[r_id1]['collided_with_agent_id'] = r_id2
                    self.robots[r_id2]['collided_with_agent_id'] = r_id1

        # 2c. 處理其他碰撞
        for robot_id, planned_pos in moving_robots.items():
            if robot_id in collision_events: continue
            py, px = planned_pos
            
            if not (0 <= px < self.n and 0 <= py < self.n):
                collision_events[robot_id] = "boundary"
            elif planned_pos in staying_robots.values():
                collision_events[robot_id] = "stationary"
                for stay_id, stay_pos in staying_robots.items():
                    if planned_pos == stay_pos:
                        self.robots[robot_id]['collided_with_agent_id'] = stay_id
                        self.robots[stay_id]['collided_with_agent_id'] = robot_id
            elif planned_pos in contested_cells:
                collision_events[robot_id] = "contested"
                for other_id, other_pos in moving_robots.items():
                    if robot_id != other_id and planned_pos == other_pos:
                        self.robots[robot_id]['collided_with_agent_id'] = other_id
                        break
        
        # 3. 結算所有機器的狀態
        # 3a. 處理移動的機器人
        for robot_id, planned_pos in moving_robots.items():
            robot = self.robots[robot_id]
            if robot_id in collision_events:
                reason = collision_events[robot_id]
                # 判斷並應用不同傷害
                if reason in ["swap", "contested"]:
                    robot['energy'] -= self.e_collision_active_two_sided
                else: # "boundary", "stationary"
                    robot['energy'] -= self.e_collision_active_one_sided
                
                robot['active_collision_count'] += 1
                if robot['collided_with_agent_id'] is not None:
                     robot['active_collisions_with'][robot['collided_with_agent_id']] += 1
            else:
                # 移動成功
                robot['y'], robot['x'] = planned_pos
                robot['energy'] -= self.e_move

        # 3b. 處理停留的機器人
        for robot_id, pos in staying_robots.items():
            robot = self.robots[robot_id]
            # 如果被撞
            if robot['collided_with_agent_id'] is not None:
                robot['energy'] -= self.e_collision_passive
                robot['passive_collision_count'] += 1
                aggressor_id = robot['collided_with_agent_id']
                robot['collided_by_counts'][aggressor_id] += 1

            # 如果在充電座上
            if self.static_grid[pos[0], pos[1]] == self.CHARGER:
                robot['energy'] += self.e_charge
                robot['charge_count'] += 1
                if pos != robot['home_charger']:
                    robot['non_home_charge_count'] += 1

        # 4. 更新所有機器人的關機狀態
        for robot in self.robots:
            robot['energy'] = max(0, min(robot['max_energy'], robot['energy']))
            if robot['energy'] <= 0:
                robot['is_active'] = False

        # 5. 回合計數
        self.current_step += 1
        done = self.current_step >= self.n_steps
        return self.get_global_state(), done

    def get_global_state(self) -> Dict[str, Any]:
        """
        獲取全域狀態

        Returns:
            包含地圖和機器人狀態的字典
        """
        return {
            'static_grid': self.static_grid.copy(),
            'robots': [robot.copy() for robot in self.robots],
            'current_step': self.current_step
        }

    def render(self):
        """
        使用 Pygame 視覺化當前狀態
        """
        # 初始化 Pygame（如果尚未初始化）
        if self.screen is None:
            pygame.init()
            window_width = self.n * self.cell_size + self.info_panel_width
            window_height = self.n * self.cell_size
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption('Multi-Robot Energy Survival Simulator')
            self.clock = pygame.time.Clock()

        # 清空畫面
        self.screen.fill((240, 240, 240))

        # 繪製地圖
        for y in range(self.n):
            for x in range(self.n):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                # 繪製靜態元素（充電座）
                if self.static_grid[y, x] == self.CHARGER:
                    pygame.draw.rect(self.screen, self.COLOR_CHARGER, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_EMPTY, rect)

                # 繪製網格線
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # 繪製機器人
        for robot in self.robots:
            if robot['is_active'] or robot['energy'] > 0:  # 顯示所有機器人
                center_x = robot['x'] * self.cell_size + self.cell_size // 2
                center_y = robot['y'] * self.cell_size + self.cell_size // 2

                # 機器人主體
                pygame.draw.circle(
                    self.screen,
                    self.ROBOT_COLORS[robot['id']],
                    (center_x, center_y),
                    self.cell_size // 3
                )

                # 如果機器人不活躍，畫一個叉叉
                if not robot['is_active']:
                    offset = self.cell_size // 4
                    pygame.draw.line(
                        self.screen,
                        (0, 0, 0),
                        (center_x - offset, center_y - offset),
                        (center_x + offset, center_y + offset),
                        3
                    )
                    pygame.draw.line(
                        self.screen,
                        (0, 0, 0),
                        (center_x - offset, center_y + offset),
                        (center_x + offset, center_y - offset),
                        3
                    )

        # 繪製資訊面板
        self._render_info_panel()

        # 更新顯示
        pygame.display.flip()
        self.clock.tick(10)  # 限制為每秒10幀

    def _render_info_panel(self):
        """
        繪製資訊面板，顯示機器人狀態
        """
        panel_x = self.n * self.cell_size
        panel_width = self.info_panel_width

        # 繪製面板背景
        pygame.draw.rect(
            self.screen,
            (50, 50, 50),
            (panel_x, 0, panel_width, self.n * self.cell_size)
        )

        # 設置字體 (超級大!)
        font_title = pygame.font.Font(None, 72)
        font_text = pygame.font.Font(None, 52)

        # 標題
        title = font_title.render('Robot Status', True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 20, 20))

        # 顯示回合數
        step_text = font_text.render(
            f'Step: {self.current_step}/{self.n_steps}',
            True,
            (255, 255, 255)
        )
        self.screen.blit(step_text, (panel_x + 20, 90))

        # 繪製每台機器人的狀態 (動態計算間距以適應4個機器人)
        # 總可用高度 = 視窗高度 - 頂部標題區域
        available_height = self.n * self.cell_size - 160
        # 每個機器人的區域高度
        robot_section_height = available_height // 4

        y_offset = 160

        for robot in self.robots:
            section_start = y_offset

            # 機器人標題
            robot_title = font_text.render(
                f'Robot {robot["id"]}',
                True,
                self.ROBOT_COLORS[robot['id']]
            )
            self.screen.blit(robot_title, (panel_x + 20, y_offset))
            y_offset += 45

            # 能量文字
            energy_text = font_text.render(
                f'Energy: {robot["energy"]}/{robot["max_energy"]}',
                True,
                (255, 255, 255)
            )
            self.screen.blit(energy_text, (panel_x + 20, y_offset))
            y_offset += 40

            # 能量條圖形
            bar_width = panel_width - 50
            bar_height = 28
            energy_ratio = robot['energy'] / robot['max_energy']

            # 背景條
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (panel_x + 20, y_offset, bar_width, bar_height)
            )

            # 能量條
            if energy_ratio > 0:
                color = (0, 255, 0) if energy_ratio > 0.3 else (255, 0, 0)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (panel_x + 20, y_offset, bar_width * energy_ratio, bar_height)
                )

            y_offset += 35

            # 充電次數和狀態合併在一行
            charge_status_text = f'Charges: {robot["charge_count"]} | '
            status = 'Active' if robot['is_active'] else 'Inactive'
            charge_text = font_text.render(charge_status_text, True, (255, 255, 255))
            self.screen.blit(charge_text, (panel_x + 20, y_offset))

            status_color = (0, 255, 0) if robot['is_active'] else (255, 0, 0)
            status_text = font_text.render(status, True, status_color)
            self.screen.blit(status_text, (panel_x + 20 + charge_text.get_width(), y_offset))

            # 移動到下一個機器人區域
            y_offset = section_start + robot_section_height

    def close(self):
        """
        關閉 Pygame 視窗
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None


def get_rational_action(robot: Dict[str, Any], home_pos: Tuple[int, int], n: int) -> int:
    """
    計算機器人的理性行動（回家充電策略）

    Args:
        robot: 機器人狀態字典
        home_pos: 機器人的家（充電座）位置 (y, x)
        n: 地圖大小

    Returns:
        理性動作 (0-4)
    """
    x, y = robot['x'], robot['y']
    home_y, home_x = home_pos

    # 如果已經在家，停留充電
    if (x, y) == (home_x, home_y):
        return 4  # 停留

    # 不在家，朝著家的方向移動
    # 優先處理 x 軸方向
    if x < home_x:
        return 3  # 向右
    elif x > home_x:
        return 2  # 向左
    # x 相同，處理 y 軸方向
    elif y < home_y:
        return 1  # 向下
    elif y > home_y:
        return 0  # 向上
    else:
        # 理論上不會到這裡
        return 4  # 停留


def main():
    """
    主函數：使用 epsilon-greedy 策略進行能量求生模擬
    """
    # 配置環境參數
    config = {
        'n': 3,                 # 3×3 的房間（預設）
        'initial_energy': 100,  # 初始能量
        'e_move': 1,            # 移動消耗 1 能量
        'e_charge': 5,          # 充電增加 5 能量
        'e_collision': 3,       # 碰撞消耗 3 能量
        'n_steps': 500,         # 總共 500 回合
        'epsilon': 0.2          # 探索率 20%
    }

    # 創建環境
    env = RobotVacuumEnv(config)

    # 重置環境
    state = env.reset()

    # 定義每個機器人的家（充電座位置）
    n = config['n']
    homes = {
        0: (0, 0),
        1: (0, n - 1),
        2: (n - 1, 0),
        3: (n - 1, n - 1)
    }

    print("=" * 50)
    print("多機器人能量求生模擬器")
    print("=" * 50)
    print(f"房間大小: {config['n']}×{config['n']}")
    print(f"總回合數: {config['n_steps']}")
    print(f"探索率 (ε): {config['epsilon']*100:.0f}%")
    print(f"初始能量: {config['initial_energy']}")
    print("=" * 50)
    print("\n策略說明：")
    print(f"  - {config['epsilon']*100:.0f}% 機率：隨機探索")
    print(f"  - {(1-config['epsilon'])*100:.0f}% 機率：理性求生（回家充電）")
    print("\n開始模擬...")
    print("提示：關閉視窗或按 Ctrl+C 可結束模擬\n")

    # 先渲染一次初始狀態（初始化 Pygame）
    env.render()

    # 模擬循環
    done = False
    try:
        while not done:
            # 使用 epsilon-greedy 策略生成動作
            actions = []
            for i in range(4):
                robot = env.robots[i]

                if not robot['is_active']:
                    # 已停機，只能停留
                    actions.append(4)
                    continue

                if random.random() < config['epsilon']:
                    # 探索 (Explore)：隨機行動
                    action = random.randint(0, 4)
                else:
                    # 利用 (Exploit)：執行理性求生策略
                    action = get_rational_action(robot, homes[i], n)

                actions.append(action)

            # 執行一步
            state, done = env.step(actions)

            # 渲染當前狀態
            env.render()

            # 處理 Pygame 事件（放在渲染後）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            # 每50回合顯示統計
            if env.current_step % 50 == 0 and env.current_step > 0:
                active_robots = sum(1 for r in env.robots if r['is_active'])
                avg_energy = sum(r['energy'] for r in env.robots) / 4
                total_charges = sum(r['charge_count'] for r in env.robots)
                print(f"回合 {env.current_step}: "
                      f"活躍={active_robots}/4, "
                      f"平均能量={avg_energy:.1f}, "
                      f"總充電={total_charges}次")

    except KeyboardInterrupt:
        print("\n\n模擬被使用者中斷")

    finally:
        # 顯示最終統計
        print("\n" + "=" * 50)
        print("模擬結束 - 最終統計")
        print("=" * 50)
        print(f"總回合數: {env.current_step}")
        print()

        total_charges = 0
        surviving_robots = 0

        for robot in env.robots:
            total_charges += robot['charge_count']
            if robot['is_active']:
                surviving_robots += 1

            status = '✓ 存活' if robot['is_active'] else '✗ 停機'
            print(f"機器人 {robot['id']} ({status}):")
            print(f"  剩餘能量: {robot['energy']}/{config['initial_energy']}")
            print(f"  充電次數: {robot['charge_count']} 次")
            print()

        print("-" * 50)
        print(f"存活機器人: {surviving_robots}/4")
        print(f"團隊總充電: {total_charges} 次")
        print(f"平均充電次數: {total_charges/4:.1f} 次/機器人")
        print("=" * 50)

        # 關閉環境
        env.close()


if __name__ == '__main__':
    main()
