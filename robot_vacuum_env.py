"""
多機器人清掃模擬器
Multi-Robot Vacuum Cleaner Simulator

這是一個用於強化學習的多智能體模擬環境
目前只實作遊戲引擎本身，不包含 RL 訓練邏輯
"""

import numpy as np
import pygame
import random
from typing import Dict, List, Tuple, Any


class RobotVacuumEnv:
    """
    多機器人清掃環境類別

    這個環境模擬4台機器人在一個 n×n 的房間中清掃垃圾
    房間包含家具（障礙物）和充電座
    """

    # 動作定義
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    # 地圖元素定義
    EMPTY = 0      # 空地
    FURNITURE = 1  # 家具/障礙物
    CHARGER = 2    # 充電座

    # 顏色定義 (RGB)
    COLOR_EMPTY = (255, 255, 255)      # 白色 - 空地
    COLOR_FURNITURE = (139, 69, 19)    # 棕色 - 家具
    COLOR_CHARGER = (0, 100, 255)      # 藍色 - 充電座
    COLOR_GARBAGE = (50, 50, 50)       # 深灰色 - 垃圾
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
                - n: 房間大小 (n × n)
                - k: 家具的最小連續格數
                - p: 每個空位每回合生成垃圾的機率
                - initial_energy: 機器人初始能量
                - e_move: 移動消耗的能量
                - e_charge: 充電增加的能量
                - e_collision: 碰撞消耗的能量
                - n_steps: 一局的總回合數
        """
        # 儲存配置參數
        self.n = config['n']
        self.k = config['k']
        self.p = config['p']
        self.initial_energy = config['initial_energy']
        self.e_move = config['e_move']
        self.e_charge = config['e_charge']
        self.e_collision = config['e_collision']
        self.n_steps = config['n_steps']

        # 初始化地圖
        self.static_grid = None   # 靜態地圖（家具、充電座）
        self.dynamic_grid = None  # 動態地圖（垃圾）

        # 初始化機器人列表
        self.robots = []

        # 回合計數器
        self.current_step = 0

        # Pygame 相關
        self.screen = None
        self.clock = None
        self.cell_size = 50  # 每個格子的像素大小
        self.info_panel_width = 300  # 資訊面板寬度

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
        # 1. 初始化地圖
        self.static_grid = np.zeros((self.n, self.n), dtype=np.int32)
        self.dynamic_grid = np.zeros((self.n, self.n), dtype=np.int32)

        # 2. 在四個角落放置充電座
        for y, x in self.charger_positions:
            self.static_grid[y, x] = self.CHARGER

        # 3. 隨機生成家具
        self._generate_furniture()

        # 4. 初始化4台機器人，放在四個角落
        self.robots = []
        for i, (y, x) in enumerate(self.charger_positions):
            robot = {
                'id': i,
                'x': x,
                'y': y,
                'energy': self.initial_energy,
                'garbage_collected': 0,
                'is_active': True,
                'charge_count': 0
            }
            self.robots.append(robot)

        # 5. 重置回合計數
        self.current_step = 0

        return self.get_global_state()

    def _generate_furniture(self):
        """
        使用隨機漫步演算法生成至少 k 格相連的家具
        確保不會擋住充電座或角落
        """
        # 從中間區域選擇一個起始點
        mid = self.n // 2
        start_x = random.randint(mid - self.n // 4, mid + self.n // 4)
        start_y = random.randint(mid - self.n // 4, mid + self.n // 4)

        # 確保起始點不在充電座上
        while (start_y, start_x) in self.charger_positions:
            start_x = random.randint(1, self.n - 2)
            start_y = random.randint(1, self.n - 2)

        # 使用隨機漫步生成家具
        furniture_cells = set()
        furniture_cells.add((start_y, start_x))

        current_x, current_y = start_x, start_y

        # 生成至少 k 格家具
        while len(furniture_cells) < self.k:
            # 隨機選擇一個方向
            direction = random.randint(0, 3)

            if direction == 0:  # 上
                new_y, new_x = current_y - 1, current_x
            elif direction == 1:  # 下
                new_y, new_x = current_y + 1, current_x
            elif direction == 2:  # 左
                new_y, new_x = current_y, current_x - 1
            else:  # 右
                new_y, new_x = current_y, current_x + 1

            # 檢查邊界和充電座
            if (0 <= new_x < self.n and 0 <= new_y < self.n and
                (new_y, new_x) not in self.charger_positions):
                furniture_cells.add((new_y, new_x))
                current_x, current_y = new_x, new_y

        # 可選：繼續擴展到更多格子（讓家具更大）
        additional_cells = random.randint(self.k // 2, self.k)
        for _ in range(additional_cells):
            if furniture_cells:
                # 從現有家具中隨機選一個格子
                current_y, current_x = random.choice(list(furniture_cells))
                direction = random.randint(0, 3)

                if direction == 0:
                    new_y, new_x = current_y - 1, current_x
                elif direction == 1:
                    new_y, new_x = current_y + 1, current_x
                elif direction == 2:
                    new_y, new_x = current_y, current_x - 1
                else:
                    new_y, new_x = current_y, current_x + 1

                if (0 <= new_x < self.n and 0 <= new_y < self.n and
                    (new_y, new_x) not in self.charger_positions):
                    furniture_cells.add((new_y, new_x))

        # 將家具設置到地圖上
        for y, x in furniture_cells:
            self.static_grid[y, x] = self.FURNITURE

    def step(self, actions: List[int]):
        """
        執行一個時間步

        Args:
            actions: 包含4個動作的列表 [action_0, action_1, action_2, action_3]
                     每個動作是 0-4 的整數
        """
        # 1. 只處理活躍的機器人
        active_robots = [r for r in self.robots if r['is_active']]

        # 2. 計算每個機器人的預定位置
        planned_positions = []
        for i, robot in enumerate(self.robots):
            if not robot['is_active']:
                planned_positions.append((robot['y'], robot['x']))
                continue

            action = actions[i]

            if action == self.ACTION_STAY:
                # 停留在原地
                planned_positions.append((robot['y'], robot['x']))
            elif action == self.ACTION_UP:
                planned_positions.append((robot['y'] - 1, robot['x']))
            elif action == self.ACTION_DOWN:
                planned_positions.append((robot['y'] + 1, robot['x']))
            elif action == self.ACTION_LEFT:
                planned_positions.append((robot['y'], robot['x'] - 1))
            elif action == self.ACTION_RIGHT:
                planned_positions.append((robot['y'], robot['x'] + 1))
            else:
                # 無效動作，視為停留
                planned_positions.append((robot['y'], robot['x']))

        # 3. 處理動作與碰撞
        for i, robot in enumerate(self.robots):
            if not robot['is_active']:
                continue

            action = actions[i]
            planned_y, planned_x = planned_positions[i]

            if action == self.ACTION_STAY:
                # 停留動作
                if self.static_grid[robot['y'], robot['x']] == self.CHARGER:
                    # 在充電座上，進行充電
                    robot['energy'] += self.e_charge
                    robot['charge_count'] += 1
                # 如果不在充電座上，不消耗能量
            else:
                # 移動動作
                collision = False

                # 碰撞檢測1: 邊界和家具
                if not (0 <= planned_x < self.n and 0 <= planned_y < self.n):
                    collision = True  # 超出邊界
                elif self.static_grid[planned_y, planned_x] == self.FURNITURE:
                    collision = True  # 撞到家具

                # 碰撞檢測2: 其他機器人
                if not collision:
                    for j, other_robot in enumerate(self.robots):
                        if i != j and other_robot['is_active']:
                            if (other_robot['x'] == planned_x and
                                other_robot['y'] == planned_y):
                                collision = True
                                break

                # 結算移動
                if collision:
                    # 移動失敗，留在原地，消耗碰撞能量
                    robot['energy'] -= self.e_collision
                else:
                    # 移動成功
                    robot['x'] = planned_x
                    robot['y'] = planned_y
                    robot['energy'] -= self.e_move

        # 4. 處理撿垃圾（在所有機器人移動後）
        for robot in self.robots:
            if not robot['is_active']:
                continue

            x, y = robot['x'], robot['y']
            if self.dynamic_grid[y, x] == 1:
                # 撿起垃圾
                robot['garbage_collected'] += 1
                self.dynamic_grid[y, x] = 0

        # 5. 更新關機狀態
        for robot in self.robots:
            # 能量不超過上限，不低於0
            robot['energy'] = max(0, min(self.initial_energy, robot['energy']))

            # 如果能量耗盡，設為非活躍
            if robot['energy'] <= 0:
                robot['is_active'] = False

        # 6. 生成新垃圾
        self._generate_garbage()

        # 7. 回合計數
        self.current_step += 1

        # 8. 檢查結束
        done = self.current_step >= self.n_steps

        return self.get_global_state(), done

    def _generate_garbage(self):
        """
        在空地上以機率 p 生成新垃圾
        確保有機器人的位置不會生成垃圾
        """
        # 獲取所有機器人的位置
        robot_positions = set()
        for robot in self.robots:
            robot_positions.add((robot['y'], robot['x']))

        # 遍歷所有格子
        for y in range(self.n):
            for x in range(self.n):
                # 只在空地上生成垃圾
                if self.static_grid[y, x] == self.EMPTY:
                    # 該格子沒有垃圾且沒有機器人
                    if (self.dynamic_grid[y, x] == 0 and
                        (y, x) not in robot_positions):
                        # 以機率 p 生成垃圾
                        if random.random() < self.p:
                            self.dynamic_grid[y, x] = 1

    def get_global_state(self) -> Dict[str, Any]:
        """
        獲取全域狀態

        Returns:
            包含地圖和機器人狀態的字典
        """
        return {
            'static_grid': self.static_grid.copy(),
            'dynamic_grid': self.dynamic_grid.copy(),
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
            pygame.display.set_caption('多機器人清掃模擬器')
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

                # 繪製靜態元素（家具、充電座）
                if self.static_grid[y, x] == self.FURNITURE:
                    pygame.draw.rect(self.screen, self.COLOR_FURNITURE, rect)
                elif self.static_grid[y, x] == self.CHARGER:
                    pygame.draw.rect(self.screen, self.COLOR_CHARGER, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_EMPTY, rect)

                # 繪製網格線
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                # 繪製垃圾
                if self.dynamic_grid[y, x] == 1:
                    center_x = x * self.cell_size + self.cell_size // 2
                    center_y = y * self.cell_size + self.cell_size // 2
                    pygame.draw.circle(
                        self.screen,
                        self.COLOR_GARBAGE,
                        (center_x, center_y),
                        self.cell_size // 6
                    )

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

        # 設置字體
        font_title = pygame.font.Font(None, 28)
        font_text = pygame.font.Font(None, 22)

        # 標題
        title = font_title.render('機器人狀態', True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 10, 10))

        # 顯示回合數
        step_text = font_text.render(
            f'回合: {self.current_step}/{self.n_steps}',
            True,
            (255, 255, 255)
        )
        self.screen.blit(step_text, (panel_x + 10, 40))

        # 繪製每台機器人的狀態
        y_offset = 80
        for robot in self.robots:
            # 機器人標題
            robot_title = font_text.render(
                f'機器人 {robot["id"]}',
                True,
                self.ROBOT_COLORS[robot['id']]
            )
            self.screen.blit(robot_title, (panel_x + 10, y_offset))
            y_offset += 25

            # 能量條
            energy_text = font_text.render(
                f'能量: {robot["energy"]}/{self.initial_energy}',
                True,
                (255, 255, 255)
            )
            self.screen.blit(energy_text, (panel_x + 10, y_offset))
            y_offset += 20

            # 能量條圖形
            bar_width = panel_width - 30
            bar_height = 15
            energy_ratio = robot['energy'] / self.initial_energy

            # 背景條
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (panel_x + 10, y_offset, bar_width, bar_height)
            )

            # 能量條
            if energy_ratio > 0:
                color = (0, 255, 0) if energy_ratio > 0.3 else (255, 0, 0)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (panel_x + 10, y_offset, bar_width * energy_ratio, bar_height)
                )

            y_offset += 20

            # 垃圾數
            garbage_text = font_text.render(
                f'垃圾: {robot["garbage_collected"]}',
                True,
                (255, 255, 255)
            )
            self.screen.blit(garbage_text, (panel_x + 10, y_offset))
            y_offset += 20

            # 充電次數
            charge_text = font_text.render(
                f'充電: {robot["charge_count"]} 次',
                True,
                (255, 255, 255)
            )
            self.screen.blit(charge_text, (panel_x + 10, y_offset))
            y_offset += 20

            # 狀態
            status = '運行中' if robot['is_active'] else '已停機'
            status_color = (0, 255, 0) if robot['is_active'] else (255, 0, 0)
            status_text = font_text.render(f'狀態: {status}', True, status_color)
            self.screen.blit(status_text, (panel_x + 10, y_offset))
            y_offset += 35

    def close(self):
        """
        關閉 Pygame 視窗
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None


def main():
    """
    主函數：示範如何使用 RobotVacuumEnv
    """
    # 配置環境參數
    config = {
        'n': 15,              # 15×15 的房間
        'k': 10,              # 家具至少 10 格
        'p': 0.05,            # 5% 機率生成垃圾
        'initial_energy': 100,  # 初始能量 100
        'e_move': 1,          # 移動消耗 1 能量
        'e_charge': 10,       # 充電增加 10 能量
        'e_collision': 5,     # 碰撞消耗 5 能量
        'n_steps': 500        # 總共 500 回合
    }

    # 創建環境
    env = RobotVacuumEnv(config)

    # 重置環境
    state = env.reset()
    print("環境已初始化！")
    print(f"房間大小: {config['n']}×{config['n']}")
    print(f"總回合數: {config['n_steps']}")
    print("\n開始模擬...")
    print("提示：關閉視窗或按 Ctrl+C 可結束模擬\n")

    # 先渲染一次初始狀態（初始化 Pygame）
    env.render()

    # 模擬循環
    done = False
    try:
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

            # 每100回合顯示統計
            if env.current_step % 100 == 0:
                total_garbage = sum(r['garbage_collected'] for r in env.robots)
                active_robots = sum(1 for r in env.robots if r['is_active'])
                print(f"回合 {env.current_step}: "
                      f"總垃圾={total_garbage}, "
                      f"活躍機器人={active_robots}")

    except KeyboardInterrupt:
        print("\n模擬被中斷")

    finally:
        # 顯示最終統計
        print("\n=== 模擬結束 ===")
        print(f"總回合數: {env.current_step}")
        for robot in env.robots:
            print(f"機器人 {robot['id']}: "
                  f"垃圾={robot['garbage_collected']}, "
                  f"能量={robot['energy']}, "
                  f"充電={robot['charge_count']}次, "
                  f"狀態={'運行中' if robot['is_active'] else '已停機'}")

        # 關閉環境
        env.close()


if __name__ == '__main__':
    main()
