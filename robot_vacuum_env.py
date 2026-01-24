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
                - num_robots: 機器人數量 (1-4)，預設 4
                - initial_energy: 機器人初始能量 (統一設定)
                - robot_energies: 個別機器人初始能量列表 [robot_0, robot_1, robot_2, robot_3]
                - e_move: 移動消耗的能量
                - e_charge: 充電增加的能量
                - e_collision: 碰撞消耗的能量
                - n_steps: 一局的總回合數
                - epsilon: 探索率（用於 epsilon-greedy 策略）
                - charger_positions: 充電器位置列表，預設為四個角落，
                                    可指定 1-4 個位置，無效座標（如 (-1,-1) 或超出邊界）會被過濾
        """
        # 儲存配置參數
        self.n = config.get('n', 3)  # 預設 3x3
        self.num_robots = config.get('num_robots', 4)  # 預設 4 台機器人
        self.initial_energy = config['initial_energy']
        # 支援個別機器人血量或統一血量
        self.robot_energies = config.get('robot_energies', [self.initial_energy] * self.num_robots)
        self.e_move = config['e_move']
        self.e_charge = config['e_charge']
        # 碰撞傷害參數
        # e_collision: 機器人之間的碰撞傷害
        # e_boundary: 撞牆的傷害（通常設置更高以避免撞牆）
        self.e_collision = config.get('e_collision', 3)
        self.e_boundary = config.get('e_boundary', 50)  # 撞牆懲罰（預設 50）
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

        # 充電座位置（可配置，預設為四個角落）
        default_charger_positions = [
            (0, 0),
            (0, self.n - 1),
            (self.n - 1, 0),
            (self.n - 1, self.n - 1)
        ]
        # 使用 or 處理 None 的情況
        configured_positions = config.get('charger_positions') or default_charger_positions

        # 過濾無效座標：(-1, -1) 或超出邊界的座標
        self.charger_positions = []
        for pos in configured_positions:
            y, x = pos
            # 檢查是否為無效座標標記或超出邊界
            if (y == -1 and x == -1) or y < 0 or x < 0 or y >= self.n or x >= self.n:
                continue  # 跳過無效座標
            self.charger_positions.append(pos)

        # 確保至少有 1 個充電器
        if len(self.charger_positions) == 0:
            raise ValueError("至少需要 1 個有效的充電器位置")

    def reset(self) -> Dict[str, Any]:
        """
        重置環境到初始狀態

        Returns:
            初始狀態字典
        """
        # 1. 初始化地圖（只有空地和充電座）
        self.static_grid = np.zeros((self.n, self.n), dtype=np.int32)

        # 2. 放置充電座（可配置位置）
        for y, x in self.charger_positions:
            self.static_grid[y, x] = self.CHARGER

        # 3. 初始化機器人，固定放在四個角落（無論充電器在哪）
        all_start_positions = [
            (0, 0),
            (0, self.n - 1),
            (self.n - 1, 0),
            (self.n - 1, self.n - 1)
        ]
        robot_start_positions = all_start_positions[:self.num_robots]

        # 計算全域最大血量（所有 robot 共用，弱者可以充電到這個上限）
        global_max_energy = max(self.robot_energies)

        self.robots = []
        for i, (y, x) in enumerate(robot_start_positions):
            # 檢查機器人初始位置是否有充電器
            has_charger_at_start = (y, x) in self.charger_positions
            robot = {
                'id': i,
                'x': x,
                'y': y,
                'energy': self.robot_energies[i],  # 使用個別初始血量
                'max_energy': global_max_energy,  # 所有人共用最大血量上限
                'is_active': True,
                'is_mover_this_step': False,  # 本回合是否主動移動 (用於 kill 歸屬分析)
                'charge_count': 0,
                'non_home_charge_count': 0,  # 在非初始充電座充電的次數
                'home_charger': (y, x) if has_charger_at_start else None,  # 只在初始位置有充電器時記錄
                'active_collision_count': 0,  # 主動移動過去碰撞的次數
                'passive_collision_count': 0,  # 被碰撞的次數
                'active_collisions_with': {j: 0 for j in range(self.num_robots)},  # 主動碰撞各機器人的次數
                'collided_with_agent_id': None,  # 本回合碰撞的對象 (用於 kill 分析，僅記錄最後一次)
                'collided_by_counts': {j: 0 for j in range(self.num_robots)},  # 被各個機器人碰撞的次數
                'collision_events': []  # 完整碰撞歷史 [(step, attacker_id, collision_type), ...]
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
        knockback_targets = {}  # {robot_id: new_pos} - 被推開的機器人的新位置

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
                # 找到被碰撞的靜止機器人
                victim_id = None
                for stay_id, stay_pos in staying_robots.items():
                    if planned_pos == stay_pos:
                        victim_id = stay_id
                        break

                if victim_id is not None:
                    # 計算推回方向：從 moving robot 的原位置指向 victim
                    mover_y, mover_x = self.robots[robot_id]['y'], self.robots[robot_id]['x']
                    victim_y, victim_x = self.robots[victim_id]['y'], self.robots[victim_id]['x']

                    # 推回方向 = victim 位置 - mover 原位置
                    dy = victim_y - mover_y
                    dx = victim_x - mover_x

                    # victim 的新位置（被推出去）
                    new_victim_y = victim_y + dy
                    new_victim_x = victim_x + dx

                    # 檢查新位置是否有效（在地圖內且沒有其他機器人）
                    if 0 <= new_victim_x < self.n and 0 <= new_victim_y < self.n:
                        # 檢查是否有其他移動機器人試圖進入該位置
                        can_push = True
                        for other_id, other_planned in moving_robots.items():
                            if other_id != robot_id and other_planned == (new_victim_y, new_victim_x):
                                can_push = False
                                break

                        # 檢查是否有其他停留機器人在該位置
                        if can_push:
                            for stay_id, stay_pos in staying_robots.items():
                                if stay_id != victim_id and stay_pos == (new_victim_y, new_victim_x):
                                    can_push = False
                                    break

                        if can_push:
                            # 成功推回
                            collision_events[robot_id] = "knockback_success"
                            knockback_targets[victim_id] = (new_victim_y, new_victim_x)
                            self.robots[robot_id]['collided_with_agent_id'] = victim_id
                            self.robots[victim_id]['collided_with_agent_id'] = robot_id
                            # 記錄完整碰撞歷史
                            self.robots[robot_id]['collision_events'].append({
                                'step': self.current_step,
                                'opponent_id': victim_id,
                                'collision_type': 'knockback_success'
                            })
                            self.robots[victim_id]['collision_events'].append({
                                'step': self.current_step,
                                'opponent_id': robot_id,
                                'collision_type': 'knockback_success'
                            })
                        else:
                            # 無路可推
                            collision_events[robot_id] = "stationary_blocked"
                            self.robots[robot_id]['collided_with_agent_id'] = victim_id
                            self.robots[victim_id]['collided_with_agent_id'] = robot_id
                            # 記錄完整碰撞歷史
                            self.robots[robot_id]['collision_events'].append({
                                'step': self.current_step,
                                'opponent_id': victim_id,
                                'collision_type': 'stationary_blocked'
                            })
                            self.robots[victim_id]['collision_events'].append({
                                'step': self.current_step,
                                'opponent_id': robot_id,
                                'collision_type': 'stationary_blocked'
                            })
                    else:
                        # 無路可推（邊界外）
                        collision_events[robot_id] = "stationary_blocked"
                        self.robots[robot_id]['collided_with_agent_id'] = victim_id
                        self.robots[victim_id]['collided_with_agent_id'] = robot_id
                        # 記錄完整碰撞歷史
                        self.robots[robot_id]['collision_events'].append({
                            'step': self.current_step,
                            'opponent_id': victim_id,
                            'collision_type': 'stationary_blocked'
                        })
                        self.robots[victim_id]['collision_events'].append({
                            'step': self.current_step,
                            'opponent_id': robot_id,
                            'collision_type': 'stationary_blocked'
                        })

            elif planned_pos in contested_cells:
                collision_events[robot_id] = "contested"
                for other_id, other_pos in moving_robots.items():
                    if robot_id != other_id and planned_pos == other_pos:
                        self.robots[robot_id]['collided_with_agent_id'] = other_id
                        # 記錄完整碰撞歷史（只記錄一次，避免重複）
                        if robot_id < other_id:  # 只讓較小的 ID 記錄，避免兩次記錄同一碰撞
                            self.robots[robot_id]['collision_events'].append({
                                'step': self.current_step,
                                'opponent_id': other_id,
                                'collision_type': 'contested'
                            })
                            self.robots[other_id]['collision_events'].append({
                                'step': self.current_step,
                                'opponent_id': robot_id,
                                'collision_type': 'contested'
                            })
                        break
        
        # 2d. 檢查移動機器人的目標位置是否是被阻擋機器人的當前位置
        # 被阻擋的機器人（在 collision_events 中）不會移動，所以他們的當前位置仍被佔據
        # 需要迭代處理，因為新被阻擋的機器人也會阻擋其他機器人
        changed = True
        while changed:
            changed = False
            blocked_positions = {}  # {pos: robot_id} - 被阻擋機器人的當前位置
            for robot_id in collision_events:
                if robot_id in moving_robots:
                    # 這個機器人被阻擋了，他的當前位置仍被佔據
                    robot = self.robots[robot_id]
                    blocked_positions[(robot['y'], robot['x'])] = robot_id
            
            # 檢查其他移動機器人是否想進入被阻擋的位置
            for robot_id, planned_pos in moving_robots.items():
                if robot_id in collision_events:
                    continue  # 已經有碰撞事件
                if planned_pos in blocked_positions:
                    # 目標位置被一個不能移動的機器人佔據
                    blocker_id = blocked_positions[planned_pos]
                    collision_events[robot_id] = "blocked_by_stuck_robot"
                    self.robots[robot_id]['collided_with_agent_id'] = blocker_id
                    self.robots[robot_id]['active_collision_count'] += 1
                    self.robots[robot_id]['active_collisions_with'][blocker_id] += 1
                    # 記錄碰撞歷史
                    self.robots[robot_id]['collision_events'].append({
                        'step': self.current_step,
                        'opponent_id': blocker_id,
                        'collision_type': 'blocked_by_stuck_robot'
                    })
                    changed = True  # 有新的阻擋，需要再次檢查
        
        # 3. 結算所有機器的狀態
        # 3a. 處理移動的機器人
        for robot_id, planned_pos in moving_robots.items():
            robot = self.robots[robot_id]
            if robot_id in collision_events:
                reason = collision_events[robot_id]

                if reason == "knockback_success":
                    # 成功推開對方：attacker 進入 victim 的位置，無碰撞傷害
                    robot['y'], robot['x'] = planned_pos
                    robot['active_collision_count'] += 1
                    if robot['collided_with_agent_id'] is not None:
                        robot['active_collisions_with'][robot['collided_with_agent_id']] += 1

                elif reason == "stationary_blocked":
                    # 無路可推：attacker 停留，無碰撞傷害
                    robot['active_collision_count'] += 1
                    if robot['collided_with_agent_id'] is not None:
                        robot['active_collisions_with'][robot['collided_with_agent_id']] += 1

                elif reason in ["swap", "contested"]:
                    # 互撞：雙方停留，都受傷
                    robot['energy'] -= self.e_collision
                    robot['active_collision_count'] += 1
                    if robot['collided_with_agent_id'] is not None:
                        robot['active_collisions_with'][robot['collided_with_agent_id']] += 1

                elif reason == "blocked_by_stuck_robot":
                    # 目標位置被一個不能移動的機器人佔據：停留原位，不受傷
                    # (碰撞計數已在 2d 中處理)
                    pass

                elif reason == "boundary":
                    # 撞牆：停留原位，受高額懲罰（使用 e_boundary）
                    robot['energy'] -= self.e_boundary
                    robot['active_collision_count'] += 1
                    # 記錄撞牆事件
                    robot['collision_events'].append({
                        'step': self.current_step,
                        'opponent_id': -1,  # -1 表示邊界
                        'collision_type': 'boundary'
                    })
            else:
                # 移動成功
                robot['y'], robot['x'] = planned_pos

        # 3b. 處理停留的機器人（包括被推開的機器人）
        for robot_id, pos in staying_robots.items():
            robot = self.robots[robot_id]

            # 檢查是否被推開
            if robot_id in knockback_targets:
                # 被推開：移動到新位置，受傷害
                new_pos = knockback_targets[robot_id]
                robot['y'], robot['x'] = new_pos
                robot['energy'] -= self.e_collision
                robot['passive_collision_count'] += 1
                aggressor_id = robot['collided_with_agent_id']
                if aggressor_id is not None:
                    robot['collided_by_counts'][aggressor_id] += 1
            else:
                # 沒有被推開：停留原位
                # 如果被撞且沒有被推開（即 stationary_blocked 情況）
                if robot['collided_with_agent_id'] is not None:
                    robot['energy'] -= self.e_collision
                    robot['passive_collision_count'] += 1
                    aggressor_id = robot['collided_with_agent_id']
                    robot['collided_by_counts'][aggressor_id] += 1

        # 3c. 檢查所有機器人最終位置是否在充電座上（無論執行什麼動作）
        # 先充電，再扣生存成本（讓敵人更有動機去充電座）
        for i, robot in enumerate(self.robots):
            if robot['is_active']:
                if self.static_grid[robot['y'], robot['x']] == self.CHARGER:
                    robot['energy'] += self.e_charge
                    robot['charge_count'] += 1
                    # 只在有 home_charger 的情況下才統計 non_home_charge
                    if robot['home_charger'] is not None and (robot['y'], robot['x']) != robot['home_charger']:
                        robot['non_home_charge_count'] += 1

        # 3d. 扣除所有活著機器人的生存成本
        for robot in self.robots:
            if robot['is_active']:
                robot['energy'] -= self.e_move

        # 4. 更新所有機器人的關機狀態
        for robot in self.robots:
            robot['energy'] = max(0, min(robot['max_energy'], robot['energy']))
            if robot['energy'] <= 0:
                robot['is_active'] = False

        # 5. 回合計數
        self.current_step += 1
        done = self.current_step >= self.n_steps
        return self.get_global_state(), done

    def step_single(self, robot_id: int, action: int) -> Dict[str, Any]:
        """
        執行單個機器人的動作（一個一個動的版本）

        Args:
            robot_id: 機器人 ID (0 到 num_robots-1)
            action: 動作 (0-4)

        Returns:
            state: 更新後的環境狀態
        """
        robot = self.robots[robot_id]

        # 1. 清除該機器人的上一次碰撞記錄，並重置 is_mover_this_step 旗標
        robot['collided_with_agent_id'] = None
        robot['is_mover_this_step'] = False

        # 如果機器人已死亡，直接返回
        if not robot['is_active']:
            return self.get_global_state()

        # 2. 處理動作
        if action == self.ACTION_STAY:
            # 停留：不移動
            pass
        else:
            # 移動動作：設定 is_mover_this_step 為 True
            robot['is_mover_this_step'] = True

            # 計算預定位置
            py, px = robot['y'], robot['x']
            if action == self.ACTION_UP:
                py -= 1
            elif action == self.ACTION_DOWN:
                py += 1
            elif action == self.ACTION_LEFT:
                px -= 1
            elif action == self.ACTION_RIGHT:
                px += 1

            # 3. 檢查碰撞
            collision_type = None

            # 3a. 檢查是否撞牆
            if not (0 <= px < self.n and 0 <= py < self.n):
                collision_type = "boundary"
                robot['energy'] -= self.e_boundary
                robot['active_collision_count'] += 1
                robot['collision_events'].append({
                    'step': self.current_step,
                    'opponent_id': -1,
                    'collision_type': 'boundary'
                })
            else:
                # 3b. 檢查目標格是否有其他機器人
                victim_id = None
                for other in self.robots:
                    if other['id'] != robot_id and other['is_active']:
                        if other['y'] == py and other['x'] == px:
                            victim_id = other['id']
                            break

                if victim_id is not None:
                    victim = self.robots[victim_id]

                    # 嘗試 knockback：計算推回方向
                    dy = py - robot['y']
                    dx = px - robot['x']
                    new_victim_y = py + dy
                    new_victim_x = px + dx

                    # 檢查新位置是否有效
                    can_push = True
                    if not (0 <= new_victim_x < self.n and 0 <= new_victim_y < self.n):
                        can_push = False
                    else:
                        # 檢查新位置是否有其他機器人
                        for other in self.robots:
                            if other['id'] != robot_id and other['id'] != victim_id and other['is_active']:
                                if other['y'] == new_victim_y and other['x'] == new_victim_x:
                                    can_push = False
                                    break

                    if can_push:
                        # knockback_success：移動者進入 victim 位置，victim 被推開
                        collision_type = "knockback_success"
                        robot['y'], robot['x'] = py, px
                        victim['y'], victim['x'] = new_victim_y, new_victim_x
                        victim['energy'] -= self.e_collision
                        victim['passive_collision_count'] += 1
                        victim['collided_by_counts'][robot_id] += 1
                        victim['collided_with_agent_id'] = robot_id

                        robot['active_collision_count'] += 1
                        robot['active_collisions_with'][victim_id] += 1
                        robot['collided_with_agent_id'] = victim_id

                        # 記錄碰撞歷史
                        robot['collision_events'].append({
                            'step': self.current_step,
                            'opponent_id': victim_id,
                            'collision_type': 'knockback_success'
                        })
                        victim['collision_events'].append({
                            'step': self.current_step,
                            'opponent_id': robot_id,
                            'collision_type': 'knockback_success'
                        })
                    else:
                        # stationary_blocked：無路可推，移動者停住
                        collision_type = "stationary_blocked"
                        victim['energy'] -= self.e_collision
                        victim['passive_collision_count'] += 1
                        victim['collided_by_counts'][robot_id] += 1
                        victim['collided_with_agent_id'] = robot_id

                        robot['active_collision_count'] += 1
                        robot['active_collisions_with'][victim_id] += 1
                        robot['collided_with_agent_id'] = victim_id

                        # 記錄碰撞歷史
                        robot['collision_events'].append({
                            'step': self.current_step,
                            'opponent_id': victim_id,
                            'collision_type': 'stationary_blocked'
                        })
                        victim['collision_events'].append({
                            'step': self.current_step,
                            'opponent_id': robot_id,
                            'collision_type': 'stationary_blocked'
                        })
                else:
                    # 沒有碰撞，正常移動
                    robot['y'], robot['x'] = py, px

        # 4. 充電判定（無論執行什麼動作，在充電座上就充電）
        if robot['is_active'] and self.static_grid[robot['y'], robot['x']] == self.CHARGER:
            robot['energy'] += self.e_charge
            robot['charge_count'] += 1
            if robot['home_charger'] is not None and (robot['y'], robot['x']) != robot['home_charger']:
                robot['non_home_charge_count'] += 1

        # 5. 扣除生存成本
        if robot['is_active']:
            robot['energy'] -= self.e_move

        # 6. 更新能量上下限並檢查死亡
        robot['energy'] = max(0, min(robot['max_energy'], robot['energy']))
        if robot['energy'] <= 0:
            robot['is_active'] = False

        # 7. 檢查被推開的 victim 的能量（可能因為被推而死亡）
        for other in self.robots:
            if other['id'] != robot_id:
                other['energy'] = max(0, min(other['max_energy'], other['energy']))
                if other['energy'] <= 0:
                    other['is_active'] = False

        return self.get_global_state()

    def advance_step(self) -> bool:
        """
        所有 agent 都行動完後，推進回合計數

        Returns:
            done: 是否達到最大步數
        """
        self.current_step += 1
        return self.current_step >= self.n_steps

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
