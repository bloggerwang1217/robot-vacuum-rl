"""
範例配置檔案
提供不同難度和場景的配置
"""

# 簡單模式：小房間，少障礙，低垃圾生成率
EASY_CONFIG = {
    'n': 10,              # 10×10 房間
    'k': 5,               # 家具至少 5 格
    'p': 0.02,            # 2% 垃圾生成率
    'initial_energy': 150,  # 較高初始能量
    'e_move': 1,          # 移動消耗 1 能量
    'e_charge': 15,       # 充電增加 15 能量
    'e_collision': 3,     # 碰撞消耗 3 能量
    'n_steps': 300        # 300 回合
}

# 中等模式：標準設定
MEDIUM_CONFIG = {
    'n': 15,              # 15×15 房間
    'k': 10,              # 家具至少 10 格
    'p': 0.05,            # 5% 垃圾生成率
    'initial_energy': 100,  # 標準初始能量
    'e_move': 1,          # 移動消耗 1 能量
    'e_charge': 10,       # 充電增加 10 能量
    'e_collision': 5,     # 碰撞消耗 5 能量
    'n_steps': 500        # 500 回合
}

# 困難模式：大房間，多障礙，高垃圾生成率
HARD_CONFIG = {
    'n': 20,              # 20×20 房間
    'k': 20,              # 家具至少 20 格
    'p': 0.08,            # 8% 垃圾生成率
    'initial_energy': 80,   # 較低初始能量
    'e_move': 2,          # 移動消耗 2 能量
    'e_charge': 8,        # 充電增加 8 能量
    'e_collision': 8,     # 碰撞消耗 8 能量
    'n_steps': 800        # 800 回合
}

# 快速測試模式：用於快速驗證
TEST_CONFIG = {
    'n': 8,               # 8×8 小房間
    'k': 3,               # 家具至少 3 格
    'p': 0.1,             # 10% 垃圾生成率（高）
    'initial_energy': 50,   # 低初始能量
    'e_move': 1,          # 移動消耗 1 能量
    'e_charge': 5,        # 充電增加 5 能量
    'e_collision': 3,     # 碰撞消耗 3 能量
    'n_steps': 100        # 100 回合（短）
}

# 能量挑戰模式：考驗能量管理
ENERGY_CHALLENGE_CONFIG = {
    'n': 15,              # 15×15 房間
    'k': 12,              # 家具至少 12 格
    'p': 0.05,            # 5% 垃圾生成率
    'initial_energy': 60,   # 低初始能量
    'e_move': 2,          # 移動消耗 2 能量（高）
    'e_charge': 5,        # 充電增加 5 能量（低）
    'e_collision': 10,    # 碰撞消耗 10 能量（高）
    'n_steps': 400        # 400 回合
}

# 垃圾爆炸模式：極高垃圾生成率
GARBAGE_EXPLOSION_CONFIG = {
    'n': 12,              # 12×12 房間
    'k': 8,               # 家具至少 8 格
    'p': 0.15,            # 15% 垃圾生成率（極高）
    'initial_energy': 120,  # 高初始能量
    'e_move': 1,          # 移動消耗 1 能量
    'e_charge': 12,       # 充電增加 12 能量
    'e_collision': 4,     # 碰撞消耗 4 能量
    'n_steps': 600        # 600 回合
}


def get_config(difficulty='medium'):
    """
    根據難度獲取配置

    Args:
        difficulty: 'easy', 'medium', 'hard', 'test', 'energy', 'garbage'

    Returns:
        配置字典
    """
    configs = {
        'easy': EASY_CONFIG,
        'medium': MEDIUM_CONFIG,
        'hard': HARD_CONFIG,
        'test': TEST_CONFIG,
        'energy': ENERGY_CHALLENGE_CONFIG,
        'garbage': GARBAGE_EXPLOSION_CONFIG
    }

    return configs.get(difficulty, MEDIUM_CONFIG)


if __name__ == '__main__':
    # 顯示所有配置
    print("可用的配置模式：\n")

    configs = {
        '簡單模式 (easy)': EASY_CONFIG,
        '中等模式 (medium)': MEDIUM_CONFIG,
        '困難模式 (hard)': HARD_CONFIG,
        '測試模式 (test)': TEST_CONFIG,
        '能量挑戰 (energy)': ENERGY_CHALLENGE_CONFIG,
        '垃圾爆炸 (garbage)': GARBAGE_EXPLOSION_CONFIG
    }

    for name, config in configs.items():
        print(f"=== {name} ===")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()
