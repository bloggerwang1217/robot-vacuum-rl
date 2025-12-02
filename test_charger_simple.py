#!/usr/bin/env python3
"""
簡單測試充電器配置邏輯
"""

def parse_charger_positions(positions_str):
    """解析充電器位置字符串"""
    if positions_str is None:
        return None

    charger_positions = []
    for pos_str in positions_str.split(';'):
        y, x = map(int, pos_str.split(','))
        charger_positions.append((y, x))
    return charger_positions

def filter_invalid_positions(positions, n):
    """過濾無效位置"""
    if positions is None:
        # 使用預設值
        return [
            (0, 0),
            (0, n - 1),
            (n - 1, 0),
            (n - 1, n - 1)
        ]

    valid_positions = []
    for y, x in positions:
        # 檢查是否為無效座標標記或超出邊界
        if (y == -1 and x == -1) or y < 0 or x < 0 or y >= n or x >= n:
            continue
        valid_positions.append((y, x))

    if len(valid_positions) == 0:
        raise ValueError("至少需要 1 個有效的充電器位置")

    return valid_positions

def calculate_obs_dim(num_chargers):
    """計算觀察空間維度"""
    # 自身位置2 + 自身能量1 + 其他機器人3*3 + 充電座2*N
    return 12 + 2 * num_chargers

def test_case(name, positions_str, n=3):
    """測試一個案例"""
    print(f"\n{'='*50}")
    print(f"測試: {name}")
    print(f"輸入: {positions_str}")

    try:
        parsed = parse_charger_positions(positions_str)
        print(f"解析結果: {parsed}")

        filtered = filter_invalid_positions(parsed, n)
        print(f"過濾後: {filtered}")

        obs_dim = calculate_obs_dim(len(filtered))
        print(f"充電器數量: {len(filtered)}")
        print(f"觀察空間維度: {obs_dim}")

        # 檢查機器人初始位置的 home_charger
        robot_positions = [
            (0, 0),
            (0, n - 1),
            (n - 1, 0),
            (n - 1, n - 1)
        ]

        print(f"\n機器人 home_charger 設置:")
        for i, (y, x) in enumerate(robot_positions):
            has_charger = (y, x) in filtered
            home_charger = (y, x) if has_charger else None
            home_str = f"({home_charger[0]},{home_charger[1]})" if home_charger else "None"
            print(f"  Robot {i} at ({y},{x}): home_charger = {home_str}")

        print(f"✓ 測試通過")
        return True

    except Exception as e:
        print(f"✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("充電器配置邏輯測試")

    # 測試案例
    test_case("預設配置", None, n=3)
    test_case("四個角落", "0,0;0,2;2,0;2,2", n=3)
    test_case("兩個充電器", "0,0;2,2", n=3)
    test_case("一個充電器", "1,1", n=3)
    test_case("三個充電器", "0,0;0,2;2,0", n=3)
    test_case("使用-1,-1禁用", "0,0;-1,-1;2,0;2,2", n=3)
    test_case("過濾無效座標", "0,0;0,2;5,5;2,2", n=3)
    test_case("過濾負數座標", "0,0;-5,-3;2,2", n=3)

    print(f"\n{'='*50}")
    print("所有測試完成！")

if __name__ == "__main__":
    main()
