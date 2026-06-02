import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_cmap(hex_color):
    """创建从白色到指定颜色的渐变色"""
    r = (hex_color >> 16) / 255.0
    g = ((hex_color >> 8) & 0xFF) / 255.0
    b = (hex_color & 0xFF) / 255.0
    return LinearSegmentedColormap.from_list('custom', ['white', (r, g, b)])


def read_data(filename):
    """读取数据文件，确保不删除前导0"""
    with open(filename, 'r') as f:
        lines = []
        for line in f:
            stripped_line = line.strip()  # 只移除首尾空白字符，保留数字0
            if stripped_line:  # 跳过空行
                lines.append(stripped_line)

    # 检查所有行长度是否一致
    lengths = [len(line) for line in lines]
    if len(set(lengths)) > 1:
        print(f"错误: 行长度不一致 - 发现以下长度: {set(lengths)}")
        print("前5行示例:")
        for i in range(min(5, len(lines))):
            print(f"行 {i + 1}: {lines[i]} (长度: {len(lines[i])})")
        raise ValueError("所有行的长度必须相同")

    print("数据读取成功，前5行示例:")
    for i in range(min(5, len(lines))):
        print(f"行 {i + 1}: {lines[i]}")

    return lines, lengths[0]


def count_bits(data, length):
    """统计每一位的0和1数量"""
    count_0 = np.zeros(length)
    count_1 = np.zeros(length)

    for line in data:
        if len(line) != length:
            raise ValueError(f"行长度不一致: 预期 {length}, 实际 {len(line)}")
        for i, bit in enumerate(line):
            if bit == '0':
                count_0[i] += 1
            elif bit == '1':
                count_1[i] += 1
            else:
                raise ValueError(f"非法字符 '{bit}' 在位置 {i}")

    return count_0, count_1


def plot_bit_counts(count_0, count_1):
    """绘制0和1的统计图"""
    positions = np.arange(len(count_0)) + 1  # 位的位置从1开始

    # 创建自定义颜色映射
    blue_cmap = create_cmap(0xABBBDD)  # 蓝色系
    pink_cmap = create_cmap(0xF4B3CB)  # 粉色系

    # 计算颜色强度
    max_count = max(max(count_0), max(count_1))
    blue_intensity = count_1 / max_count if max_count > 0 else np.zeros_like(count_1)
    pink_intensity = count_0 / max_count if max_count > 0 else np.zeros_like(count_0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制1的柱状图（向上）
    bars1 = ax.bar(positions, count_1, color=blue_cmap(blue_intensity),
                   edgecolor='black', linewidth=0.5)

    # 绘制0的柱状图（向下）
    bars0 = ax.bar(positions, -count_0, color=pink_cmap(pink_intensity),
                   edgecolor='black', linewidth=0.5)

    # 添加参考线
    ax.axhline(0, color='black', linewidth=0.8)

    plt.rcParams['font.family'] = 'Arial'
    plt.xlabel('Feature Position', fontweight='bold', fontsize=16)
    # 设置图表属性
    plt.ylabel('Count', fontweight='bold', fontsize=16)

    ax.set_xticks(positions)
    plt.grid(axis='y', alpha=0.3)

    # 添加图例
    ax.bar(0, 0, color=blue_cmap(1), label='1(blue)')
    ax.bar(0, 0, color=pink_cmap(1), label='0(pink)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('gpt_1.png', dpi=1000, bbox_inches='tight')
    plt.show()


def main():
    try:
        # 假设数据文件名为CPP_Code.txt
        filename = './gpt/cpp_1.txt'
        print(f"正在读取文件: {filename}")
        data, length = read_data(filename)
        print(f"总行数: {len(data)}, 每行长度: {length}")

        count_0, count_1 = count_bits(data, length)
        print("\n位数统计结果:")
        print("位置\t0的个数\t1的个数")
        for i in range(length):
            print(f"{i + 1}\t{int(count_0[i])}\t{int(count_1[i])}")

        plot_bit_counts(count_0, count_1)
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == '__main__':
    main()