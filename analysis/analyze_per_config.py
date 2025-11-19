"""
為每個配置產生詳細的 non_home_charges 分析圖表

針對每個配置，分析 non_home_charges 與其他所有指標的關係
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from scipy import stats

# 設定樣式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']  # 支援中文
plt.rcParams['axes.unicode_minus'] = False


class PerConfigAnalyzer:
    """為每個配置產生詳細分析"""

    def __init__(self, wandb_dir: str = "wandb_data"):
        self.wandb_dir = Path(wandb_dir)
        self.configs = {}

    def load_data(self):
        """載入 WandB CSV 檔案"""
        print(f"從 {self.wandb_dir} 載入資料...")

        csv_files = list(self.wandb_dir.glob("*.csv"))
        print(f"找到 {len(csv_files)} 個 CSV 檔案\n")

        for csv_file in csv_files:
            config_name = csv_file.stem
            try:
                df = pd.read_csv(csv_file)

                # 檢查是否有必要的欄位
                required_cols = ['episode', 'total_non_home_charges_per_episode']
                if all(col in df.columns for col in required_cols):
                    # 選擇所有可用的指標
                    metric_cols = [
                        'episode',
                        'total_non_home_charges_per_episode',
                        'total_kills_per_episode',
                        'total_immediate_kills_per_episode',
                        'total_agent_collisions_per_episode',
                        'total_charges_per_episode',
                        'mean_episode_reward',
                        'std_episode_reward',
                        'episode_length',
                        'survival_rate',
                        'mean_final_energy',
                        'epsilon',
                        'global_step'
                    ]

                    available_cols = [col for col in metric_cols if col in df.columns]
                    df_clean = df[available_cols].dropna(subset=['total_non_home_charges_per_episode'])

                    if not df_clean.empty:
                        self.configs[config_name] = df_clean
                        print(f"✓ {config_name}: {len(df_clean)} 個回合")
                    else:
                        print(f"✗ {config_name}: 無有效資料")
                else:
                    print(f"✗ {config_name}: 缺少必要欄位")

            except Exception as e:
                print(f"✗ {config_name}: 載入錯誤 - {e}")

        print(f"\n成功載入 {len(self.configs)} 個配置")

    def create_config_analysis(self, config_name: str, output_dir: str = "analysis_output"):
        """為單一配置建立完整分析圖表"""
        if config_name not in self.configs:
            print(f"配置 '{config_name}' 不存在！")
            return

        df = self.configs[config_name]
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n正在分析: {config_name}")
        print(f"  回合數: {len(df)}")
        print(f"  非零充電回合: {len(df[df['total_non_home_charges_per_episode'] > 0])} ({len(df[df['total_non_home_charges_per_episode'] > 0])/len(df)*100:.1f}%)")

        # 建立大型圖表 (4x3 = 12 個子圖)
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Non-home charges 分布
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_distribution(df, ax1)

        # 2. Non-home charges vs Kills
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_scatter(df, 'total_kills_per_episode', 'Kills', ax2)

        # 3. Non-home charges vs Collisions
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_scatter(df, 'total_agent_collisions_per_episode', 'Agent Collisions', ax3)

        # 4. Non-home charges vs Reward
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_scatter(df, 'mean_episode_reward', 'Mean Reward', ax4)

        # 5. Non-home charges vs Episode Length
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_scatter(df, 'episode_length', 'Episode Length', ax5)

        # 6. Non-home charges vs Survival Rate
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_scatter(df, 'survival_rate', 'Survival Rate', ax6)

        # 7. Non-home charges vs Final Energy
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_scatter(df, 'mean_final_energy', 'Final Energy', ax7)

        # 8. Non-home charges vs Total Charges
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_scatter(df, 'total_charges_per_episode', 'Total Charges', ax8)

        # 9. Timeline: Non-home charges over episodes
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_timeline(df, ax9)

        # 10. 相關係數矩陣
        ax10 = fig.add_subplot(gs[3, 0])
        self._plot_correlation_matrix(df, ax10)

        # 11. 統計摘要
        ax11 = fig.add_subplot(gs[3, 1])
        self._plot_statistics_table(df, ax11)

        # 12. 分組比較 (有充電 vs 無充電)
        ax12 = fig.add_subplot(gs[3, 2])
        self._plot_group_comparison(df, ax12)

        # 設定標題
        fig.suptitle(f'配置分析: {config_name}\n總回合數: {len(df)} | 非零充電: {len(df[df["total_non_home_charges_per_episode"] > 0])} ({len(df[df["total_non_home_charges_per_episode"] > 0])/len(df)*100:.1f}%)',
                     fontsize=20, fontweight='bold', y=0.995)

        # 儲存
        output_file = output_path / f"{config_name}_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 已儲存: {output_file}")

    def _plot_distribution(self, df, ax):
        """Non-home charges 分布圖"""
        data = df['total_non_home_charges_per_episode']

        ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'平均: {data.mean():.1f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'中位數: {data.median():.1f}')

        ax.set_xlabel('Non-Home Charges per Episode', fontsize=11, fontweight='bold')
        ax.set_ylabel('頻率', fontsize=11, fontweight='bold')
        ax.set_title('Non-Home Charges 分布', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_scatter(self, df, metric_col, metric_name, ax):
        """散點圖 + 趨勢線"""
        if metric_col not in df.columns:
            ax.text(0.5, 0.5, f'無 {metric_name} 資料',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'Non-Home Charges vs {metric_name}', fontsize=12, fontweight='bold')
            return

        x = df['total_non_home_charges_per_episode']
        y = df[metric_col]

        # 散點圖
        ax.scatter(x, y, alpha=0.5, s=30, color='steelblue')

        # 計算相關係數
        if len(x) > 2:
            correlation = x.corr(y)

            # 趨勢線
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # 顯示相關係數
            ax.text(0.05, 0.95, f'相關係數: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top')

        ax.set_xlabel('Non-Home Charges', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_title(f'Non-Home Charges vs {metric_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_timeline(self, df, ax):
        """時間序列圖"""
        df_sorted = df.sort_values('episode')

        ax.plot(df_sorted['episode'], df_sorted['total_non_home_charges_per_episode'],
               alpha=0.6, linewidth=1, color='steelblue', label='原始資料')

        # 移動平均
        if len(df_sorted) >= 50:
            window = 50
            ma = df_sorted['total_non_home_charges_per_episode'].rolling(window=window, center=True).mean()
            ax.plot(df_sorted['episode'], ma, color='red', linewidth=2, label=f'{window}-回合移動平均')

        ax.set_xlabel('Episode', fontsize=10, fontweight='bold')
        ax.set_ylabel('Non-Home Charges', fontsize=10, fontweight='bold')
        ax.set_title('訓練過程中的 Non-Home Charges', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_correlation_matrix(self, df, ax):
        """相關係數矩陣"""
        # 選擇數值欄位
        numeric_cols = [
            'total_non_home_charges_per_episode',
            'total_kills_per_episode',
            'total_agent_collisions_per_episode',
            'mean_episode_reward',
            'episode_length',
            'survival_rate',
            'mean_final_energy'
        ]

        available_cols = [col for col in numeric_cols if col in df.columns]
        corr_data = df[available_cols].corr()

        # 簡化欄位名稱
        rename_dict = {
            'total_non_home_charges_per_episode': 'Non-Home\nCharges',
            'total_kills_per_episode': 'Kills',
            'total_agent_collisions_per_episode': 'Collisions',
            'mean_episode_reward': 'Reward',
            'episode_length': 'Length',
            'survival_rate': 'Survival',
            'mean_final_energy': 'Energy'
        }

        corr_data = corr_data.rename(index=rename_dict, columns=rename_dict)

        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, cbar_kws={'label': '相關係數'},
                   square=True, linewidths=1)

        ax.set_title('相關係數矩陣', fontsize=12, fontweight='bold')

    def _plot_statistics_table(self, df, ax):
        """統計摘要表格"""
        ax.axis('tight')
        ax.axis('off')

        stats_data = []
        metrics = {
            'total_non_home_charges_per_episode': 'Non-Home Charges',
            'total_kills_per_episode': 'Kills',
            'total_agent_collisions_per_episode': 'Collisions',
            'mean_episode_reward': 'Reward',
            'episode_length': 'Episode Length',
            'survival_rate': 'Survival Rate',
            'mean_final_energy': 'Final Energy'
        }

        for col, name in metrics.items():
            if col in df.columns:
                data = df[col]
                stats_data.append([
                    name,
                    f'{data.mean():.2f}',
                    f'{data.std():.2f}',
                    f'{data.min():.1f}',
                    f'{data.max():.1f}'
                ])

        table = ax.table(cellText=stats_data,
                        colLabels=['指標', '平均', '標準差', '最小', '最大'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # 設定標題樣式
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 交替行顏色
        for i in range(1, len(stats_data) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax.set_title('統計摘要', fontsize=12, fontweight='bold', pad=20)

    def _plot_group_comparison(self, df, ax):
        """有充電 vs 無充電的比較"""
        zero_charges = df[df['total_non_home_charges_per_episode'] == 0]
        nonzero_charges = df[df['total_non_home_charges_per_episode'] > 0]

        # 選擇要比較的指標
        metrics = {
            'total_kills_per_episode': 'Kills',
            'mean_episode_reward': 'Reward',
            'episode_length': 'Length',
            'survival_rate': 'Survival'
        }

        comparison_data = []
        labels = []

        for col, name in metrics.items():
            if col in df.columns:
                labels.append(name)
                if len(zero_charges) > 0:
                    comparison_data.append([
                        zero_charges[col].mean(),
                        nonzero_charges[col].mean() if len(nonzero_charges) > 0 else 0
                    ])

        if comparison_data:
            x = np.arange(len(labels))
            width = 0.35

            comparison_array = np.array(comparison_data)

            bars1 = ax.bar(x - width/2, comparison_array[:, 0], width,
                          label='無充電', color='lightcoral', alpha=0.8)
            bars2 = ax.bar(x + width/2, comparison_array[:, 1], width,
                          label='有充電', color='lightgreen', alpha=0.8)

            ax.set_ylabel('平均值', fontsize=10, fontweight='bold')
            ax.set_title('有充電 vs 無充電比較', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            # 在柱狀圖上顯示數值
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=8)

    def create_all_analyses(self, output_dir: str = "analysis_output"):
        """為所有配置建立分析圖表"""
        print(f"\n開始為 {len(self.configs)} 個配置建立分析圖表...\n")

        for config_name in self.configs.keys():
            self.create_config_analysis(config_name, output_dir)

        print(f"\n✓ 完成！所有分析圖表已儲存至 {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="為每個配置產生詳細的 non_home_charges 分析圖表"
    )

    parser.add_argument("--wandb-dir", type=str, default="wandb_data",
                       help="WandB CSV 檔案目錄")
    parser.add_argument("--output", type=str, default="analysis_output",
                       help="輸出目錄")
    parser.add_argument("--config", type=str, default=None,
                       help="只分析特定配置（不指定則分析全部）")

    args = parser.parse_args()

    # 建立分析器
    analyzer = PerConfigAnalyzer(args.wandb_dir)

    # 載入資料
    analyzer.load_data()

    if analyzer.configs:
        if args.config:
            # 只分析特定配置
            analyzer.create_config_analysis(args.config, args.output)
        else:
            # 分析所有配置
            analyzer.create_all_analyses(args.output)
    else:
        print("沒有可用的資料！")


if __name__ == "__main__":
    main()
