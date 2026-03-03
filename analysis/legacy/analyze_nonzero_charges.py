"""
Analyze episodes where non-home charges > 0

Focus on understanding what happens during episodes when robots use charging stations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class NonZeroChargesAnalyzer:
    """Analyze episodes with non-zero charging station usage"""

    def __init__(self, wandb_dir: str = "wandb_data"):
        self.wandb_dir = Path(wandb_dir)
        self.all_data = None
        self.nonzero_data = None

    def load_data(self):
        """Load WandB CSV files"""
        print(f"Loading data from {self.wandb_dir}...")

        data_frames = []
        csv_files = list(self.wandb_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files\n")

        for csv_file in csv_files:
            config_name = csv_file.stem
            try:
                df = pd.read_csv(csv_file)

                # Select relevant columns (all available metrics)
                keep_cols = [
                    'episode',
                    'total_kills_per_episode',
                    'total_non_home_charges_per_episode',
                    'total_charges_per_episode',
                    'total_agent_collisions_per_episode',
                    'mean_episode_reward',
                    'episode_length',
                    'survival_rate',
                    'mean_final_energy',
                    'total_immediate_kills_per_episode'
                ]

                available_cols = [col for col in keep_cols if col in df.columns]
                df_clean = df[available_cols].dropna(subset=['total_non_home_charges_per_episode'])
                df_clean['config'] = config_name

                data_frames.append(df_clean)

                # Print stats for this config
                total = len(df_clean)
                nonzero = len(df_clean[df_clean['total_non_home_charges_per_episode'] > 0])
                pct = (nonzero / total * 100) if total > 0 else 0

                print(f"{config_name}:")
                print(f"  Total episodes: {total}")
                print(f"  Non-zero charges: {nonzero} ({pct:.1f}%)")
                if nonzero > 0:
                    print(f"  Avg charges (when >0): {df_clean[df_clean['total_non_home_charges_per_episode'] > 0]['total_non_home_charges_per_episode'].mean():.1f}")
                print()

            except Exception as e:
                print(f"  Error loading {csv_file}: {e}\n")

        if data_frames:
            self.all_data = pd.concat(data_frames, ignore_index=True)
            self.nonzero_data = self.all_data[self.all_data['total_non_home_charges_per_episode'] > 0].copy()

            print(f"="*80)
            print(f"OVERALL SUMMARY")
            print(f"="*80)
            print(f"Total episodes: {len(self.all_data)}")
            print(f"Episodes with non-zero charges: {len(self.nonzero_data)} ({len(self.nonzero_data)/len(self.all_data)*100:.1f}%)")
            print(f"Configurations: {self.all_data['config'].nunique()}")
            print()

    def print_detailed_comparison(self):
        """Print detailed statistics comparing zero vs non-zero episodes"""
        if self.all_data is None:
            print("No data loaded!")
            return

        zero_data = self.all_data[self.all_data['total_non_home_charges_per_episode'] == 0]

        print("\n" + "="*80)
        print("DETAILED COMPARISON: Episodes with vs without Non-Home Charges")
        print("="*80)

        metrics = {
            'total_kills_per_episode': 'Kills',
            'total_agent_collisions_per_episode': 'Collisions',
            'mean_episode_reward': 'Mean Reward',
            'episode_length': 'Episode Length',
            'survival_rate': 'Survival Rate',
            'mean_final_energy': 'Final Energy',
            'total_immediate_kills_per_episode': 'Immediate Kills',
            'total_charges_per_episode': 'Total Charges'
        }

        for col, label in metrics.items():
            if col in self.nonzero_data.columns:
                print(f"\n{label}:")
                print(f"  Episodes WITH non-home charges (n={len(self.nonzero_data)}):")
                print(f"    Mean: {self.nonzero_data[col].mean():.3f}")
                print(f"    Std:  {self.nonzero_data[col].std():.3f}")
                print(f"    Min:  {self.nonzero_data[col].min():.3f}")
                print(f"    Max:  {self.nonzero_data[col].max():.3f}")

                print(f"  Episodes WITHOUT non-home charges (n={len(zero_data)}):")
                print(f"    Mean: {zero_data[col].mean():.3f}")
                print(f"    Std:  {zero_data[col].std():.3f}")
                print(f"    Min:  {zero_data[col].min():.3f}")
                print(f"    Max:  {zero_data[col].max():.3f}")

                # Difference
                diff = self.nonzero_data[col].mean() - zero_data[col].mean()
                pct_diff = (diff / zero_data[col].mean() * 100) if zero_data[col].mean() != 0 else 0
                print(f"  DIFFERENCE: {diff:+.3f} ({pct_diff:+.1f}%)")

    def print_top_charging_episodes(self, n=20):
        """Print details of episodes with highest charging usage"""
        if self.nonzero_data is None or len(self.nonzero_data) == 0:
            print("No episodes with non-zero charges!")
            return

        print("\n" + "="*80)
        print(f"TOP {n} EPISODES WITH HIGHEST NON-HOME CHARGING")
        print("="*80)

        top_episodes = self.nonzero_data.nlargest(n, 'total_non_home_charges_per_episode')

        display_cols = [
            'config',
            'episode',
            'total_non_home_charges_per_episode',
            'total_kills_per_episode',
            'total_agent_collisions_per_episode',
            'mean_episode_reward',
            'episode_length',
            'survival_rate'
        ]

        available_display_cols = [col for col in display_cols if col in top_episodes.columns]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)

        print(top_episodes[available_display_cols].to_string(index=False))

    def create_visualizations(self, output_dir: str = "analysis_output"):
        """Create visualizations comparing zero vs non-zero episodes"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if self.all_data is None:
            print("No data to visualize!")
            return

        print(f"\nCreating visualizations in {output_dir}/...")

        # 1. Comparison of all metrics
        self._plot_metric_comparison(output_path)

        # 2. Distribution of charges (non-zero only)
        self._plot_charge_distribution(output_path)

        # 3. Timeline of charging events
        self._plot_charging_timeline(output_path)

        # 4. Scatter matrix for non-zero episodes
        self._plot_scatter_matrix(output_path)

        print("\nAll visualizations saved!")

    def _plot_metric_comparison(self, output_path: Path):
        """Box plot comparison of metrics"""
        zero_data = self.all_data[self.all_data['total_non_home_charges_per_episode'] == 0]

        metrics = [
            'total_kills_per_episode',
            'mean_episode_reward',
            'episode_length',
            'survival_rate',
            'total_agent_collisions_per_episode',
            'mean_final_energy'
        ]

        available_metrics = [m for m in metrics if m in self.all_data.columns]
        n_metrics = len(available_metrics)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(available_metrics):
            data_to_plot = [
                zero_data[metric].dropna(),
                self.nonzero_data[metric].dropna()
            ]

            bp = axes[idx].boxplot(
                data_to_plot,
                labels=['Charges = 0', 'Charges > 0'],
                patch_artist=True
            )

            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')

            axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Metric Comparison: Episodes with vs without Non-Home Charges',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'nonzero_metric_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created nonzero_metric_comparison.png")

    def _plot_charge_distribution(self, output_path: Path):
        """Distribution of non-home charges (non-zero only)"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        axes[0].hist(
            self.nonzero_data['total_non_home_charges_per_episode'],
            bins=50,
            color='coral',
            alpha=0.7,
            edgecolor='black'
        )
        axes[0].set_xlabel('Non-Home Charges per Episode', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Non-Home Charges (>0 only)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Box plot by configuration
        configs = self.nonzero_data['config'].unique()
        data_by_config = [
            self.nonzero_data[self.nonzero_data['config'] == config]['total_non_home_charges_per_episode'].values
            for config in configs
        ]

        bp = axes[1].boxplot(data_by_config, labels=configs, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')

        axes[1].set_ylabel('Non-Home Charges per Episode', fontsize=12)
        axes[1].set_title('Non-Home Charges by Configuration', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'nonzero_charge_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created nonzero_charge_distribution.png")

    def _plot_charging_timeline(self, output_path: Path):
        """Timeline showing when charging events occur"""
        fig, ax = plt.subplots(figsize=(15, 8))

        configs = self.nonzero_data['config'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

        for idx, config in enumerate(configs):
            config_data = self.nonzero_data[self.nonzero_data['config'] == config]

            ax.scatter(
                config_data['episode'],
                config_data['total_non_home_charges_per_episode'],
                label=config,
                alpha=0.6,
                s=100,
                color=colors[idx]
            )

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Non-Home Charges', fontsize=12)
        ax.set_title('Timeline of Charging Station Usage', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'nonzero_charging_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created nonzero_charging_timeline.png")

    def _plot_scatter_matrix(self, output_path: Path):
        """Scatter matrix for non-zero episodes"""
        metrics = [
            'total_non_home_charges_per_episode',
            'total_kills_per_episode',
            'mean_episode_reward',
            'episode_length'
        ]

        available_metrics = [m for m in metrics if m in self.nonzero_data.columns]

        if len(available_metrics) < 2:
            print("  ⚠ Not enough metrics for scatter matrix")
            return

        plot_data = self.nonzero_data[available_metrics].dropna()

        pd.plotting.scatter_matrix(
            plot_data,
            figsize=(15, 15),
            alpha=0.6,
            diagonal='hist',
            color='coral'
        )

        plt.suptitle('Scatter Matrix: Non-Zero Charging Episodes', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path / 'nonzero_scatter_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created nonzero_scatter_matrix.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze episodes with non-zero charging station usage"
    )

    parser.add_argument("--wandb-dir", type=str, default="wandb_data",
                       help="Directory containing WandB CSV exports")
    parser.add_argument("--output", type=str, default="analysis_output",
                       help="Output directory for visualizations")
    parser.add_argument("--top-n", type=int, default=20,
                       help="Number of top charging episodes to display")

    args = parser.parse_args()

    # Create analyzer
    analyzer = NonZeroChargesAnalyzer(args.wandb_dir)

    # Load data
    analyzer.load_data()

    if analyzer.all_data is not None and not analyzer.all_data.empty:
        # Print detailed comparison
        analyzer.print_detailed_comparison()

        # Print top charging episodes
        analyzer.print_top_charging_episodes(args.top_n)

        # Create visualizations
        analyzer.create_visualizations(args.output)
    else:
        print("No data to analyze!")


if __name__ == "__main__":
    main()
