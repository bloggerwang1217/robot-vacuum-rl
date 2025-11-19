"""
Analyze relationship between Kills and Non-Home Charges

This script focuses on understanding how non-home charging station usage
relates to kill performance across different training configurations.
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


class KillsChargesAnalyzer:
    """Analyze kills vs non-home charges relationship"""

    def __init__(self, wandb_dir: str = "wandb_data"):
        self.wandb_dir = Path(wandb_dir)
        self.data = []
        self.df = None

    def load_wandb_data(self):
        """Load WandB CSV files with kills and non_home_charges data"""
        print(f"Loading data from {self.wandb_dir}...")

        csv_files = list(self.wandb_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            config_name = csv_file.stem
            try:
                df = pd.read_csv(csv_file)

                # Check if file has required columns
                required_cols = ['episode', 'total_kills_per_episode', 'total_non_home_charges_per_episode']
                if all(col in df.columns for col in required_cols):
                    df_clean = df[required_cols].dropna()
                    df_clean['config'] = config_name
                    self.data.append(df_clean)
                    print(f"  Loaded {len(df_clean)} episodes from {config_name}")
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")

        if self.data:
            self.df = pd.concat(self.data, ignore_index=True)
            print(f"\nTotal: {len(self.df)} episodes from {len(self.data)} configurations")
        else:
            print("No data loaded!")

    def create_visualizations(self, output_dir: str = "analysis_output"):
        """Create comprehensive visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if self.df is None or self.df.empty:
            print("No data to visualize!")
            return

        print(f"\nCreating visualizations in {output_dir}/...")

        # 1. Scatter plot: Kills vs Non-Home Charges
        self._plot_kills_vs_charges(output_path)

        # 2. Time series: Evolution over training
        self._plot_time_series(output_path)

        # 3. Distribution comparison
        self._plot_distributions(output_path)

        # 4. Correlation heatmap
        self._plot_correlation(output_path)

        # 5. Configuration comparison
        self._plot_config_comparison(output_path)

        # 6. Moving average analysis
        self._plot_moving_averages(output_path)

        print("\nAll visualizations saved!")

    def _plot_kills_vs_charges(self, output_path: Path):
        """Scatter plot of kills vs non-home charges"""
        plt.figure(figsize=(15, 10))

        configs = self.df['config'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

        for idx, config in enumerate(configs):
            config_df = self.df[self.df['config'] == config]
            plt.scatter(
                config_df['total_non_home_charges_per_episode'],
                config_df['total_kills_per_episode'],
                label=config,
                alpha=0.6,
                s=50,
                color=colors[idx]
            )

        plt.xlabel('Non-Home Charges per Episode', fontsize=14)
        plt.ylabel('Kills per Episode', fontsize=14)
        plt.title('Relationship between Non-Home Charges and Kills', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'kills_vs_charges_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created kills_vs_charges_scatter.png")

    def _plot_time_series(self, output_path: Path):
        """Time series showing evolution during training"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))

        configs = self.df['config'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

        for idx, config in enumerate(configs):
            config_df = self.df[self.df['config'] == config].sort_values('episode')

            # Plot kills over time
            axes[0].plot(
                config_df['episode'],
                config_df['total_kills_per_episode'],
                label=config,
                alpha=0.7,
                color=colors[idx]
            )

            # Plot non-home charges over time
            axes[1].plot(
                config_df['episode'],
                config_df['total_non_home_charges_per_episode'],
                label=config,
                alpha=0.7,
                color=colors[idx]
            )

        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel('Kills per Episode', fontsize=12)
        axes[0].set_title('Kills Evolution During Training', fontsize=14, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Non-Home Charges per Episode', fontsize=12)
        axes[1].set_title('Non-Home Charges Evolution During Training', fontsize=14, fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'time_series_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created time_series_evolution.png")

    def _plot_distributions(self, output_path: Path):
        """Distribution comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Kills distribution
        for config in self.df['config'].unique():
            config_df = self.df[self.df['config'] == config]
            axes[0].hist(
                config_df['total_kills_per_episode'],
                alpha=0.5,
                label=config,
                bins=30
            )

        axes[0].set_xlabel('Kills per Episode', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Kills', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Non-home charges distribution
        for config in self.df['config'].unique():
            config_df = self.df[self.df['config'] == config]
            axes[1].hist(
                config_df['total_non_home_charges_per_episode'],
                alpha=0.5,
                label=config,
                bins=30
            )

        axes[1].set_xlabel('Non-Home Charges per Episode', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Non-Home Charges', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created distributions.png")

    def _plot_correlation(self, output_path: Path):
        """Correlation analysis for each configuration"""
        configs = self.df['config'].unique()
        n_configs = len(configs)

        fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 5))
        if n_configs == 1:
            axes = [axes]

        for idx, config in enumerate(configs):
            config_df = self.df[self.df['config'] == config]

            # Calculate correlation
            corr = config_df[['total_kills_per_episode', 'total_non_home_charges_per_episode']].corr()

            sns.heatmap(
                corr,
                annot=True,
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                ax=axes[idx],
                cbar_kws={'label': 'Correlation'}
            )
            axes[idx].set_title(f'{config}\n(n={len(config_df)})', fontsize=10)

        plt.suptitle('Correlation: Kills vs Non-Home Charges', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created correlation_heatmaps.png")

    def _plot_config_comparison(self, output_path: Path):
        """Box plot comparison across configurations"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Kills comparison
        kills_data = []
        labels = []
        for config in self.df['config'].unique():
            config_df = self.df[self.df['config'] == config]
            kills_data.append(config_df['total_kills_per_episode'].values)
            labels.append(config)

        bp1 = axes[0].boxplot(kills_data, labels=labels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        axes[0].set_ylabel('Kills per Episode', fontsize=12)
        axes[0].set_title('Kills Comparison Across Configurations', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Non-home charges comparison
        charges_data = []
        for config in self.df['config'].unique():
            config_df = self.df[self.df['config'] == config]
            charges_data.append(config_df['total_non_home_charges_per_episode'].values)

        bp2 = axes[1].boxplot(charges_data, labels=labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
        axes[1].set_ylabel('Non-Home Charges per Episode', fontsize=12)
        axes[1].set_title('Non-Home Charges Comparison', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'config_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created config_comparison.png")

    def _plot_moving_averages(self, output_path: Path):
        """Moving average analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        window = 50

        configs = self.df['config'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

        for idx, config in enumerate(configs):
            config_df = self.df[self.df['config'] == config].sort_values('episode')

            if len(config_df) >= window:
                # Moving average for kills
                kills_ma = config_df['total_kills_per_episode'].rolling(window=window, center=True).mean()
                axes[0].plot(
                    config_df['episode'],
                    kills_ma,
                    label=config,
                    linewidth=2,
                    color=colors[idx]
                )

                # Moving average for non-home charges
                charges_ma = config_df['total_non_home_charges_per_episode'].rolling(window=window, center=True).mean()
                axes[1].plot(
                    config_df['episode'],
                    charges_ma,
                    label=config,
                    linewidth=2,
                    color=colors[idx]
                )

        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel(f'Kills (MA-{window})', fontsize=12)
        axes[0].set_title(f'Kills Moving Average (window={window})', fontsize=14, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel(f'Non-Home Charges (MA-{window})', fontsize=12)
        axes[1].set_title(f'Non-Home Charges Moving Average (window={window})', fontsize=14, fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'moving_averages.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Created moving_averages.png")

    def print_summary_stats(self):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS: Kills vs Non-Home Charges")
        print("="*80)

        for config in self.df['config'].unique():
            config_df = self.df[self.df['config'] == config]

            print(f"\n{config}:")
            print(f"  Episodes: {len(config_df)}")
            print(f"  Kills per Episode:")
            print(f"    Mean: {config_df['total_kills_per_episode'].mean():.2f}")
            print(f"    Std:  {config_df['total_kills_per_episode'].std():.2f}")
            print(f"    Max:  {config_df['total_kills_per_episode'].max():.0f}")
            print(f"  Non-Home Charges per Episode:")
            print(f"    Mean: {config_df['total_non_home_charges_per_episode'].mean():.2f}")
            print(f"    Std:  {config_df['total_non_home_charges_per_episode'].std():.2f}")
            print(f"    Max:  {config_df['total_non_home_charges_per_episode'].max():.0f}")

            # Correlation
            corr = config_df[['total_kills_per_episode', 'total_non_home_charges_per_episode']].corr().iloc[0, 1]
            print(f"  Correlation: {corr:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze relationship between kills and non-home charges"
    )

    parser.add_argument("--wandb-dir", type=str, default="wandb_data",
                       help="Directory containing WandB CSV exports")
    parser.add_argument("--output", type=str, default="analysis_output",
                       help="Output directory for visualizations")

    args = parser.parse_args()

    # Create analyzer
    analyzer = KillsChargesAnalyzer(args.wandb_dir)

    # Load data
    analyzer.load_wandb_data()

    if analyzer.df is not None and not analyzer.df.empty:
        # Print summary
        analyzer.print_summary_stats()

        # Create visualizations
        analyzer.create_visualizations(args.output)
    else:
        print("No data to analyze!")


if __name__ == "__main__":
    main()
