"""
Training Log Analysis and Visualization Script

This script parses training logs and creates comprehensive visualizations
to understand the training dynamics of the multi-robot DQN agents.

Usage:
    python analyze_training.py --log-file <path_to_training.log>
    python analyze_training.py --log-dir <directory_with_multiple_logs>
"""

import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class TrainingLogParser:
    """Parse training logs and extract metrics"""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics = {
            'episode': [],
            'steps': [],
            'survival': [],
            'mean_reward': [],
            'collisions': [],
            'kills': []
        }
        self.key_models = []  # Store episodes where key models were saved

    def parse(self) -> Dict[str, List]:
        """Parse the log file and extract metrics"""

        # Pattern to match episode lines
        # Example: [Episode 0] Steps: 41 | Survival: 2/4 | Mean Reward: 0.33 | Collisions: 23 | Kills: 2
        episode_pattern = r'\[Episode (\d+)\] Steps: (\d+) \| Survival: (\d+)/\d+ \| Mean Reward: ([-\d.]+) \| Collisions: (\d+) \| Kills: (\d+)'

        # Pattern to match key model saves
        # Example: [Key Model Saved] Episode 0: New record of 2 kills!
        key_model_pattern = r'\[Key Model Saved\] Episode (\d+): New record of (\d+) kills!'

        with open(self.log_file, 'r') as f:
            for line in f:
                # Try to match episode data
                episode_match = re.search(episode_pattern, line)
                if episode_match:
                    episode, steps, survival, mean_reward, collisions, kills = episode_match.groups()
                    self.metrics['episode'].append(int(episode))
                    self.metrics['steps'].append(int(steps))
                    self.metrics['survival'].append(int(survival))
                    self.metrics['mean_reward'].append(float(mean_reward))
                    self.metrics['collisions'].append(int(collisions))
                    self.metrics['kills'].append(int(kills))

                # Try to match key model saves
                key_model_match = re.search(key_model_pattern, line)
                if key_model_match:
                    episode, kills = key_model_match.groups()
                    self.key_models.append({
                        'episode': int(episode),
                        'kills': int(kills)
                    })

        return self.metrics

    def get_summary_stats(self) -> Dict:
        """Compute summary statistics"""
        if not self.metrics['episode']:
            return {}

        return {
            'total_episodes': len(self.metrics['episode']),
            'avg_steps': np.mean(self.metrics['steps']),
            'avg_survival': np.mean(self.metrics['survival']),
            'avg_reward': np.mean(self.metrics['mean_reward']),
            'avg_collisions': np.mean(self.metrics['collisions']),
            'avg_kills': np.mean(self.metrics['kills']),
            'max_reward': np.max(self.metrics['mean_reward']),
            'max_kills': np.max(self.metrics['kills']),
            'max_steps': np.max(self.metrics['steps']),
            'final_reward': np.mean(self.metrics['mean_reward'][-10:]) if len(self.metrics['mean_reward']) >= 10 else np.mean(self.metrics['mean_reward']),
            'reward_improvement': self.metrics['mean_reward'][-1] - self.metrics['mean_reward'][0] if len(self.metrics['mean_reward']) > 1 else 0
        }


class TrainingVisualizer:
    """Create comprehensive visualizations of training progress"""

    def __init__(self, metrics: Dict[str, List], key_models: List[Dict], run_name: str = "Training Run"):
        self.metrics = metrics
        self.key_models = key_models
        self.run_name = run_name

    def plot_all(self, save_path: str = None, window_size: int = 50):
        """Create all visualizations in a single figure"""

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Reward over time with smoothing
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_metric_with_smoothing(ax1, 'mean_reward', 'Mean Episode Reward',
                                        color='#2E86AB', window_size=window_size)

        # 2. Survival rate over time
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_metric_simple(ax2, 'survival', 'Survival Count',
                                color='#06A77D', ylabel='Robots Alive')

        # 3. Episode length over time
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_metric_with_smoothing(ax3, 'steps', 'Episode Length',
                                        color='#A23B72', window_size=window_size)

        # 4. Collisions over time
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_metric_with_smoothing(ax4, 'collisions', 'Agent Collisions',
                                        color='#F18F01', window_size=window_size)

        # 5. Kills over time
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_metric_with_smoothing(ax5, 'kills', 'Kills per Episode',
                                        color='#C73E1D', window_size=window_size)

        # 6. Reward distribution (histogram)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_distribution(ax6, 'mean_reward', 'Reward Distribution',
                               color='#2E86AB', bins=30)

        # 7. Correlation heatmap
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_correlation_heatmap(ax7)

        # 8. Learning phases analysis
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_learning_phases(ax8, window_size=window_size)

        # Add title and metadata
        fig.suptitle(f'{self.run_name} - Training Analysis',
                    fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def _plot_metric_with_smoothing(self, ax, metric_key: str, title: str,
                                   color: str, window_size: int = 50):
        """Plot a metric with raw data and smoothed curve"""
        episodes = self.metrics['episode']
        values = self.metrics[metric_key]

        # Raw data with transparency
        ax.plot(episodes, values, alpha=0.3, color=color, linewidth=1, label='Raw')

        # Smoothed data
        if len(values) >= window_size:
            smoothed = self._moving_average(values, window_size)
            ax.plot(episodes[window_size-1:], smoothed, color=color,
                   linewidth=2.5, label=f'Smoothed (window={window_size})')

        # Mark key model saves
        for km in self.key_models:
            if km['episode'] < len(episodes):
                ax.axvline(x=km['episode'], color='red', linestyle='--',
                          alpha=0.5, linewidth=1)
                ax.scatter(km['episode'], values[km['episode']],
                          color='red', s=100, zorder=5, marker='*')

        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_metric_simple(self, ax, metric_key: str, title: str,
                          color: str, ylabel: str = None):
        """Plot a simple metric without smoothing"""
        episodes = self.metrics['episode']
        values = self.metrics[metric_key]

        ax.scatter(episodes, values, alpha=0.5, color=color, s=10)
        ax.plot(episodes, values, alpha=0.3, color=color, linewidth=1)

        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel or title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _plot_distribution(self, ax, metric_key: str, title: str,
                          color: str, bins: int = 30):
        """Plot distribution histogram"""
        values = self.metrics[metric_key]

        ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(values):.2f}')
        ax.axvline(np.median(values), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(values):.2f}')

        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_correlation_heatmap(self, ax):
        """Plot correlation heatmap between metrics"""
        # Select numeric metrics
        data_dict = {
            'Steps': self.metrics['steps'],
            'Survival': self.metrics['survival'],
            'Reward': self.metrics['mean_reward'],
            'Collisions': self.metrics['collisions'],
            'Kills': self.metrics['kills']
        }

        # Compute correlation matrix
        import pandas as pd
        df = pd.DataFrame(data_dict)
        corr = df.corr()

        # Plot heatmap
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)

        # Add correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)

        ax.set_title('Metric Correlations')
        plt.colorbar(im, ax=ax)

    def _plot_learning_phases(self, ax, window_size: int = 50):
        """Analyze and plot different learning phases"""
        rewards = np.array(self.metrics['mean_reward'])

        if len(rewards) < window_size:
            ax.text(0.5, 0.5, 'Insufficient data for phase analysis',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Compute rolling statistics
        smoothed = self._moving_average(rewards, window_size)
        episodes = np.array(self.metrics['episode'][window_size-1:])

        # Compute gradient (learning rate)
        gradient = np.gradient(smoothed)

        # Plot reward with gradient coloring
        scatter = ax.scatter(episodes, smoothed, c=gradient, cmap='RdYlGn',
                           s=30, alpha=0.7)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Smoothed Reward')
        ax.set_title('Learning Rate Analysis\n(Color = Improvement Rate)')
        plt.colorbar(scatter, ax=ax, label='Gradient')
        ax.grid(True, alpha=0.3)

    def _moving_average(self, data: List[float], window_size: int) -> np.ndarray:
        """Compute moving average"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def plot_comparison(self, other_metrics: Dict[str, List],
                       other_name: str, save_path: str = None):
        """Compare two training runs"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Comparison: {self.run_name} vs {other_name}',
                    fontsize=16, fontweight='bold')

        metrics_to_compare = ['mean_reward', 'steps', 'survival',
                             'collisions', 'kills']
        titles = ['Mean Reward', 'Episode Length', 'Survival Count',
                 'Collisions', 'Kills']

        for idx, (metric, title) in enumerate(zip(metrics_to_compare, titles)):
            ax = axes[idx // 3, idx % 3]

            # Plot first run
            ax.plot(self.metrics['episode'], self.metrics[metric],
                   alpha=0.6, label=self.run_name, linewidth=2)

            # Plot second run
            ax.plot(other_metrics['episode'], other_metrics[metric],
                   alpha=0.6, label=other_name, linewidth=2)

            ax.set_xlabel('Episode')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide the last subplot if odd number of metrics
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")

        plt.show()


def print_summary_report(stats: Dict, run_name: str):
    """Print a formatted summary report"""
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY REPORT: {run_name}")
    print("="*70)
    print(f"Total Episodes:        {stats['total_episodes']}")
    print(f"Average Steps:         {stats['avg_steps']:.2f}")
    print(f"Average Survival:      {stats['avg_survival']:.2f}/4")
    print(f"Average Reward:        {stats['avg_reward']:.2f}")
    print(f"Average Collisions:    {stats['avg_collisions']:.2f}")
    print(f"Average Kills:         {stats['avg_kills']:.2f}")
    print("-"*70)
    print(f"Max Reward:            {stats['max_reward']:.2f}")
    print(f"Max Kills:             {stats['max_kills']}")
    print(f"Max Steps:             {stats['max_steps']}")
    print("-"*70)
    print(f"Final 10-ep Avg Reward: {stats['final_reward']:.2f}")
    print(f"Reward Improvement:     {stats['reward_improvement']:.2f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize training logs for multi-robot DQN"
    )

    parser.add_argument("--log-file", type=str,
                       help="Path to a single training.log file")
    parser.add_argument("--log-dir", type=str,
                       help="Directory containing multiple training.log files")
    parser.add_argument("--output-dir", type=str, default="./analysis_output",
                       help="Directory to save visualization outputs")
    parser.add_argument("--window-size", type=int, default=50,
                       help="Window size for smoothing plots")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple runs (requires --log-dir)")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    if args.log_file:
        # Analyze single log file
        print(f"Analyzing: {args.log_file}")

        parser = TrainingLogParser(args.log_file)
        metrics = parser.parse()
        stats = parser.get_summary_stats()

        run_name = Path(args.log_file).parent.name
        print_summary_report(stats, run_name)

        # Create visualizations
        visualizer = TrainingVisualizer(metrics, parser.key_models, run_name)
        save_path = output_path / f"{run_name}_analysis.png"
        visualizer.plot_all(save_path=str(save_path), window_size=args.window_size)

    elif args.log_dir:
        # Find all training.log files
        log_files = list(Path(args.log_dir).rglob("training.log"))

        if not log_files:
            print(f"No training.log files found in {args.log_dir}")
            return

        print(f"Found {len(log_files)} training logs")

        all_parsers = []
        all_metrics = []
        all_names = []

        for log_file in log_files:
            print(f"\nAnalyzing: {log_file}")

            parser = TrainingLogParser(str(log_file))
            metrics = parser.parse()
            stats = parser.get_summary_stats()

            if not metrics['episode']:
                print(f"Skipping {log_file} (no data found)")
                continue

            run_name = log_file.parent.name
            print_summary_report(stats, run_name)

            all_parsers.append(parser)
            all_metrics.append(metrics)
            all_names.append(run_name)

            # Create individual visualizations
            visualizer = TrainingVisualizer(metrics, parser.key_models, run_name)
            save_path = output_path / f"{run_name}_analysis.png"
            visualizer.plot_all(save_path=str(save_path), window_size=args.window_size)

        # Optional: Create comparison plots
        if args.compare and len(all_parsers) >= 2:
            print("\nCreating comparison plots...")
            base_visualizer = TrainingVisualizer(all_metrics[0],
                                                all_parsers[0].key_models,
                                                all_names[0])

            for i in range(1, len(all_metrics)):
                comp_path = output_path / f"comparison_{all_names[0]}_vs_{all_names[i]}.png"
                base_visualizer.plot_comparison(all_metrics[i], all_names[i],
                                               save_path=str(comp_path))

    else:
        print("Please provide either --log-file or --log-dir")
        parser.print_help()


if __name__ == "__main__":
    main()
