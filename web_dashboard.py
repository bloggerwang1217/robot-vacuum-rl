"""
Interactive HTML Dashboard for Training Analysis

This creates a standalone HTML file with interactive Plotly visualizations
to compare different training runs with various hyperparameters.

Usage:
    python web_dashboard.py --models-dir models/ --output dashboard.html

Then open dashboard.html in your browser
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo


class TrainingDataLoader:
    """Load and parse all training logs from a directory"""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.runs_data = []

    def load_all_runs(self) -> pd.DataFrame:
        """Find and parse all training.log files"""
        log_files = list(self.models_dir.rglob("training.log"))

        print(f"Found {len(log_files)} training logs")

        for log_file in log_files:
            run_data = self._parse_log(log_file)
            if run_data is not None:
                self.runs_data.append(run_data)

        # Combine all runs into a single DataFrame
        if self.runs_data:
            return pd.concat(self.runs_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def _parse_log(self, log_file: Path) -> pd.DataFrame:
        """Parse a single training log file"""

        # Extract hyperparameters from directory name
        config_name = log_file.parent.name
        params = self._parse_config_name(config_name)

        # Parse episode data
        episode_data = []
        episode_pattern = r'\[Episode (\d+)\] Steps: (\d+) \| Survival: (\d+)/\d+ \| Mean Reward: ([-\d.]+) \| Collisions: (\d+) \| Kills: (\d+)'

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    match = re.search(episode_pattern, line)
                    if match:
                        episode, steps, survival, reward, collisions, kills = match.groups()
                        episode_data.append({
                            'episode': int(episode),
                            'steps': int(steps),
                            'survival': int(survival),
                            'reward': float(reward),
                            'collisions': int(collisions),
                            'kills': int(kills),
                            'config': config_name,
                            **params
                        })
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            return None

        if not episode_data:
            return None

        df = pd.DataFrame(episode_data)
        print(f"  Loaded {len(df)} episodes from {config_name}")
        return df

    def _parse_config_name(self, config_name: str) -> Dict:
        """Extract hyperparameters from configuration name"""
        params = {
            'collision_penalty': None,
            'initial_energy': None,
            'epsilon': None,
            'gamma': None,
            'epsilon_decay': False
        }

        # Parse collision penalty
        collision_match = re.search(r'collision-(\d+)', config_name)
        if collision_match:
            params['collision_penalty'] = int(collision_match.group(1))

        # Parse initial energy
        energy_match = re.search(r'energy-(\d+)', config_name)
        if energy_match:
            params['initial_energy'] = int(energy_match.group(1))

        # Parse epsilon
        epsilon_match = re.search(r'epsilon-([\d.]+)', config_name)
        if epsilon_match:
            params['epsilon'] = float(epsilon_match.group(1))

        # Parse gamma
        gamma_match = re.search(r'gamma-([\d.]+)', config_name)
        if gamma_match:
            params['gamma'] = float(gamma_match.group(1))

        # Check for epsilon decay
        if 'epsilon-decay' in config_name:
            params['epsilon_decay'] = True
            params['epsilon'] = 'decay'

        return params


class HTMLDashboardGenerator:
    """Generate standalone HTML dashboard with interactive plots"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.window_size = 50

    def generate(self, output_file: str):
        """Generate complete HTML dashboard"""

        print("Generating visualizations...")

        # Create all plots
        plots = []

        # Summary statistics
        summary_html = self._create_summary_table()

        # Main metric plots
        plots.append(self._create_metric_plot('reward', 'Mean Episode Reward'))
        plots.append(self._create_metric_plot('survival', 'Survival Count'))
        plots.append(self._create_metric_plot('steps', 'Episode Length'))
        plots.append(self._create_metric_plot('collisions', 'Agent Collisions'))
        plots.append(self._create_metric_plot('kills', 'Kills per Episode'))

        # Scatter matrix - moved before hyperparameter impact
        plots.append(self._create_scatter_matrix())

        # Hyperparameter impact - moved to the end
        plots.append(self._create_param_impact())

        # Combine all plots into HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Robot Vacuum RL Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        h2 {{
            color: #555;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 10px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #2E86AB;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .info-box {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #2E86AB;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Robot Vacuum RL Training Dashboard</h1>

        <div class="info-box">
            <strong>Dataset Summary:</strong>
            {len(self.df)} total episodes from {self.df['config'].nunique()} different configurations
        </div>

        <h2>Summary Statistics (Last 100 Episodes)</h2>
        {summary_html}

        <h2>Training Dynamics</h2>
"""

        # Add each plot
        for i, (plot_div, plot_script) in enumerate(plots):
            html_content += f"""
        <div class="plot-container">
            {plot_div}
        </div>
"""

        html_content += """
    </div>
"""

        # Add all scripts at the end
        for plot_div, plot_script in plots:
            html_content += plot_script

        html_content += """
</body>
</html>
"""

        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"\nDashboard saved to: {output_file}")
        print(f"Open {output_file} in your browser to view the dashboard")

    def _create_metric_plot(self, metric: str, title: str) -> tuple:
        """Create an interactive line plot for a metric"""

        fig = go.Figure()

        for config in sorted(self.df['config'].unique()):
            config_df = self.df[self.df['config'] == config].sort_values('episode')

            # Raw data (semi-transparent) - convert to list to avoid binary encoding
            fig.add_trace(go.Scatter(
                x=config_df['episode'].tolist(),
                y=config_df[metric].tolist(),
                mode='lines',
                name=f'{config} (raw)',
                line=dict(width=1),
                opacity=0.3,
                legendgroup=config,
                showlegend=False,
                hovertemplate=f'<b>{config}</b><br>Episode: %{{x}}<br>{title}: %{{y:.2f}}<extra></extra>'
            ))

            # Smoothed data - convert to list to avoid binary encoding
            if len(config_df) >= self.window_size:
                smoothed = config_df[metric].rolling(window=self.window_size, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=config_df['episode'].tolist(),
                    y=smoothed.tolist(),
                    mode='lines',
                    name=config,
                    line=dict(width=2.5),
                    legendgroup=config,
                    hovertemplate=f'<b>{config}</b><br>Episode: %{{x}}<br>{title} (smoothed): %{{y:.2f}}<extra></extra>'
                ))

        fig.update_layout(
            title=title,
            xaxis_title='Episode',
            yaxis_title=title,
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=-0.3,  # Below the chart
                xanchor="center",
                x=0.5,
                font=dict(size=10)  # Smaller font
            )
        )

        # Convert to HTML div - disable binary encoding by converting to JSON first
        import plotly.io as pio
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False,
                               config={'displayModeBar': False},
                               validate=False, auto_play=False)

        return plot_div, ""

    def _create_scatter_matrix(self) -> tuple:
        """Create scatter matrix of final metrics"""

        # Compute final metrics for each config
        final_metrics = []
        for config in self.df['config'].unique():
            config_df = self.df[self.df['config'] == config]

            # Take last 100 episodes
            last_n = config_df.nlargest(min(100, len(config_df)), 'episode')

            final_metrics.append({
                'config': config,
                'Avg Reward': float(last_n['reward'].mean()),
                'Avg Survival': float(last_n['survival'].mean()),
                'Avg Steps': float(last_n['steps'].mean()),
                'Avg Collisions': float(last_n['collisions'].mean()),
                'Avg Kills': float(last_n['kills'].mean())
            })

        final_df = pd.DataFrame(final_metrics)

        # Create scatter matrix using Splom (avoids binary encoding issues)
        dimensions = ['Avg Reward', 'Avg Survival', 'Avg Steps', 'Avg Collisions', 'Avg Kills']

        fig = go.Figure(data=go.Splom(
            dimensions=[dict(label=dim, values=final_df[dim].tolist()) for dim in dimensions],
            text=final_df['config'].tolist(),
            marker=dict(
                size=7,
                showscale=False,
                line_color='white',
                line_width=0.5
            ),
            diagonal_visible=False,
            showupperhalf=False
        ))

        fig.update_layout(
            title='Final Performance Metrics Correlation (Last 100 Episodes)',
            height=700,
            hovermode='closest'
        )

        import plotly.io as pio
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False,
                               config={'displayModeBar': False},
                               validate=False, auto_play=False)

        return plot_div, ""

    def _create_param_impact(self) -> tuple:
        """Analyze hyperparameter impact on final performance"""

        # Compute final metrics for each config
        final_metrics = []
        for config in self.df['config'].unique():
            config_df = self.df[self.df['config'] == config]
            last_n = config_df.nlargest(min(100, len(config_df)), 'episode')

            row = {
                'config': config,
                'avg_reward': last_n['reward'].mean(),
                'collision_penalty': config_df['collision_penalty'].iloc[0],
                'initial_energy': config_df['initial_energy'].iloc[0],
                'epsilon': str(config_df['epsilon'].iloc[0]),
                'gamma': config_df['gamma'].iloc[0]
            }
            final_metrics.append(row)

        final_df = pd.DataFrame(final_metrics)

        # Create subplots for each hyperparameter
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Collision Penalty vs Reward',
                          'Initial Energy vs Reward',
                          'Epsilon vs Reward',
                          'Gamma vs Reward')
        )

        # Collision penalty - convert to list to avoid binary encoding
        if final_df['collision_penalty'].notna().any() and final_df['collision_penalty'].nunique() > 1:
            for config in final_df['config'].unique():
                row_data = final_df[final_df['config'] == config]
                fig.add_trace(
                    go.Scatter(
                        x=row_data['collision_penalty'].tolist(),
                        y=row_data['avg_reward'].tolist(),
                        mode='markers',
                        marker=dict(size=12),
                        name=config,
                        legendgroup=config,
                        showlegend=True,  # Show in legend from first subplot
                        hovertemplate=f'<b>{config}</b><br>Collision Penalty: %{{x}}<br>Avg Reward: %{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )

        # Initial energy - convert to list to avoid binary encoding
        if final_df['initial_energy'].notna().any() and final_df['initial_energy'].nunique() > 1:
            for config in final_df['config'].unique():
                row_data = final_df[final_df['config'] == config]
                fig.add_trace(
                    go.Scatter(
                        x=row_data['initial_energy'].tolist(),
                        y=row_data['avg_reward'].tolist(),
                        mode='markers',
                        marker=dict(size=12),
                        name=config,
                        legendgroup=config,
                        showlegend=False,  # Already shown in first subplot
                        hovertemplate=f'<b>{config}</b><br>Initial Energy: %{{x}}<br>Avg Reward: %{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=2
                )

        # Epsilon (categorical) - convert to list to avoid binary encoding
        if final_df['epsilon'].notna().any():
            for epsilon_val in final_df['epsilon'].unique():
                subset = final_df[final_df['epsilon'] == epsilon_val]
                fig.add_trace(
                    go.Box(
                        y=subset['avg_reward'].tolist(),
                        name=str(epsilon_val),
                        showlegend=False
                    ),
                    row=2, col=1
                )

        # Gamma - convert to list to avoid binary encoding
        if final_df['gamma'].notna().any() and final_df['gamma'].nunique() > 1:
            for config in final_df['config'].unique():
                row_data = final_df[final_df['config'] == config]
                fig.add_trace(
                    go.Scatter(
                        x=row_data['gamma'].tolist(),
                        y=row_data['avg_reward'].tolist(),
                        mode='markers',
                        marker=dict(size=12),
                        name=config,
                        legendgroup=config,
                        showlegend=False,  # Already shown in first subplot
                        hovertemplate=f'<b>{config}</b><br>Gamma: %{{x}}<br>Avg Reward: %{{y:.2f}}<extra></extra>'
                    ),
                    row=2, col=2
                )

        fig.update_xaxes(title_text="Collision Penalty", row=1, col=1)
        fig.update_xaxes(title_text="Initial Energy", row=1, col=2)
        fig.update_xaxes(title_text="Epsilon", row=2, col=1)
        fig.update_xaxes(title_text="Gamma", row=2, col=2)

        fig.update_yaxes(title_text="Avg Reward", row=1, col=1)
        fig.update_yaxes(title_text="Avg Reward", row=1, col=2)
        fig.update_yaxes(title_text="Avg Reward", row=2, col=1)
        fig.update_yaxes(title_text="Avg Reward", row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="Hyperparameter Impact Analysis",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            )
        )

        import plotly.io as pio
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=False,
                               config={'displayModeBar': False},
                               validate=False, auto_play=False)

        return plot_div, ""

    def _create_summary_table(self) -> str:
        """Create summary statistics HTML table"""

        summary_data = []
        for config in sorted(self.df['config'].unique()):
            config_df = self.df[self.df['config'] == config]
            last_n = config_df.nlargest(min(100, len(config_df)), 'episode')

            summary_data.append({
                'Configuration': config,
                'Episodes': len(config_df),
                'Avg Reward': f"{last_n['reward'].mean():.2f}",
                'Avg Survival': f"{last_n['survival'].mean():.2f}",
                'Avg Steps': f"{last_n['steps'].mean():.2f}",
                'Avg Collisions': f"{last_n['collisions'].mean():.2f}",
                'Avg Kills': f"{last_n['kills'].mean():.2f}",
                'Max Reward': f"{config_df['reward'].max():.2f}",
                'Final Reward': f"{config_df.iloc[-1]['reward']:.2f}"
            })

        summary_df = pd.DataFrame(summary_data)

        # Convert to HTML table
        html = summary_df.to_html(index=False, escape=False, border=0)

        return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML dashboard for training analysis"
    )

    parser.add_argument("--models-dir", type=str, default="./models",
                       help="Directory containing training logs")
    parser.add_argument("--output", type=str, default="training_dashboard.html",
                       help="Output HTML file path")

    args = parser.parse_args()

    # Load data
    print(f"Loading training data from {args.models_dir}...")
    loader = TrainingDataLoader(args.models_dir)
    df = loader.load_all_runs()

    if df.empty:
        print("No training data found!")
        return

    print(f"\nLoaded {len(df)} total episodes from {df['config'].nunique()} configurations")

    # Generate dashboard
    print("\nGenerating dashboard...")
    generator = HTMLDashboardGenerator(df)
    generator.generate(args.output)


if __name__ == "__main__":
    main()
