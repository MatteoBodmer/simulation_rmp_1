#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Handle NumPy compatibility issue
try:
    import numpy as np
    # Check NumPy version
    numpy_version = np.__version__
    print(f"NumPy version: {numpy_version}")
    
    if numpy_version.startswith('2.'):
        print("Warning: NumPy 2.x detected. Some packages may have compatibility issues.")
        print("If you encounter errors, consider downgrading: pip install 'numpy<2'")
    
except ImportError:
    print("Error: NumPy not found. Install with: pip install numpy")
    sys.exit(1)

# Try to import plotting libraries with better error handling
try:
    import pandas as pd
except ImportError:
    print("Error: pandas not found. Install with: pip install pandas")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)
except Exception as e:
    print(f"Error importing matplotlib: {e}")
    print("This is likely a NumPy compatibility issue.")
    print("Try: pip install 'numpy<2' matplotlib --force-reinstall")
    sys.exit(1)

try:
    import seaborn as sns
except ImportError:
    print("Error: seaborn not found. Install with: pip install seaborn")
    sys.exit(1)
except Exception as e:
    print(f"Error importing seaborn: {e}")
    print("This is likely a NumPy compatibility issue.")
    print("Try: pip install 'numpy<2' seaborn --force-reinstall")
    sys.exit(1)

class SimulationPlotter:
    def __init__(self, csv_file_path):
        """
        Initialize the plotter with CSV file from analysis_simulations.py
        Can handle both summary CSV and analysis report CSV
        """
        self.csv_file_path = Path(csv_file_path)
        self.data = None
        self.is_summary_csv = False
        self.is_report_csv = False
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.load_data()
        
    def load_data(self):
        """Load and identify the type of CSV file"""
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        
        # Try to read the CSV
        df = pd.read_csv(self.csv_file_path)
        
        # Check if it's a summary CSV (has simulation data columns)
        summary_columns = ['simulation_id', 'goal_reached', 'num_obstacles', 'execution_time']
        if any(col in df.columns for col in summary_columns):
            self.is_summary_csv = True
            self.data = df
            print(f"✓ Loaded summary CSV with {len(df)} simulations")
        else:
            # Assume it's an analysis report CSV - we'll parse it differently
            self.is_report_csv = True
            self.parse_report_csv()
            print(f"✓ Loaded analysis report CSV")
    
    def parse_report_csv(self):
        """Parse the structured analysis report CSV"""
        # For analysis report, we'll extract key tables and create plots
        # This is a simplified approach - in practice you might want more sophisticated parsing
        df = pd.read_csv(self.csv_file_path, header=None)
        
        # Look for key sections and extract data
        self.report_data = {}
        current_section = None
        
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and '===' in str(row.iloc[0]):
                current_section = str(row.iloc[0]).replace('===', '').strip()
                continue
            
            # Skip empty rows or metadata
            if pd.isna(row.iloc[0]) or row.iloc[0] in ['SIMULATION ANALYSIS REPORT', 'Generated:', 'Source file:']:
                continue
                
            # Try to parse table data
            if current_section and pd.notna(row.iloc[0]):
                if current_section not in self.report_data:
                    self.report_data[current_section] = []
                self.report_data[current_section].append(row.values)
    
    def create_success_rate_plots(self):
        """Create plots showing success rates"""
        if not self.is_summary_csv:
            print("Success rate plots require summary CSV data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Simulation Success Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall success rate pie chart
        success_counts = self.data['goal_reached'].value_counts()
        colors = ['#ff7f7f', '#7fbf7f']  # Red for failure, green for success
        wedges, texts, autotexts = axes[0,0].pie(success_counts.values, 
                                                labels=['Failed', 'Successful'], 
                                                colors=colors,
                                                autopct='%1.1f%%',
                                                startangle=90)
        axes[0,0].set_title('Overall Success Rate')
        
        # 2. Success rate by obstacle count
        if 'num_obstacles' in self.data.columns:
            obstacle_success = self.data.groupby('num_obstacles')['goal_reached'].agg(['count', 'sum', 'mean']).reset_index()
            obstacle_success['success_rate'] = obstacle_success['mean'] * 100
            
            bars = axes[0,1].bar(obstacle_success['num_obstacles'], obstacle_success['success_rate'], 
                               color='skyblue', alpha=0.7, edgecolor='navy')
            axes[0,1].set_xlabel('Number of Obstacles')
            axes[0,1].set_ylabel('Success Rate (%)')
            axes[0,1].set_title('Success Rate by Obstacle Complexity')
            axes[0,1].set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, rate in zip(bars, obstacle_success['success_rate']):
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                             f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Completion time distribution (successful runs only)
        if 'goal_reach_time' in self.data.columns:
            successful_times = self.data[self.data['goal_reached'] == True]['goal_reach_time'].dropna()
            if len(successful_times) > 0:
                axes[1,0].hist(successful_times, bins=20, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
                axes[1,0].axvline(successful_times.mean(), color='red', linestyle='--', 
                                label=f'Mean: {successful_times.mean():.2f}s')
                axes[1,0].axvline(successful_times.median(), color='orange', linestyle='--', 
                                label=f'Median: {successful_times.median():.2f}s')
                axes[1,0].set_xlabel('Completion Time (seconds)')
                axes[1,0].set_ylabel('Frequency')
                axes[1,0].set_title('Distribution of Completion Times')
                axes[1,0].legend()
        
        # 4. Safety metrics overview
        safety_metrics = []
        safety_values = []
        
        if 'close_calls_5cm' in self.data.columns:
            total_close_calls = self.data['close_calls_5cm'].fillna(0).sum()
            safety_metrics.append('Close Calls\n(<5cm)')
            safety_values.append(int(total_close_calls))
        
        if 'safety_violations_2cm' in self.data.columns:
            total_violations = self.data['safety_violations_2cm'].fillna(0).sum()
            safety_metrics.append('Safety Violations\n(<2cm)')
            safety_values.append(int(total_violations))
        
        if 'collision_count' in self.data.columns:
            total_collisions = self.data['collision_count'].fillna(0).sum()
            safety_metrics.append('Total\nCollisions')
            safety_values.append(int(total_collisions))
        
        if safety_metrics:
            bars = axes[1,1].bar(safety_metrics, safety_values, color=['orange', 'red', 'darkred'])
            axes[1,1].set_ylabel('Count')
            axes[1,1].set_title('Safety Incidents Overview')
            
            # Add value labels
            for bar, value in zip(bars, safety_values):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + max(safety_values)*0.01,
                             str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_performance_analysis_plots(self):
        """Create plots analyzing robot performance metrics"""
        if not self.is_summary_csv:
            print("Performance analysis plots require summary CSV data")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Robot Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distance analysis
        if 'min_distance_achieved' in self.data.columns:
            success_dist = self.data[self.data['goal_reached'] == True]['min_distance_achieved'].dropna()
            fail_dist = self.data[self.data['goal_reached'] == False]['min_distance_achieved'].dropna()
            
            if len(success_dist) > 0 and len(fail_dist) > 0:
                axes[0,0].hist([success_dist, fail_dist], bins=20, alpha=0.7, 
                             label=['Successful', 'Failed'], color=['green', 'red'])
                axes[0,0].set_xlabel('Minimum Distance Achieved (m)')
                axes[0,0].set_ylabel('Frequency')
                axes[0,0].set_title('Distance Distribution by Outcome')
                axes[0,0].legend()
        
        # 2. Path efficiency analysis
        if 'path_efficiency_ratio' in self.data.columns:
            efficiency_data = self.data['path_efficiency_ratio'].dropna()
            if len(efficiency_data) > 0:
                axes[0,1].hist(efficiency_data, bins=20, color='lightblue', alpha=0.7, edgecolor='navy')
                axes[0,1].axvline(efficiency_data.mean(), color='red', linestyle='--', 
                                label=f'Mean: {efficiency_data.mean():.2f}')
                axes[0,1].set_xlabel('Path Efficiency Ratio')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].set_title('Path Efficiency Distribution')
                axes[0,1].legend()
        
        # 3. Velocity analysis
        if 'overall_avg_velocity' in self.data.columns:
            velocity_data = self.data['overall_avg_velocity'].dropna()
            if len(velocity_data) > 0:
                axes[0,2].boxplot(velocity_data)
                axes[0,2].set_ylabel('Average Velocity (rad/s)')
                axes[0,2].set_title('Joint Velocity Distribution')
                axes[0,2].set_xticklabels(['All Simulations'])
        
        # 4. Safety score by obstacle complexity
        if 'safety_score' in self.data.columns and 'num_obstacles' in self.data.columns:
            # Box plot of safety scores by obstacle count
            obstacle_counts = sorted(self.data['num_obstacles'].dropna().unique())
            safety_by_obstacles = [self.data[self.data['num_obstacles'] == n]['safety_score'].dropna().values 
                                 for n in obstacle_counts]
            
            if any(len(scores) > 0 for scores in safety_by_obstacles):
                bp = axes[1,0].boxplot(safety_by_obstacles, labels=[f'{int(n)}' for n in obstacle_counts])
                axes[1,0].set_xlabel('Number of Obstacles')
                axes[1,0].set_ylabel('Safety Score')
                axes[1,0].set_title('Safety Scores by Complexity')
        
        # 5. Execution time vs success rate
        if 'execution_time' in self.data.columns and 'num_obstacles' in self.data.columns:
            execution_stats = self.data.groupby('num_obstacles').agg({
                'execution_time': ['mean', 'std'],
                'goal_reached': 'mean'
            }).reset_index()
            
            execution_stats.columns = ['num_obstacles', 'mean_time', 'std_time', 'success_rate']
            
            ax2 = axes[1,1].twinx()
            
            # Plot execution time
            bars1 = axes[1,1].bar(execution_stats['num_obstacles'] - 0.2, execution_stats['mean_time'], 
                                width=0.4, alpha=0.7, color='lightcoral', label='Avg Execution Time')
            axes[1,1].set_xlabel('Number of Obstacles')
            axes[1,1].set_ylabel('Average Execution Time (s)', color='red')
            axes[1,1].tick_params(axis='y', labelcolor='red')
            
            # Plot success rate
            line1 = ax2.plot(execution_stats['num_obstacles'], execution_stats['success_rate'] * 100, 
                           'bo-', alpha=0.7, label='Success Rate')
            ax2.set_ylabel('Success Rate (%)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            axes[1,1].set_title('Execution Time vs Success Rate')
        
        # 6. Collision analysis
        if 'collision_count' in self.data.columns:
            collision_data = self.data['collision_count'].fillna(0)
            collision_counts = collision_data.value_counts().sort_index()
            
            bars = axes[1,2].bar(collision_counts.index, collision_counts.values, 
                               color='salmon', alpha=0.7, edgecolor='darkred')
            axes[1,2].set_xlabel('Number of Collisions')
            axes[1,2].set_ylabel('Number of Simulations')
            axes[1,2].set_title('Collision Distribution')
            
            # Add percentage labels
            total_sims = len(self.data)
            for bar, count in zip(bars, collision_counts.values):
                height = bar.get_height()
                pct = (count / total_sims) * 100
                axes[1,2].text(bar.get_x() + bar.get_width()/2., height + max(collision_counts.values)*0.01,
                             f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap of key metrics"""
        if not self.is_summary_csv:
            print("Correlation heatmap requires summary CSV data")
            return
        
        # Select numeric columns for correlation
        numeric_cols = [
            'num_obstacles', 'execution_time', 'goal_reach_time',
            'min_distance_achieved', 'path_efficiency_ratio', 'total_curvature',
            'overall_avg_velocity', 'safety_score', 'collision_count'
        ]
        
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        if len(available_cols) < 3:
            print("Not enough numeric columns for correlation analysis")
            return
        
        correlation_data = self.data[available_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix of Simulation Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return plt.gcf()
    
    def create_failure_analysis_plots(self):
        """Create plots specifically analyzing failures"""
        if not self.is_summary_csv:
            print("Failure analysis plots require summary CSV data")
            return
        
        failed_data = self.data[self.data['goal_reached'] == False].copy()
        if len(failed_data) == 0:
            print("No failed simulations to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Failure Analysis ({len(failed_data)} failed simulations)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Failure reasons breakdown
        failure_reasons = []
        if 'collision_count' in failed_data.columns:
            collisions = len(failed_data[failed_data['collision_count'] > 0])
            failure_reasons.extend(['Collision'] * collisions)
            failure_reasons.extend(['Other'] * (len(failed_data) - collisions))
        
        if failure_reasons:
            failure_counts = pd.Series(failure_reasons).value_counts()
            axes[0,0].pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
                         colors=['red', 'orange'])
            axes[0,0].set_title('Failure Causes')
        
        # 2. Safety metrics in failures
        if 'safety_score' in failed_data.columns:
            safety_scores = failed_data['safety_score'].dropna()
            if len(safety_scores) > 0:
                axes[0,1].hist(safety_scores, bins=15, color='red', alpha=0.7, edgecolor='darkred')
                axes[0,1].axvline(safety_scores.mean(), color='yellow', linestyle='--',
                                label=f'Mean: {safety_scores.mean():.1f}')
                axes[0,1].set_xlabel('Safety Score')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].set_title('Safety Scores in Failed Simulations')
                axes[0,1].legend()
        
        # 3. Minimum distances in failures vs successes
        if 'min_distance_achieved' in self.data.columns:
            success_dist = self.data[self.data['goal_reached'] == True]['min_distance_achieved'].dropna()
            fail_dist = failed_data['min_distance_achieved'].dropna()
            
            if len(success_dist) > 0 and len(fail_dist) > 0:
                axes[1,0].boxplot([success_dist, fail_dist], labels=['Success', 'Failure'])
                axes[1,0].set_ylabel('Minimum Distance (m)')
                axes[1,0].set_title('Distance Comparison: Success vs Failure')
        
        # 4. Failure rate by obstacle complexity
        if 'num_obstacles' in self.data.columns:
            failure_by_obstacles = self.data.groupby('num_obstacles').agg({
                'goal_reached': ['count', lambda x: (~x).sum()]
            }).reset_index()
            failure_by_obstacles.columns = ['num_obstacles', 'total', 'failures']
            failure_by_obstacles['failure_rate'] = (failure_by_obstacles['failures'] / 
                                                   failure_by_obstacles['total']) * 100
            
            bars = axes[1,1].bar(failure_by_obstacles['num_obstacles'], 
                               failure_by_obstacles['failure_rate'],
                               color='red', alpha=0.7, edgecolor='darkred')
            axes[1,1].set_xlabel('Number of Obstacles')
            axes[1,1].set_ylabel('Failure Rate (%)')
            axes[1,1].set_title('Failure Rate by Obstacle Complexity')
            
            # Add value labels
            for bar, rate in zip(bars, failure_by_obstacles['failure_rate']):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                             f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, output_dir=None):
        """Generate and save all plots"""
        if output_dir is None:
            output_dir = self.csv_file_path.parent / "plots"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_plots = []
        
        if self.is_summary_csv:
            print("Generating plots from summary CSV data...")
            
            # Success analysis plots
            try:
                fig1 = self.create_success_rate_plots()
                if fig1:
                    plot_path = output_dir / f"success_analysis_{timestamp}.png"
                    fig1.savefig(plot_path, dpi=300, bbox_inches='tight')
                    saved_plots.append(plot_path)
                    plt.close(fig1)
            except Exception as e:
                print(f"Error creating success plots: {e}")
            
            # Performance analysis plots
            try:
                fig2 = self.create_performance_analysis_plots()
                if fig2:
                    plot_path = output_dir / f"performance_analysis_{timestamp}.png"
                    fig2.savefig(plot_path, dpi=300, bbox_inches='tight')
                    saved_plots.append(plot_path)
                    plt.close(fig2)
            except Exception as e:
                print(f"Error creating performance plots: {e}")
            
            # Correlation heatmap
            try:
                fig3 = self.create_correlation_heatmap()
                if fig3:
                    plot_path = output_dir / f"correlation_heatmap_{timestamp}.png"
                    fig3.savefig(plot_path, dpi=300, bbox_inches='tight')
                    saved_plots.append(plot_path)
                    plt.close(fig3)
            except Exception as e:
                print(f"Error creating correlation heatmap: {e}")
            
            # Failure analysis plots
            try:
                fig4 = self.create_failure_analysis_plots()
                if fig4:
                    plot_path = output_dir / f"failure_analysis_{timestamp}.png"
                    fig4.savefig(plot_path, dpi=300, bbox_inches='tight')
                    saved_plots.append(plot_path)
                    plt.close(fig4)
            except Exception as e:
                print(f"Error creating failure plots: {e}")
        
        else:
            print("Analysis report CSV detected - creating summary visualizations...")
            # For report CSV, we could create simpler plots based on extracted data
            # This would require more sophisticated parsing of the report structure
            
        return saved_plots


def main():
    parser = argparse.ArgumentParser(description='Create visualization plots from simulation analysis CSV')
    parser.add_argument('csv_file', help='Path to CSV file (summary or analysis report)')
    parser.add_argument('--output-dir', help='Directory to save plots (default: same as CSV location)')
    
    args = parser.parse_args()
    
    try:
        plotter = SimulationPlotter(args.csv_file)
        saved_plots = plotter.save_all_plots(args.output_dir)
        
        print(f"\n✓ Successfully generated {len(saved_plots)} plots:")
        for plot_path in saved_plots:
            print(f"  - {plot_path}")
        
        print(f"\nPlots saved in: {Path(saved_plots[0]).parent if saved_plots else 'No plots generated'}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # If run without arguments, use default path
    if len(sys.argv) == 1:
        default_csv = "/home/matteo/Simulation_rmp/Run_1000_1_290925/simulation_summary_1000_20241006_143252.csv"
        print(f"No CSV file specified, using default: {default_csv}")
        sys.argv.append(default_csv)
    
    main()