#!/usr/bin/env python3
"""
Distance Data Plotting Script for RMP Evaluation

This script loads JSON results from the evaluation manager and creates
interactive plots showing distance from each robot link to obstacles over time.

Command: python3 /home/matteo/franka_ros2_ws/src/simulation_rmp_1/scripts/plot_distance_data.py

CONFIGURATION - Edit these variables at the top:
"""

# ==================== MANUAL CONFIGURATION ====================
# Edit these variables to specify which data to plot:

RUN_FOLDER_NAME = "Run_20250814_145854"  # Name of the run folder (e.g., "Run_20250808_143022")
JSON_FILENAME = "evaluation_results_with_distances.json"  # Name of the JSON file

# Specify which simulations to plot:
# Option 1: Plot all simulations
PLOT_ALL_SIMULATIONS = True

# Option 2: Plot specific simulation numbers (set PLOT_ALL_SIMULATIONS = False first)
SPECIFIC_SIMULATIONS = [1, 2, 3]  # List of simulation numbers to plot

# Base directory (usually doesn't need to be changed)
BASE_SIMULATION_DIR = "/home/matteo/Simulation_rmp"

# =============================================================

import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
import argparse

def get_configured_paths():
    """Get the configured file paths"""
    run_dir = os.path.join(BASE_SIMULATION_DIR, RUN_FOLDER_NAME)
    json_file_path = os.path.join(run_dir, JSON_FILENAME)
    
    return run_dir, json_file_path

def list_available_simulations(json_filename):
    """List all available simulations in the JSON file"""
    if not os.path.exists(json_filename):
        print(f"Error: File {json_filename} not found!")
        return []
    
    try:
        with open(json_filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []
    
    simulations = [key for key in data.keys() if key.startswith('simulation_')]
    simulations.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically
    
    print(f"Available simulations in {json_filename}:")
    for sim in simulations:
        sim_data = data[sim]
        timestamp = sim_data.get('timestamp', 'Unknown time')
        goal_reached = sim_data.get('goal_reached', 'Unknown')
        num_obstacles = sim_data.get('num_obstacles', 'Unknown')
        print(f"  {sim}: {timestamp} - Goal reached: {goal_reached} - Obstacles: {num_obstacles}")
    
    return simulations

def plot_single_simulation(data, simulation_key, save_dir=None, json_filename=None):
    """
    Create interactive distance plot for a single simulation
    
    Args:
        data (dict): Full JSON data
        simulation_key (str): Key for the simulation (e.g., 'simulation_1')
        save_dir (str): Directory to save plots
        json_filename (str): Original JSON filename for reference
    """
    
    if simulation_key not in data:
        print(f"Error: {simulation_key} not found in data!")
        return None
    
    result = data[simulation_key]
    
    if not result.get('raw_distance_data'):
        print(f"No raw distance data available for {simulation_key}")
        return None
    
    # Get obstacle information for this simulation
    num_obstacles = result.get('num_obstacles', 'Unknown')
    obstacle_positions = result.get('obstacle_positions', [])
    
    print(f"Plotting {simulation_key} with {len(result['raw_distance_data'])} distance measurements")
    print(f"  Number of obstacles in this simulation: {num_obstacles}")
    if obstacle_positions:
        print(f"  Obstacle positions:")
        for i, pos in enumerate(obstacle_positions):
            print(f"    Obstacle {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Extract data for plotting
    timestamps = []
    link_distances = {
        'link2': [], 'link3': [], 'link4': [], 'link5': [], 
        'link6': [], 'link7': [], 'hand': [], 'end_effector': []
    }
    
    for entry in result['raw_distance_data']:
        timestamps.append(entry['timestamp'])
        for link_name in link_distances.keys():
            if link_name in entry['link_distances']:
                distance = entry['link_distances'][link_name]
                # Replace infinity with None for plotting
                distance = distance if distance != float('inf') else None
                link_distances[link_name].append(distance)
            else:
                link_distances[link_name].append(None)
    
    # Create the plot
    fig = go.Figure()
    
    # Define colors for each link (distinct and visually appealing)
    colors = {
        'link2': '#FF6B6B',      # Red
        'link3': '#4ECDC4',      # Teal
        'link4': '#45B7D1',      # Blue
        'link5': '#96CEB4',      # Green
        'link6': '#FECA57',      # Yellow/Orange
        'link7': '#FF9FF3',      # Pink
        'hand': '#54A0FF',       # Light Blue
        'end_effector': '#5F27CD' # Purple
    }
    
    # Count how many links have actual data
    links_with_data = 0
    
    # Add a trace for each link
    for link_name, distances in link_distances.items():
        # Only plot if there's actual data (not all None/inf)
        valid_distances = [d for d in distances if d is not None]
        if valid_distances:
            links_with_data += 1
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=distances,
                mode='lines+markers',
                name=link_name,
                line=dict(color=colors[link_name], width=2.5),
                marker=dict(size=3),
                connectgaps=False,  # Don't connect across None values
                hovertemplate=f'<b>{link_name}</b><br>' +
                             'Time: %{x:.2f}s<br>' +
                             'Distance: %{y:.3f}m<br>' +
                             '<extra></extra>'
            ))
    
    print(f"Plotting data for {links_with_data} links in {simulation_key}")
    
    # Create title with simulation info including obstacle count
    sim_number = simulation_key.split('_')[1]
    title_text = f'Robot Link Distances to Obstacles Over Time - Simulation {sim_number}'
    subtitle_parts = []
    
    # Add run folder info to subtitle
    subtitle_parts.append(f'Run: {RUN_FOLDER_NAME}')
    
    # Add obstacle information
    subtitle_parts.append(f'Obstacles: {num_obstacles}')
    
    if 'target_position' in result:
        target_pos = result['target_position']
        subtitle_parts.append(f'Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]m')
    
    if 'goal_reached' in result:
        subtitle_parts.append(f'Goal Reached: {result["goal_reached"]}')
    
    if 'goal_reach_time' in result and result['goal_reach_time'] is not None:
        subtitle_parts.append(f'Time: {result["goal_reach_time"]:.2f}s')
    
    # Add direct sampling info if available
    if result.get('direct_sampling_used'):
        subtitle_parts.append('Direct Sampling')
    
    if 'timestamp' in result:
        subtitle_parts.append(f'Date: {result["timestamp"][:19]}')
    
    if subtitle_parts:
        title_text += f'<br><sub>{" | ".join(subtitle_parts)}</sub>'
    
    # Update layout with increased top margin to prevent collision
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            y=0.95,  # Move title higher
            yanchor='top'
        ),
        xaxis_title='Time (seconds)',
        yaxis_title='Distance to Obstacles (meters)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bordercolor="Black",
            borderwidth=1
        ),
        hovermode='x unified',
        width=1200,
        height=700,
        margin=dict(r=180, l=80, t=130, b=80),  # Increased top margin from 100 to 130
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Add horizontal reference lines for safety thresholds
    fig.add_hline(y=0.10, line_dash="dash", line_color="orange", line_width=2,
                  annotation_text="10cm (Warning)", annotation_position="bottom right")
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="5cm (Close Call)", annotation_position="bottom right")
    fig.add_hline(y=0.02, line_dash="dash", line_color="darkred", line_width=3,
                  annotation_text="2cm (Safety Violation)", annotation_position="bottom right")
    
    # Add goal achievement marker if applicable with adjusted position
    if result.get('goal_reached') and result.get('goal_reach_time'):
        fig.add_vline(x=result['goal_reach_time'], line_dash="dot", line_color="green", line_width=3,
                      annotation_text=f"Goal Reached<br>{result['goal_reach_time']:.2f}s",
                      annotation_position="top",
                      annotation=dict(
                          yshift=0  # Move annotation down to avoid collision with subtitle
                      ))
    
    # Add obstacle position annotations if available
    if obstacle_positions:
        # Add text box with obstacle positions
        obstacle_text = f"Obstacle Positions ({num_obstacles} total):<br>"
        for i, pos in enumerate(obstacle_positions):
            obstacle_text += f"#{i+1}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]<br>"
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,  # Top-left corner
            text=obstacle_text,
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            align="left"
        )
    
    # Determine save directory - use same directory as JSON file if not specified
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_filename)) if json_filename else os.getcwd()
        print(f"Auto-detected save directory: {save_dir}")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamped filenames with obstacle count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"distance_plot_{simulation_key}_{num_obstacles}obs_{timestamp}"
    
    # Save as HTML (always works)
    html_filename = os.path.join(save_dir, f"{base_name}.html")
    fig.write_html(html_filename)
    print(f"Interactive plot saved to: {html_filename}")
    
    # Try to save as PNG (requires kaleido)
    try:
        png_filename = os.path.join(save_dir, f"{base_name}.png")
        fig.write_image(png_filename, width=1400, height=800, scale=2)
        print(f"Plot image saved to: {png_filename}")
    except Exception as e:
        print(f"Could not save PNG image: {e}")
        print("   Install kaleido for image export: pip install kaleido")
    
    # Print summary statistics including obstacle information
    print(f"\nPlot Summary for {simulation_key}:")
    print(f"   Time range: {min(timestamps):.2f}s to {max(timestamps):.2f}s")
    print(f"   Links with data: {links_with_data}/8")
    print(f"   Number of obstacles: {num_obstacles}")
    
    if obstacle_positions:
        print(f"   Obstacle positions:")
        for i, pos in enumerate(obstacle_positions):
            print(f"     Obstacle {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    if 'distance_analysis' in result:
        analysis = result['distance_analysis']
        if 'min_distance_achieved' in analysis:
            print(f"   Minimum distance: {analysis['min_distance_achieved']:.3f}m")
            print(f"   Close calls (<5cm): {analysis['close_calls']}")
            print(f"   Safety violations (<2cm): {analysis['safety_violations']}")
    
    # Show the plot in browser
    print(f"\nOpening {simulation_key} plot in browser...")
    fig.show()
    
    return fig

def plot_distance_data_from_file(json_filename, simulation_numbers=None, save_dir=None):
    """
    Load JSON file and create interactive distance plots
    
    Args:
        json_filename (str): Path to the JSON results file
        simulation_numbers (list): List of simulation numbers to plot (None for all)
        save_dir (str): Directory to save plots (if None, uses JSON file's directory)
    """
    
    # Check if file exists
    if not os.path.exists(json_filename):
        print(f"Error: File {json_filename} not found!")
        return None
    
    # Auto-detect save directory from JSON file location if not specified
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_filename))
        print(f"📁 Using JSON file directory for plots: {save_dir}")
    
    # Load the data
    try:
        with open(json_filename, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_filename}")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None
    
    # Get available simulations
    available_sims = [key for key in data.keys() if key.startswith('simulation_')]
    available_sims.sort(key=lambda x: int(x.split('_')[1]))
    
    if not available_sims:
        print("No simulation data found in the file!")
        return None
    
    print(f"Found {len(available_sims)} simulations: {', '.join(available_sims)}")
    
    # Show obstacle count distribution
    obstacle_counts = {}
    for sim_key in available_sims:
        num_obs = data[sim_key].get('num_obstacles', 'Unknown')
        if num_obs != 'Unknown':
            obstacle_counts[num_obs] = obstacle_counts.get(num_obs, 0) + 1
    
    if obstacle_counts:
        print("Obstacle count distribution:")
        for count, freq in sorted(obstacle_counts.items()):
            print(f"  {count} obstacles: {freq} simulation(s)")
    
    # Determine which simulations to plot
    if simulation_numbers is None:
        # Plot all simulations
        sims_to_plot = available_sims
    else:
        # Plot specified simulations
        sims_to_plot = [f'simulation_{num}' for num in simulation_numbers if f'simulation_{num}' in data]
        
        # Check for invalid simulation numbers
        invalid_sims = [num for num in simulation_numbers if f'simulation_{num}' not in data]
        if invalid_sims:
            print(f"⚠️  Warning: Simulations not found: {invalid_sims}")
    
    if not sims_to_plot:
        print("No valid simulations to plot!")
        return None
    
    print(f"Plotting {len(sims_to_plot)} simulation(s): {', '.join(sims_to_plot)}")
    
    # Plot each simulation separately
    figures = []
    for sim_key in sims_to_plot:
        print(f"\n" + "="*50)
        fig = plot_single_simulation(data, sim_key, save_dir, json_filename)
        if fig:
            figures.append((sim_key, fig))
    
    print(f"\n🎉 Successfully created {len(figures)} plots!")
    return figures

def main():
    """Main function that uses the configured settings"""
    
    # Install check
    try:
        import plotly
    except ImportError:
        print("❌ Error: plotly not installed")
        print("Install with: pip install plotly")
        print("For image export: pip install kaleido")
        sys.exit(1)
    
    # Get configured paths
    run_dir, json_file_path = get_configured_paths()
    
    # Print configuration
    print("=" * 60)
    print("DISTANCE DATA PLOTTING TOOL - MULTI-OBSTACLE VERSION")
    print("=" * 60)
    print(f"📁 Base directory: {BASE_SIMULATION_DIR}")
    print(f"📂 Run folder: {RUN_FOLDER_NAME}")
    print(f"📄 JSON file: {JSON_FILENAME}")
    print(f"🎯 Full path: {json_file_path}")
    print(f"💾 Plots will be saved to: {run_dir}")
    print("=" * 60)
    
    # Check if paths exist
    if not os.path.exists(run_dir):
        print(f"❌ Error: Run directory not found: {run_dir}")
        print(f"Available run folders in {BASE_SIMULATION_DIR}:")
        try:
            for item in os.listdir(BASE_SIMULATION_DIR):
                if os.path.isdir(os.path.join(BASE_SIMULATION_DIR, item)) and item.startswith('Run_'):
                    print(f"   - {item}")
        except:
            print("   Could not list directories")
        return
    
    if not os.path.exists(json_file_path):
        print(f"❌ Error: JSON file not found: {json_file_path}")
        print(f"Available files in {run_dir}:")
        try:
            for item in os.listdir(run_dir):
                if item.endswith('.json'):
                    print(f"   - {item}")
        except:
            print("   Could not list files")
        return
    
    # List available simulations first
    print("\n📋 Available simulations:")
    available_sims = list_available_simulations(json_file_path)
    if not available_sims:
        return
    
    # Determine which simulations to plot based on configuration
    if PLOT_ALL_SIMULATIONS:
        simulation_numbers = None  # Plot all
        print(f"\n🎨 Configuration: Plot ALL simulations")
    else:
        simulation_numbers = SPECIFIC_SIMULATIONS
        print(f"\n🎨 Configuration: Plot specific simulations: {SPECIFIC_SIMULATIONS}")
    
    # Create plots
    print(f"\n📈 Processing data from: {json_file_path}")
    figures = plot_distance_data_from_file(json_file_path, simulation_numbers, run_dir)
    
    if figures is None:
        print(f"❌ Failed to create plots")
    else:
        print(f"\n✅ Successfully created {len(figures)} separate plots!")
        print(f"📁 All plots saved to: {run_dir}")

def show_usage():
    """Show usage information"""
    print("Distance Data Plotting Script - Multi-Obstacle Version")
    print("=" * 60)
    print("To use this script:")
    print("1. Edit the configuration variables at the top of this file:")
    print(f"   - RUN_FOLDER_NAME (currently: '{RUN_FOLDER_NAME}')")
    print(f"   - JSON_FILENAME (currently: '{JSON_FILENAME}')")
    print(f"   - PLOT_ALL_SIMULATIONS (currently: {PLOT_ALL_SIMULATIONS})")
    print(f"   - SPECIFIC_SIMULATIONS (currently: {SPECIFIC_SIMULATIONS})")
    print()
    print("2. Run the script:")
    print("   python3 plot_distance_data.py")
    print()
    print("3. Optional command line arguments:")
    print("   python3 plot_distance_data.py --list    # List available simulations")
    print("   python3 plot_distance_data.py --help    # Show this help")
    print()
    print("This version accounts for varying numbers of obstacles per simulation.")
    print("=" * 60)

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            show_usage()
            sys.exit(0)
        elif sys.argv[1] in ['--list', '-l']:
            run_dir, json_file_path = get_configured_paths()
            if os.path.exists(json_file_path):
                list_available_simulations(json_file_path)
            else:
                print(f"❌ Error: JSON file not found: {json_file_path}")
            sys.exit(0)
    
    # Run main function
    main()