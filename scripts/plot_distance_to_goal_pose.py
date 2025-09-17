#!/usr/bin/env python3
"""
Distance to Goal Plotting Script for RMP Evaluation

This script loads JSON results from the evaluation manager and creates
interactive plots showing distance from end-effector to target goal over time.

Command: python3 /home/matteo/franka_ros2_ws/src/simulation_rmp_1/scripts/plot_distance_to_goal_pose.py

CONFIGURATION - Edit these variables at the top:
"""

# ==================== MANUAL CONFIGURATION ====================
# Edit these variables to specify which data to plot:

RUN_FOLDER_NAME = "Run_20250917_143905"  # Name of the run folder (e.g., "Run_20250808_143022")
JSON_FILENAME = "evaluation_results_with_distances.json"  # Name of the JSON file

# Specify which simulation to plot:
SIMULATION_NUMBER = 4  # Simulation number to plot

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
        target_pos = sim_data.get('target_position', [])
        if len(target_pos) == 3:
            target_str = f"[{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]"
        else:
            target_str = "Unknown"
        print(f"  {sim}: {timestamp} - Goal reached: {goal_reached} - Obstacles: {num_obstacles} - Target: {target_str}")
    
    return simulations

def plot_distance_to_goal(data, simulation_key, save_dir=None, json_filename=None):
    """
    Create interactive distance to goal plot for a single simulation
    
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
    
    # Check if distance_to_target data is available
    has_goal_distance_data = any('distance_to_target' in entry and entry['distance_to_target'] is not None 
                                 for entry in result['raw_distance_data'])
    
    if not has_goal_distance_data:
        print(f"No distance_to_target data available for {simulation_key}")
        print("This data is only available in newer evaluation runs that include goal distance logging.")
        return None
    
    # Get simulation information
    num_obstacles = result.get('num_obstacles', 'Unknown')
    target_position = result.get('target_position', [])
    goal_reached = result.get('goal_reached', False)
    goal_reach_time = result.get('goal_reach_time', None)
    goal_tolerance = result.get('goal_tolerance', 0.02)
    
    print(f"Plotting {simulation_key} distance to goal")
    print(f"  Number of obstacles: {num_obstacles}")
    if target_position:
        print(f"  Target position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
    print(f"  Goal reached: {goal_reached}")
    if goal_reach_time:
        print(f"  Goal reach time: {goal_reach_time:.2f}s")
    
    # Extract data for plotting
    timestamps = []
    distances_to_goal = []
    
    for entry in result['raw_distance_data']:
        if 'distance_to_target' in entry and entry['distance_to_target'] is not None:
            timestamps.append(entry['timestamp'])
            distances_to_goal.append(entry['distance_to_target'])
    
    if not timestamps:
        print(f"No valid distance_to_target data found in {simulation_key}")
        return None
    
    print(f"Plotting {len(timestamps)} distance-to-goal measurements")
    
    # Create the plot
    fig = go.Figure()
    
    # Add main distance trace
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=distances_to_goal,
        mode='lines+markers',
        name='Distance to Goal',
        line=dict(color='#2E86C1', width=3),
        marker=dict(size=4, color='#2E86C1'),
        hovertemplate='<b>Distance to Goal</b><br>' +
                     'Time: %{x:.2f}s<br>' +
                     'Distance: %{y:.3f}m<br>' +
                     '<extra></extra>'
    ))
    
    # Create title with simulation info
    sim_number = simulation_key.split('_')[1]
    title_text = f'Distance to Goal Over Time - Simulation {sim_number}'
    subtitle_parts = []
    
    # Add run folder info to subtitle
    subtitle_parts.append(f'Run: {RUN_FOLDER_NAME}')
    
    # Add obstacle information
    subtitle_parts.append(f'Obstacles: {num_obstacles}')
    
    if target_position:
        subtitle_parts.append(f'Target: [{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}]m')
    
    subtitle_parts.append(f'Goal Reached: {goal_reached}')
    
    if goal_reach_time is not None:
        subtitle_parts.append(f'Time: {goal_reach_time:.2f}s')
    
    if 'timestamp' in result:
        subtitle_parts.append(f'Date: {result["timestamp"][:19]}')
    
    if subtitle_parts:
        title_text += f'<br><sub>{" | ".join(subtitle_parts)}</sub>'
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ),
        xaxis_title='Time (seconds)',
        yaxis_title='Distance to Goal (meters)',
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
        margin=dict(r=180, l=80, t=130, b=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Add horizontal reference line for goal tolerance
    fig.add_hline(y=goal_tolerance, line_dash="dash", line_color="green", line_width=2,
                  annotation_text=f"Goal Tolerance ({goal_tolerance*100:.0f}cm)", 
                  annotation_position="bottom right")
    
    # Add goal achievement marker if applicable
    if goal_reached and goal_reach_time is not None:
        fig.add_vline(x=goal_reach_time, line_dash="dot", line_color="green", line_width=3,
                      annotation_text=f"Goal Reached<br>{goal_reach_time:.2f}s",
                      annotation_position="top")
        
        # Add a marker at the exact goal achievement point
        goal_distance_at_time = None
        for i, t in enumerate(timestamps):
            if abs(t - goal_reach_time) < 0.1:  # Find closest timestamp
                goal_distance_at_time = distances_to_goal[i]
                break
        
        if goal_distance_at_time is not None:
            fig.add_trace(go.Scatter(
                x=[goal_reach_time],
                y=[goal_distance_at_time],
                mode='markers',
                name='Goal Achievement',
                marker=dict(size=15, color='green', symbol='star'),
                hovertemplate=f'<b>Goal Achieved!</b><br>' +
                             f'Time: {goal_reach_time:.2f}s<br>' +
                             f'Distance: {goal_distance_at_time:.3f}m<br>' +
                             '<extra></extra>'
            ))
    
    # Add information box with simulation details
    info_text = f"Simulation Details:<br>"
    info_text += f"• Target: [{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}]m<br>" if target_position else ""
    info_text += f"• Goal tolerance: {goal_tolerance*100:.0f}cm<br>"
    info_text += f"• Obstacles: {num_obstacles}<br>"
    info_text += f"• Goal reached: {'✓ Yes' if goal_reached else '✗ No'}<br>"
    if goal_reach_time is not None:
        info_text += f"• Achievement time: {goal_reach_time:.2f}s<br>"
    
    # Add initial and final distances
    if distances_to_goal:
        info_text += f"• Initial distance: {distances_to_goal[0]:.3f}m<br>"
        info_text += f"• Final distance: {distances_to_goal[-1]:.3f}m<br>"
        info_text += f"• Min distance: {min(distances_to_goal):.3f}m<br>"
        distance_reduction = distances_to_goal[0] - distances_to_goal[-1]
        info_text += f"• Distance reduction: {distance_reduction:.3f}m"
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=info_text,
        showarrow=False,
        font=dict(size=10, color="black"),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="gray",
        borderwidth=1,
        align="left"
    )
    
    # Determine save directory
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_filename)) if json_filename else os.getcwd()
        print(f"Auto-detected save directory: {save_dir}")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    goal_status = "reached" if goal_reached else "not_reached"
    base_name = f"goal_distance_plot_{simulation_key}_{num_obstacles}obs_{goal_status}_{timestamp}"
    
    # Save as HTML
    html_filename = os.path.join(save_dir, f"{base_name}.html")
    fig.write_html(html_filename)
    print(f"Interactive plot saved to: {html_filename}")
    
    # Try to save as PNG
    try:
        png_filename = os.path.join(save_dir, f"{base_name}.png")
        fig.write_image(png_filename, width=1400, height=800, scale=2)
        print(f"Plot image saved to: {png_filename}")
    except Exception as e:
        print(f"Could not save PNG image: {e}")
        print("   Install kaleido for image export: pip install kaleido")
    
    # Print summary statistics
    print(f"\nPlot Summary for {simulation_key}:")
    print(f"   Time range: {min(timestamps):.2f}s to {max(timestamps):.2f}s")
    print(f"   Data points: {len(timestamps)}")
    print(f"   Goal reached: {goal_reached}")
    
    if distances_to_goal:
        print(f"   Initial distance to goal: {distances_to_goal[0]:.3f}m")
        print(f"   Final distance to goal: {distances_to_goal[-1]:.3f}m")
        print(f"   Minimum distance achieved: {min(distances_to_goal):.3f}m")
        print(f"   Distance reduction: {distances_to_goal[0] - distances_to_goal[-1]:.3f}m")
    
    if goal_reach_time is not None:
        print(f"   Goal achievement time: {goal_reach_time:.2f}s")
    
    # Show the plot in browser
    print(f"\nOpening {simulation_key} goal distance plot in browser...")
    fig.show()
    
    return fig

def plot_goal_distance_from_file(json_filename, simulation_number, save_dir=None):
    """
    Load JSON file and create interactive goal distance plot
    
    Args:
        json_filename (str): Path to the JSON results file
        simulation_number (int): Simulation number to plot
        save_dir (str): Directory to save plots (if None, uses JSON file's directory)
    """
    
    # Check if file exists
    if not os.path.exists(json_filename):
        print(f"Error: File {json_filename} not found!")
        return None
    
    # Auto-detect save directory from JSON file location if not specified
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_filename))
        print(f" Using JSON file directory for plots: {save_dir}")
    
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
    
    # Check if requested simulation exists
    simulation_key = f'simulation_{simulation_number}'
    if simulation_key not in data:
        print(f" Error: {simulation_key} not found in data!")
        print(f"Available simulations: {', '.join([s.split('_')[1] for s in available_sims])}")
        return None
    
    print(f"Plotting simulation {simulation_number}")
    
    # Create the plot
    fig = plot_distance_to_goal(data, simulation_key, save_dir, json_filename)
    
    if fig is None:
        print(f" Failed to create plot")
        return None
    else:
        print(f" Successfully created goal distance plot!")
        print(f" Plot saved to: {save_dir}")
        return fig

def main():
    """Main function that uses the configured settings"""
    
    # Install check
    try:
        import plotly
    except ImportError:
        print(" Error: plotly not installed")
        print("Install with: pip install plotly")
        print("For image export: pip install kaleido")
        sys.exit(1)
    
    # Get configured paths
    run_dir, json_file_path = get_configured_paths()
    
    # Print configuration
    print("=" * 60)
    print("DISTANCE TO GOAL PLOTTING TOOL")
    print("=" * 60)
    print(f" Base directory: {BASE_SIMULATION_DIR}")
    print(f" Run folder: {RUN_FOLDER_NAME}")
    print(f" JSON file: {JSON_FILENAME}")
    print(f" Simulation number: {SIMULATION_NUMBER}")
    print(f" Full path: {json_file_path}")
    print(f" Plots will be saved to: {run_dir}")
    print("=" * 60)
    
    # Check if paths exist
    if not os.path.exists(run_dir):
        print(f" Error: Run directory not found: {run_dir}")
        print(f"Available run folders in {BASE_SIMULATION_DIR}:")
        try:
            for item in os.listdir(BASE_SIMULATION_DIR):
                if os.path.isdir(os.path.join(BASE_SIMULATION_DIR, item)) and item.startswith('Run_'):
                    print(f"   - {item}")
        except:
            print("   Could not list directories")
        return
    
    if not os.path.exists(json_file_path):
        print(f" Error: JSON file not found: {json_file_path}")
        print(f"Available files in {run_dir}:")
        try:
            for item in os.listdir(run_dir):
                if item.endswith('.json'):
                    print(f"   - {item}")
        except:
            print("   Could not list files")
        return
    
    # List available simulations first
    print("\n Available simulations:")
    available_sims = list_available_simulations(json_file_path)
    if not available_sims:
        return
    
    # Create the plot
    print(f"\n Processing goal distance data from: {json_file_path}")
    fig = plot_goal_distance_from_file(json_file_path, SIMULATION_NUMBER, run_dir)
    
    if fig is None:
        print(f" Failed to create goal distance plot")
    else:
        print(f"\n Successfully created goal distance plot!")
        print(f" Plot saved to: {run_dir}")

def show_usage():
    """Show usage information"""
    print("Distance to Goal Plotting Script")
    print("=" * 50)
    print("To use this script:")
    print("1. Edit the configuration variables at the top of this file:")
    print(f"   - RUN_FOLDER_NAME (currently: '{RUN_FOLDER_NAME}')")
    print(f"   - JSON_FILENAME (currently: '{JSON_FILENAME}')")
    print(f"   - SIMULATION_NUMBER (currently: {SIMULATION_NUMBER})")
    print()
    print("2. Run the script:")
    print("   python3 plot_distance_to_goal_pose.py")
    print()
    print("3. Optional command line arguments:")
    print("   python3 plot_distance_to_goal_pose.py --list    # List available simulations")
    print("   python3 plot_distance_to_goal_pose.py --help    # Show this help")
    print()
    print("This plots the distance from end-effector to target goal over time.")
    print("Shows goal achievement status and timing if applicable.")
    print("=" * 50)

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
                print(f"Error: JSON file not found: {json_file_path}")
            sys.exit(0)
    
    # Run main function
    main()