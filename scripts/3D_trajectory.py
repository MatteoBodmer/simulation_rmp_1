#!/usr/bin/env python3
"""
3D Trajectory Plotting Script for RMP Evaluation

This script creates 3D visualizations showing:
- End-effector trajectory during simulation (reconstructed from actual data)
- Obstacle cylinder position (automatically extracted from evaluation data)
- Target position (automatically extracted from evaluation data)
- Start position (automatically extracted from evaluation data)

Command: python3 /home/matteo/franka_ros2_ws/src/simulation_rmp/scripts/3D_trajectory.py

CONFIGURATION - Edit these variables at the top:
"""

# ==================== MANUAL CONFIGURATION ====================
# Edit these variables to specify which data to plot:

RUN_FOLDER_NAME = "Run_20250808_150110"  # Name of the run folder
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
import numpy as np

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
        print(f"  {sim}: {timestamp} - Goal reached: {goal_reached}")
    
    return simulations

def extract_cylinder_dimensions_from_evaluation_manager():
    """
    Extract cylinder dimensions from the evaluation manager source code.
    This reads the actual values used in the evaluation manager.
    """
    # Default values (matching evaluation_manager_rmp.py)
    default_height = 0.4  # meters
    default_radius = 0.12  # meters
    
    try:
        # Try to read from the evaluation manager source file
        evaluation_manager_path = "/home/matteo/franka_ros2_ws/src/simulation_rmp/simulation_rmp/evaluation_manager_rmp.py"
        
        if os.path.exists(evaluation_manager_path):
            with open(evaluation_manager_path, 'r') as f:
                content = f.read()
                
                # Look for cylinder dimensions in the spawn_obstacle method
                # cylinder.dimensions = [height, radius]
                import re
                
                # Search for cylinder.dimensions = [height, radius] pattern
                dimensions_match = re.search(r'cylinder\.dimensions\s*=\s*\[([0-9.]+),\s*([0-9.]+)\]', content)
                if dimensions_match:
                    height = float(dimensions_match.group(1))
                    radius = float(dimensions_match.group(2))
                    print(f"Extracted cylinder dimensions from evaluation manager: height={height}m, radius={radius}m")
                    return height, radius
                
                # Alternative search patterns
                height_match = re.search(r'cylinder_height\s*=\s*([0-9.]+)', content)
                radius_match = re.search(r'cylinder_radius\s*=\s*([0-9.]+)', content)
                
                if height_match and radius_match:
                    height = float(height_match.group(1))
                    radius = float(radius_match.group(1))
                    print(f"Extracted cylinder dimensions from variables: height={height}m, radius={radius}m")
                    return height, radius
    except Exception as e:
        print(f"Could not extract dimensions from evaluation manager: {e}")
    
    print(f"Using default cylinder dimensions: height={default_height}m, radius={default_radius}m")
    return default_height, default_radius

def create_cylinder_mesh(center, radius, height, resolution=20):
    """Create a cylinder mesh for 3D visualization"""
    # Create cylinder surface
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(-height/2, height/2, 10)
    
    # Cylinder side surface
    theta_mesh, z_mesh = np.meshgrid(theta, z)
    x_mesh = center[0] + radius * np.cos(theta_mesh)
    y_mesh = center[1] + radius * np.sin(theta_mesh)
    z_mesh = center[2] + z_mesh
    
    return x_mesh, y_mesh, z_mesh

def extract_real_trajectory_from_distance_data(raw_distance_data):
    """
    Extract real end-effector positions from the distance data.
    """
    trajectory = []
    timestamps = []
    
    if not raw_distance_data:
        print("  No distance data available")
        return [], []
    
    # Check if actual end-effector positions are stored
    for entry in raw_distance_data:
        if 'end_effector_position' in entry and entry['end_effector_position'] is not None:
            timestamps.append(entry['timestamp'])
            trajectory.append(entry['end_effector_position'])
    
    if len(trajectory) > 0:
        print(f"  Found {len(trajectory)} real end-effector positions!")
        return np.array(trajectory), timestamps
    else:
        print("  No real end-effector positions found in data")
        print("  Make sure evaluation_manager_rmp.py stores 'end_effector_position' in distance entries")
        return [], []

def create_placeholder_warning_message():
    """Create a text annotation explaining the missing trajectory data"""
    return {
        'text': "Real trajectory data not available<br>" +
                "Modify evaluation_manager_rmp.py to store<br>" +
                "end-effector positions for real trajectories",
        'x': 0.02,
        'y': 0.98,
        'xref': 'paper',
        'yref': 'paper',
        'showarrow': False,
        'bgcolor': 'rgba(255, 255, 0, 0.8)',
        'bordercolor': 'orange',
        'borderwidth': 2,
        'font': {'size': 10, 'color': 'black'}
    }


def plot_3d_simulation(data, simulation_key, save_dir=None, json_filename=None):
    """
    Create 3D trajectory plot for a single simulation using data from evaluation manager
    
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
    
    # Extract positions directly from evaluation manager data
    obstacle_pos = result.get('obstacle_position')
    target_pos = result.get('target_position')
    
    if not obstacle_pos or not target_pos:
        print(f"Error: Missing position data in {simulation_key}")
        return None
    
    # Extract cylinder dimensions from evaluation manager
    cylinder_height, cylinder_radius = extract_cylinder_dimensions_from_evaluation_manager()
    
    print(f"Creating 3D plot for {simulation_key}")
    print(f"  Obstacle at: [{obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}, {obstacle_pos[2]:.3f}]")
    print(f"  Target at: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    print(f"  Cylinder: height={cylinder_height}m, radius={cylinder_radius}m")
    
    # Create the 3D plot
    fig = go.Figure()
    
    # Try to extract real trajectory from distance data
    trajectory, timestamps = extract_real_trajectory_from_distance_data(result.get('raw_distance_data', []))
    
    has_real_trajectory = len(trajectory) > 0
    
    if has_real_trajectory:
        # Add real trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1], 
            z=trajectory[:, 2],
            mode='lines+markers',
            name='End-Effector Trajectory',
            line=dict(color='blue', width=4),
            marker=dict(size=2, color='blue'),
            hovertemplate='<b>End-Effector Trajectory</b><br>' +
                         'Time: %{text}s<br>' +
                         'X: %{x:.3f}m<br>' +
                         'Y: %{y:.3f}m<br>' +
                         'Z: %{z:.3f}m<br>' +
                         '<extra></extra>',
            text=[f'{t:.2f}' for t in timestamps]
        ))
        
        # Add start position marker (smaller and more elegant) - FIXED SYMBOL
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode='markers',
            name='Start Position',
            marker=dict(size=8, color='darkgreen', symbol='circle', 
                       line=dict(width=2, color='white')),
            hovertemplate='<b>Start Position</b><br>' +
                         'X: %{x:.3f}m<br>' +
                         'Y: %{y:.3f}m<br>' +
                         'Z: %{z:.3f}m<br>' +
                         '<extra></extra>'
        ))
        
        # Add final position marker (smaller and more elegant) - FIXED SYMBOL
        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            z=[trajectory[-1, 2]],
            mode='markers',
            name='Final Position',
            marker=dict(size=8, color='darkorange', symbol='square',
                       line=dict(width=2, color='white')),
            hovertemplate='<b>Final Position</b><br>' +
                         'X: %{x:.3f}m<br>' +
                         'Y: %{y:.3f}m<br>' +
                         'Z: %{z:.3f}m<br>' +
                         '<extra></extra>'
        ))
    else:
        print("  Warning: No real trajectory data available - showing positions only")
        
        # Show approximate start position (typical Franka home) - FIXED SYMBOL
        approx_start = [0.4, 0.0, 0.6]
        fig.add_trace(go.Scatter3d(
            x=[approx_start[0]],
            y=[approx_start[1]],
            z=[approx_start[2]],
            mode='markers',
            name='Approx Start Position',
            marker=dict(size=8, color='lightgreen', symbol='circle-open',
                       line=dict(width=2, color='darkgreen')),
            hovertemplate='<b>Approximate Start Position</b><br>' +
                         'X: %{x:.3f}m<br>' +
                         'Y: %{y:.3f}m<br>' +
                         'Z: %{z:.3f}m<br>' +
                         '<extra></extra>'
        ))
    
    # Add target position marker (smaller and more elegant) - FIXED SYMBOL
    fig.add_trace(go.Scatter3d(
        x=[target_pos[0]],
        y=[target_pos[1]],
        z=[target_pos[2]],
        mode='markers',
        name='Target Position',
        marker=dict(size=8, color='red', symbol='circle',  # Changed from 'x-thin' to 'x'
                   line=dict(width=2, color='darkred')),
        hovertemplate='<b>Target Position</b><br>' +
                     'X: %{x:.3f}m<br>' +
                     'Y: %{y:.3f}m<br>' +
                     'Z: %{z:.3f}m<br>' +
                     '<extra></extra>'
    ))
    
    # Create cylinder obstacle using extracted dimensions
    x_cyl, y_cyl, z_cyl = create_cylinder_mesh(
        obstacle_pos, cylinder_radius, cylinder_height
    )
    
    # Add cylinder surface
    fig.add_trace(go.Surface(
        x=x_cyl,
        y=y_cyl,
        z=z_cyl,
        name='Obstacle Cylinder',
        opacity=0.7,
        colorscale='Reds',
        showscale=False,
        hovertemplate='<b>Obstacle Cylinder</b><br>' +
                     'Center: [%.3f, %.3f, %.3f]<br>' % tuple(obstacle_pos) +
                     'Radius: %.3fm<br>' % cylinder_radius +
                     'Height: %.3fm<br>' % cylinder_height +
                     '<extra></extra>'
    ))
    
    # Add cylinder top and bottom caps using extracted dimensions
    theta = np.linspace(0, 2*np.pi, 20)
    r = np.linspace(0, cylinder_radius, 10)
    theta_cap, r_cap = np.meshgrid(theta, r)
    
    # Top cap
    x_top = obstacle_pos[0] + r_cap * np.cos(theta_cap)
    y_top = obstacle_pos[1] + r_cap * np.sin(theta_cap)
    z_top = np.full_like(x_top, obstacle_pos[2] + cylinder_height/2)
    
    fig.add_trace(go.Surface(
        x=x_top, y=y_top, z=z_top,
        name='Cylinder Top',
        opacity=0.7,
        colorscale='Reds',
        showscale=False,
        showlegend=False
    ))
    
    # Bottom cap
    z_bottom = np.full_like(x_top, obstacle_pos[2] - cylinder_height/2)
    fig.add_trace(go.Surface(
        x=x_top, y=y_top, z=z_bottom,
        name='Cylinder Bottom',
        opacity=0.7,
        colorscale='Reds',
        showscale=False,
        showlegend=False
    ))
    
    # Create title with simulation info
    sim_number = simulation_key.split('_')[1]
    title_text = f'3D End-Effector Trajectory - Simulation {sim_number}'
    subtitle_parts = []
    
    # Add run folder info to subtitle
    subtitle_parts.append(f'Run: {RUN_FOLDER_NAME}')
    
    # Add obstacle and target info
    subtitle_parts.append(f'Obstacle: [{obstacle_pos[0]:.2f}, {obstacle_pos[1]:.2f}, {obstacle_pos[2]:.2f}]')
    subtitle_parts.append(f'Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]')
    
    if 'goal_reached' in result:
        subtitle_parts.append(f'Goal Reached: {result["goal_reached"]}')
    
    if 'goal_reach_time' in result and result['goal_reach_time'] is not None:
        subtitle_parts.append(f'Time: {result["goal_reach_time"]:.2f}s')
    
    if 'timestamp' in result:
        subtitle_parts.append(f'Date: {result["timestamp"][:19]}')
    
    if subtitle_parts:
        title_text += f'<br><sub>{" | ".join(subtitle_parts)}</sub>'
    
    # Update layout for 3D
    layout_dict = {
        'title': dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ),
        'scene': dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'  # Keep proportions
        ),
        'width': 1000,
        'height': 800,
        'margin': dict(r=100, l=100, t=120, b=100)
    }
    
    # Add warning annotation if no real trajectory data
    if not has_real_trajectory:
        layout_dict['annotations'] = [create_placeholder_warning_message()]
    
    fig.update_layout(layout_dict)
    
    # Determine save directory
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_filename)) if json_filename else os.getcwd()
        print(f"Auto-detected save directory: {save_dir}")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"3d_trajectory_{simulation_key}_{timestamp}"
    
    # Save as HTML
    html_filename = os.path.join(save_dir, f"{base_name}.html")
    fig.write_html(html_filename)
    print(f"3D plot saved to: {html_filename}")
    
    # Try to save as PNG
    try:
        png_filename = os.path.join(save_dir, f"{base_name}.png")
        fig.write_image(png_filename, width=1200, height=900, scale=2)
        print(f"3D plot image saved to: {png_filename}")
    except Exception as e:
        print(f"Could not save PNG image: {e}")
        print("   Install kaleido for image export: pip install kaleido")
    
    # Print summary with extracted data
    print(f"\n3D Plot Summary for {simulation_key}:")
    print(f"   Obstacle position: [{obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}, {obstacle_pos[2]:.3f}]")
    print(f"   Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    print(f"   Cylinder dimensions: {cylinder_height}m height, {cylinder_radius}m radius")
    
    if has_real_trajectory:
        print(f"   Real trajectory points: {len(trajectory)}")
        print(f"   Start position: [{trajectory[0, 0]:.3f}, {trajectory[0, 1]:.3f}, {trajectory[0, 2]:.3f}]")
        print(f"   End position: [{trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f}, {trajectory[-1, 2]:.3f}]")
        
        # Calculate trajectory distance
        total_distance = 0
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i] - trajectory[i-1])
            total_distance += dist
        print(f"   Total trajectory length: {total_distance:.3f}m")
        
        # Distance from final position to target
        final_to_target = np.linalg.norm(trajectory[-1] - np.array(target_pos))
        print(f"   Final distance to target: {final_to_target:.3f}m")
    else:
        print("   No real trajectory data available")
        print("   To get actual trajectories, modify evaluation_manager_rmp.py")
    
    # Show the plot
    print(f"\nOpening 3D trajectory plot in browser...")
    fig.show()
    
    return fig

def plot_3d_trajectories_from_file(json_filename, simulation_numbers=None, save_dir=None):
    """
    Load JSON file and create 3D trajectory plots
    """
    
    # Check if file exists
    if not os.path.exists(json_filename):
        print(f"Error: File {json_filename} not found!")
        return None
    
    # Auto-detect save directory
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_filename))
        print(f"Using JSON file directory for plots: {save_dir}")
    
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
    
    # Determine which simulations to plot
    if simulation_numbers is None:
        sims_to_plot = available_sims
    else:
        sims_to_plot = [f'simulation_{num}' for num in simulation_numbers if f'simulation_{num}' in data]
        
        invalid_sims = [num for num in simulation_numbers if f'simulation_{num}' not in data]
        if invalid_sims:
            print(f"Warning: Simulations not found: {invalid_sims}")
    
    if not sims_to_plot:
        print("No valid simulations to plot!")
        return None
    
    print(f"Creating 3D plots for {len(sims_to_plot)} simulation(s): {', '.join(sims_to_plot)}")
    
    # Plot each simulation separately
    figures = []
    for sim_key in sims_to_plot:
        print(f"\n" + "="*50)
        fig = plot_3d_simulation(data, sim_key, save_dir, json_filename)
        if fig:
            figures.append((sim_key, fig))
    
    print(f"\nSuccessfully created {len(figures)} 3D trajectory plots!")
    return figures

def main():
    """Main function that uses the configured settings"""
    
    # Install check
    try:
        import plotly
    except ImportError:
        print("Error: plotly not installed")
        print("Install with: pip install plotly")
        print("For image export: pip install kaleido")
        sys.exit(1)
    
    # Get configured paths
    run_dir, json_file_path = get_configured_paths()
    
    # Print configuration
    print("=" * 60)
    print("3D TRAJECTORY PLOTTING TOOL")
    print("=" * 60)
    print(f"Base directory: {BASE_SIMULATION_DIR}")
    print(f"Run folder: {RUN_FOLDER_NAME}")
    print(f"JSON file: {JSON_FILENAME}")
    print(f"Full path: {json_file_path}")
    print(f"Plots will be saved to: {run_dir}")
    print("=" * 60)
    
    # Check if paths exist
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        print(f"Available run folders in {BASE_SIMULATION_DIR}:")
        try:
            for item in os.listdir(BASE_SIMULATION_DIR):
                if os.path.isdir(os.path.join(BASE_SIMULATION_DIR, item)) and item.startswith('Run_'):
                    print(f"   - {item}")
        except:
            print("   Could not list directories")
        return
    
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found: {json_file_path}")
        print(f"Available files in {run_dir}:")
        try:
            for item in os.listdir(run_dir):
                if item.endswith('.json'):
                    print(f"   - {item}")
        except:
            print("   Could not list files")
        return
    
    # List available simulations
    print("\nAvailable simulations:")
    available_sims = list_available_simulations(json_file_path)
    if not available_sims:
        return
    
    # Determine which simulations to plot
    if PLOT_ALL_SIMULATIONS:
        simulation_numbers = None
        print(f"\nConfiguration: Plot ALL simulations")
    else:
        simulation_numbers = SPECIFIC_SIMULATIONS
        print(f"\nConfiguration: Plot specific simulations: {SPECIFIC_SIMULATIONS}")
    
    # Create plots
    print(f"\nProcessing 3D trajectory data from: {json_file_path}")
    figures = plot_3d_trajectories_from_file(json_file_path, simulation_numbers, run_dir)
    
    if figures is None:
        print(f"Failed to create plots")
    else:
        print(f"\nSuccessfully created {len(figures)} 3D trajectory plots!")
        print(f"All plots saved to: {run_dir}")
        print("\nTo get real trajectory data instead of approximations,")
        print("modify evaluation_manager_rmp.py to store end-effector positions.")

def show_usage():
    """Show usage information"""
    print("3D Trajectory Plotting Script - Manual Configuration Mode")
    print("=" * 60)
    print("To use this script:")
    print("1. Edit the configuration variables at the top of this file:")
    print(f"   - RUN_FOLDER_NAME (currently: '{RUN_FOLDER_NAME}')")
    print(f"   - JSON_FILENAME (currently: '{JSON_FILENAME}')")
    print(f"   - PLOT_ALL_SIMULATIONS (currently: {PLOT_ALL_SIMULATIONS})")
    print(f"   - SPECIFIC_SIMULATIONS (currently: {SPECIFIC_SIMULATIONS})")
    print()
    print("2. Run the script:")
    print("   python3 plot_3d_trajectory.py")
    print()
    print("3. Optional command line arguments:")
    print("   python3 plot_3d_trajectory.py --list    # List available simulations")
    print("   python3 plot_3d_trajectory.py --help    # Show this help")
    print("=" * 60)
    print("\nNote: For real trajectories, modify evaluation_manager_rmp.py")
    print("to store actual end-effector positions during simulation.")

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