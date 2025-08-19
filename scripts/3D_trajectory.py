#!/usr/bin/env python3
"""
3D Trajectory Plotting Script for RMP Evaluation - Multi-Obstacle Version with Exact Geometry

This script creates 3D visualizations showing:
- End-effector trajectory during simulation (reconstructed from actual data)
- Multiple obstacle cylinder positions with EXACT sizes and orientations from simulation
- Target position (automatically extracted from evaluation data)
- Start position (automatically extracted from evaluation data)
- Straight-line distance from start to target (NEW)

Command: python3 /home/matteo/franka_ros2_ws/src/simulation_rmp_1/scripts/3D_trajectory.py

CONFIGURATION - Edit these variables at the top:
"""

# ==================== MANUAL CONFIGURATION ====================
# Edit these variables to specify which data to plot:

RUN_FOLDER_NAME = "Run_20250819_111316"  # Name of the run folder
JSON_FILENAME = "evaluation_results_with_distances.json"  # Name of the JSON file

# Specify which simulations to plot:
# Option 1: Plot all simulations
PLOT_ALL_SIMULATIONS = False

# Option 2: Plot specific simulation numbers (set PLOT_ALL_SIMULATIONS = False first)
SPECIFIC_SIMULATIONS = [5]  # List of simulation numbers to plot

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
        num_obstacles = sim_data.get('num_obstacles', 'Unknown')
        print(f"  {sim}: {timestamp} - Goal reached: {goal_reached} - Obstacles: {num_obstacles}")
    
    return simulations

def create_cylinder_mesh_with_orientation(center, radius, height, orientation_type, resolution=20):
    """Create a cylinder mesh for 3D visualization with correct orientation"""
    # Create basic vertical cylinder first
    theta = np.linspace(0, 2*np.pi, resolution)
    z_local = np.linspace(-height/2, height/2, 10)
    
    # Create meshgrid for cylinder surface
    theta_mesh, z_mesh = np.meshgrid(theta, z_local)
    x_local = radius * np.cos(theta_mesh)
    y_local = radius * np.sin(theta_mesh)
    z_local_mesh = z_mesh
    
    # Apply orientation transformation
    if orientation_type == 'vertical':
        # No transformation needed - default vertical
        x_mesh = center[0] + x_local
        y_mesh = center[1] + y_local
        z_mesh = center[2] + z_local_mesh
        
    elif orientation_type == 'horizontal_x':
        # Cylinder lies along X axis (rotated 90° around Y axis)
        # Transform: x->z, y->y, z->-x
        x_mesh = center[0] + z_local_mesh  # z becomes new x (cylinder extends along x)
        y_mesh = center[1] + y_local       # y stays same
        z_mesh = center[2] - x_local       # x becomes new z (with sign flip)
        
    elif orientation_type == 'horizontal_y':
        # Cylinder lies along Y axis (rotated 90° around X axis)
        # Transform: x->x, y->z, z->y
        x_mesh = center[0] + x_local       # x stays same
        y_mesh = center[1] + z_local_mesh  # z becomes new y (cylinder extends along y)
        z_mesh = center[2] + y_local       # y becomes new z
    
    return x_mesh, y_mesh, z_mesh

def create_cylinder_caps_with_orientation(center, radius, height, orientation_type, resolution=20):
    """Create cylinder end caps with correct orientation"""
    # Create circular cap in local coordinates
    theta = np.linspace(0, 2*np.pi, resolution)
    r = np.linspace(0, radius, 10)
    theta_cap, r_cap = np.meshgrid(theta, r)
    
    x_cap_local = r_cap * np.cos(theta_cap)
    y_cap_local = r_cap * np.sin(theta_cap)
    
    caps = []
    
    if orientation_type == 'vertical':
        # Top cap (z = center[2] + height/2)
        x_top = center[0] + x_cap_local
        y_top = center[1] + y_cap_local
        z_top = np.full_like(x_top, center[2] + height/2)
        
        # Bottom cap (z = center[2] - height/2)
        x_bottom = center[0] + x_cap_local
        y_bottom = center[1] + y_cap_local
        z_bottom = np.full_like(x_bottom, center[2] - height/2)
        
        caps = [
            (x_top, y_top, z_top),
            (x_bottom, y_bottom, z_bottom)
        ]
        
    elif orientation_type == 'horizontal_x':
        # Cylinder along X axis - caps at x = center[0] ± height/2
        # Cap plane is in Y-Z plane
        y_cap1 = center[1] + x_cap_local  # x_local becomes y
        z_cap1 = center[2] + y_cap_local  # y_local becomes z
        x_cap1 = np.full_like(y_cap1, center[0] + height/2)  # positive X cap
        
        y_cap2 = center[1] + x_cap_local
        z_cap2 = center[2] + y_cap_local
        x_cap2 = np.full_like(y_cap2, center[0] - height/2)  # negative X cap
        
        caps = [
            (x_cap1, y_cap1, z_cap1),
            (x_cap2, y_cap2, z_cap2)
        ]
        
    elif orientation_type == 'horizontal_y':
        # Cylinder along Y axis - caps at y = center[1] ± height/2
        # Cap plane is in X-Z plane
        x_cap1 = center[0] + x_cap_local  # x_local stays x
        z_cap1 = center[2] + y_cap_local  # y_local becomes z
        y_cap1 = np.full_like(x_cap1, center[1] + height/2)  # positive Y cap
        
        x_cap2 = center[0] + x_cap_local
        z_cap2 = center[2] + y_cap_local
        y_cap2 = np.full_like(x_cap2, center[1] - height/2)  # negative Y cap
        
        caps = [
            (x_cap1, y_cap1, z_cap1),
            (x_cap2, y_cap2, z_cap2)
        ]
    
    return caps

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
    Create 3D trajectory plot for a single simulation using EXACT geometry from evaluation manager
    
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
    
    # Extract obstacle data from evaluation manager
    obstacle_positions = result.get('obstacle_positions', [])
    obstacle_sizes = result.get('obstacle_sizes', [])  # NEW: Get actual sizes
    target_pos = result.get('target_position')
    num_obstacles = result.get('num_obstacles', len(obstacle_positions))
    
    # NEW: Extract path metrics for straight-line distance
    path_metrics = result.get('path_metrics', {})
    
    if not obstacle_positions or not target_pos:
        print(f"Error: Missing position data in {simulation_key}")
        print(f"  Obstacle positions: {len(obstacle_positions) if obstacle_positions else 0}")
        print(f"  Target position: {'Available' if target_pos else 'Missing'}")
        return None
    
    # Check if we have size data (new format with random sizes)
    if not obstacle_sizes:
        print(f"  Warning: No obstacle size data found - using default dimensions")
        # Fallback to default sizes if data doesn't exist
        default_height, default_radius = 0.4, 0.12
        obstacle_sizes = [
            {'height': default_height, 'radius': default_radius, 'orientation': 'vertical'}
            for _ in range(num_obstacles)
        ]
    
    print(f"Creating 3D plot for {simulation_key} with EXACT geometry")
    print(f"  Number of obstacles: {num_obstacles}")
    
    # Print detailed obstacle information
    for i, (obs_pos, obs_size) in enumerate(zip(obstacle_positions, obstacle_sizes)):
        print(f"  Obstacle {i+1}:")
        print(f"    Position: [{obs_pos[0]:.3f}, {obs_pos[1]:.3f}, {obs_pos[2]:.3f}]")
        print(f"    Size: height={obs_size['height']:.3f}m, radius={obs_size['radius']:.3f}m")
        print(f"    Orientation: {obs_size['orientation']}")
    
    print(f"  Target at: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # NEW: Print path metrics info
    if path_metrics:
        print(f"  Path metrics available:")
        if path_metrics.get('total_distance_traveled') is not None:
            print(f"    Total distance: {path_metrics['total_distance_traveled']:.3f}m")
        if path_metrics.get('straight_line_distance') is not None:
            print(f"    Straight-line distance: {path_metrics['straight_line_distance']:.3f}m")
        if path_metrics.get('path_efficiency_ratio') is not None:
            print(f"    Path efficiency ratio: {path_metrics['path_efficiency_ratio']:.2f}")
    
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
        
        # Add start position marker
        start_pos = trajectory[0]
        fig.add_trace(go.Scatter3d(
            x=[start_pos[0]],
            y=[start_pos[1]],
            z=[start_pos[2]],
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
        
        # Add final position marker
        final_pos = trajectory[-1]
        fig.add_trace(go.Scatter3d(
            x=[final_pos[0]],
            y=[final_pos[1]],
            z=[final_pos[2]],
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
        
        # NEW: Add straight-line distance from start to target 
        straight_line_distance = path_metrics.get('straight_line_distance', 'N/A')
        fig.add_trace(go.Scatter3d(
            x=[start_pos[0], target_pos[0]],
            y=[start_pos[1], target_pos[1]],
            z=[start_pos[2], target_pos[2]],
            mode='lines',
            name=f'Straight Line (d={straight_line_distance:.3f}m)' if isinstance(straight_line_distance, (int, float)) else 'Straight Line',
            line=dict(
                color='black',  
                width=6,           # Thin
                dash='dot'         # Dotted
            ),
            hovertemplate='<b>Straight-Line Distance</b><br>' +
                         'Distance: %{text}<br>' +
                         'Start: [%.3f, %.3f, %.3f]<br>' % tuple(start_pos) +
                         'Target: [%.3f, %.3f, %.3f]<br>' % tuple(target_pos) +
                         '<extra></extra>',
            text=f'{straight_line_distance:.3f}m' if isinstance(straight_line_distance, (int, float)) else 'N/A'
        ))
        
    else:
        print("  Warning: No real trajectory data available - showing positions only")
        
        # Show approximate start position (typical Franka home)
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
        
        # NEW: Add approximate straight-line from approximate start to target
        fig.add_trace(go.Scatter3d(
            x=[approx_start[0], target_pos[0]],
            y=[approx_start[1], target_pos[1]],
            z=[approx_start[2], target_pos[2]],
            mode='lines',
            name='Approx Straight Line',
            line=dict(
                color='lightgray',  # Light colored
                width=2,           # Thin
                dash='dot'         # Dotted
            ),
            hovertemplate='<b>Approximate Straight-Line Distance</b><br>' +
                         'From approx start to target<br>' +
                         'Start: [%.3f, %.3f, %.3f]<br>' % tuple(approx_start) +
                         'Target: [%.3f, %.3f, %.3f]<br>' % tuple(target_pos) +
                         '<extra></extra>'
        ))
    
    # Add target position marker
    fig.add_trace(go.Scatter3d(
        x=[target_pos[0]],
        y=[target_pos[1]],
        z=[target_pos[2]],
        mode='markers',
        name='Target Position',
        marker=dict(size=8, color='red', symbol='circle',
                   line=dict(width=2, color='darkred')),
        hovertemplate='<b>Target Position</b><br>' +
                     'X: %{x:.3f}m<br>' +
                     'Y: %{y:.3f}m<br>' +
                     'Z: %{z:.3f}m<br>' +
                     '<extra></extra>'
    ))
    
    # UPDATED: Add multiple cylinder obstacles with EXACT geometry from evaluation manager
    obstacle_colors = ['Reds', 'Oranges', 'YlOrRd', 'OrRd', 'Greys']
    
    for i, (obstacle_pos, obstacle_size) in enumerate(zip(obstacle_positions, obstacle_sizes)):
        # Get exact dimensions and orientation from evaluation manager
        height = obstacle_size['height']
        radius = obstacle_size['radius']
        orientation = obstacle_size['orientation']
        
        # Create cylinder obstacle using EXACT dimensions and orientation
        x_cyl, y_cyl, z_cyl = create_cylinder_mesh_with_orientation(
            obstacle_pos, radius, height, orientation
        )
        
        # Use different color for each obstacle
        color_scale = obstacle_colors[i % len(obstacle_colors)]
        
        # Add cylinder surface with exact geometry
        fig.add_trace(go.Surface(
            x=x_cyl,
            y=y_cyl,
            z=z_cyl,
            name=f'Obstacle {i+1} Cylinder',
            opacity=0.7,
            colorscale=color_scale,
            showscale=False,
            hovertemplate=f'<b>Obstacle {i+1} Cylinder</b><br>' +
                         'Center: [%.3f, %.3f, %.3f]<br>' % tuple(obstacle_pos) +
                         'Radius: %.3fm<br>' % radius +
                         'Height: %.3fm<br>' % height +
                         'Orientation: %s<br>' % orientation +
                         '<extra></extra>'
        ))
        
        # Add cylinder caps with correct orientation
        caps = create_cylinder_caps_with_orientation(obstacle_pos, radius, height, orientation)
        
        for j, (x_cap, y_cap, z_cap) in enumerate(caps):
            fig.add_trace(go.Surface(
                x=x_cap, y=y_cap, z=z_cap,
                name=f'Obstacle {i+1} Cap {j+1}',
                opacity=0.7,
                colorscale=color_scale,
                showscale=False,
                showlegend=False,
                hovertemplate=f'<b>Obstacle {i+1} End Cap</b><br>' +
                             'Orientation: %s<br>' % orientation +
                             '<extra></extra>'
            ))
    
    # Create title with simulation info including exact geometry info
    sim_number = simulation_key.split('_')[1]
    title_text = f'3D End-Effector Trajectory - Simulation {sim_number} (Exact Geometry)'
    subtitle_parts = []
    
    # Add run folder info to subtitle
    subtitle_parts.append(f'Run: {RUN_FOLDER_NAME}')
    
    # Add obstacle count and target info
    subtitle_parts.append(f'Obstacles: {num_obstacles}')
    subtitle_parts.append(f'Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]')
    
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
    else:
        # Add obstacle details annotation box with exact geometry info AND path metrics
        obstacle_text = f"Obstacle Details ({num_obstacles} total):<br>"
        for i, (pos, size) in enumerate(zip(obstacle_positions, obstacle_sizes)):
            obstacle_text += f"#{i+1}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]<br>"
            obstacle_text += f"    h={size['height']:.2f}m, r={size['radius']:.2f}m<br>"
            obstacle_text += f"    {size['orientation'].replace('_', ' ')}<br>"
        
        # NEW: Add path metrics to annotation
        if path_metrics:
            obstacle_text += f"<br>Path Metrics:<br>"
            if path_metrics.get('total_distance_traveled') is not None:
                obstacle_text += f"Total distance: {path_metrics['total_distance_traveled']:.3f}m<br>"
            if path_metrics.get('straight_line_distance') is not None:
                obstacle_text += f"Straight line: {path_metrics['straight_line_distance']:.3f}m<br>"
            if path_metrics.get('path_efficiency_ratio') is not None:
                obstacle_text += f"Efficiency ratio: {path_metrics['path_efficiency_ratio']:.2f}<br>"
        
        layout_dict['annotations'] = [{
            'xref': "paper", 'yref': "paper",
            'x': 0.02, 'y': 0.98,  # Top-left corner
            'text': obstacle_text,
            'showarrow': False,
            'font': dict(size=10, color="black"),
            'bgcolor': "rgba(255, 255, 255, 0.9)",
            'bordercolor': "gray",
            'borderwidth': 1,
            'align': "left"
        }]
    
    fig.update_layout(layout_dict)
    
    # Determine save directory
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(json_filename)) if json_filename else os.getcwd()
        print(f"Auto-detected save directory: {save_dir}")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamped filenames with obstacle count and geometry info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"3d_trajectory_{simulation_key}_{num_obstacles}obs_exact_geometry_{timestamp}"
    
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
    
    # Print summary with exact extracted geometry
    print(f"\n3D Plot Summary for {simulation_key} (EXACT GEOMETRY):")
    print(f"   Number of obstacles: {num_obstacles}")
    
    for i, (obs_pos, obs_size) in enumerate(zip(obstacle_positions, obstacle_sizes)):
        print(f"   Obstacle {i+1}:")
        print(f"     Position: [{obs_pos[0]:.3f}, {obs_pos[1]:.3f}, {obs_pos[2]:.3f}]")
        print(f"     Exact size: height={obs_size['height']:.3f}m, radius={obs_size['radius']:.3f}m")
        print(f"     Exact orientation: {obs_size['orientation']}")
    
    print(f"   Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
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
        
        # NEW: Print straight-line distance information
        if path_metrics.get('straight_line_distance') is not None:
            print(f"   Straight-line distance (start to target): {path_metrics['straight_line_distance']:.3f}m")
            if path_metrics.get('path_efficiency_ratio') is not None:
                print(f"   Path efficiency ratio: {path_metrics['path_efficiency_ratio']:.2f}")
        else:
            # Calculate it manually if not in path_metrics
            straight_dist = np.linalg.norm(np.array(target_pos) - trajectory[0])
            print(f"   Straight-line distance (calculated): {straight_dist:.3f}m")
            if total_distance > 0:
                efficiency = total_distance / straight_dist
                print(f"   Path efficiency ratio (calculated): {efficiency:.2f}")
        
        # Distance from final position to target
        final_to_target = np.linalg.norm(trajectory[-1] - np.array(target_pos))
        print(f"   Final distance to target: {final_to_target:.3f}m")
        
        # Calculate minimum distance to each obstacle during trajectory (center-to-center)
        if obstacle_positions:
            print(f"   Minimum center-to-center distances to obstacles during trajectory:")
            for i, obs_pos in enumerate(obstacle_positions):
                min_dist_to_obs = float('inf')
                for traj_point in trajectory:
                    dist = np.linalg.norm(traj_point - np.array(obs_pos))
                    min_dist_to_obs = min(min_dist_to_obs, dist)
                print(f"     To obstacle {i+1}: {min_dist_to_obs:.3f}m (center-to-center)")
    else:
        print("   No real trajectory data available")
        print("   To get actual trajectories, modify evaluation_manager_rmp.py")
        print("   Showing approximate straight-line distance")
    
    # Show the plot
    print(f"\nOpening 3D trajectory plot with EXACT geometry and straight-line distance in browser...")
    fig.show()
    
    return fig

def plot_3d_trajectories_from_file(json_filename, simulation_numbers=None, save_dir=None):
    """
    Load JSON file and create 3D trajectory plots with exact geometry
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
    
    # Show obstacle count and geometry distribution
    obstacle_counts = {}
    orientation_counts = {}
    size_ranges = {'height': [], 'radius': []}
    
    for sim_key in available_sims:
        num_obs = data[sim_key].get('num_obstacles', 'Unknown')
        obstacle_sizes = data[sim_key].get('obstacle_sizes', [])
        
        if num_obs != 'Unknown':
            obstacle_counts[num_obs] = obstacle_counts.get(num_obs, 0) + 1
        
        # Analyze orientations and sizes
        for obs_size in obstacle_sizes:
            orientation = obs_size.get('orientation', 'Unknown')
            orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1
            
            if 'height' in obs_size and 'radius' in obs_size:
                size_ranges['height'].append(obs_size['height'])
                size_ranges['radius'].append(obs_size['radius'])
    
    if obstacle_counts:
        print("Obstacle count distribution:")
        for count, freq in sorted(obstacle_counts.items()):
            print(f"  {count} obstacles: {freq} simulation(s)")
    
    if orientation_counts:
        print("Orientation distribution:")
        for orientation, freq in sorted(orientation_counts.items()):
            print(f"  {orientation}: {freq} cylinder(s)")
    
    if size_ranges['height']:
        print("Size ranges:")
        print(f"  Height: {min(size_ranges['height']):.3f}m - {max(size_ranges['height']):.3f}m")
        print(f"  Radius: {min(size_ranges['radius']):.3f}m - {max(size_ranges['radius']):.3f}m")
    
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
    
    print(f"Creating 3D plots with EXACT geometry and straight-line distances for {len(sims_to_plot)} simulation(s): {', '.join(sims_to_plot)}")
    
    # Plot each simulation separately
    figures = []
    for sim_key in sims_to_plot:
        print(f"\n" + "="*50)
        fig = plot_3d_simulation(data, sim_key, save_dir, json_filename)
        if fig:
            figures.append((sim_key, fig))
    
    print(f"\nSuccessfully created {len(figures)} 3D trajectory plots with EXACT geometry and straight-line distances!")
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
    print("3D TRAJECTORY PLOTTING TOOL - EXACT GEOMETRY VERSION WITH STRAIGHT-LINE DISTANCE")
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
        print(f"\nConfiguration: Plot ALL simulations with EXACT geometry and straight-line distances")
    else:
        simulation_numbers = SPECIFIC_SIMULATIONS
        print(f"\nConfiguration: Plot specific simulations with EXACT geometry and straight-line distances: {SPECIFIC_SIMULATIONS}")
    
    # Create plots
    print(f"\nProcessing 3D trajectory data with EXACT geometry and straight-line distances from: {json_file_path}")
    figures = plot_3d_trajectories_from_file(json_file_path, simulation_numbers, run_dir)
    
    if figures is None:
        print(f"Failed to create plots")
    else:
        print(f"\nSuccessfully created {len(figures)} 3D trajectory plots with EXACT geometry and straight-line distances!")
        print(f"All plots saved to: {run_dir}")

def show_usage():
    """Show usage information"""
    print("3D Trajectory Plotting Script - Exact Geometry Version with Straight-Line Distance")
    print("=" * 60)
    print("To use this script:")
    print("1. Edit the configuration variables at the top of this file:")
    print(f"   - RUN_FOLDER_NAME (currently: '{RUN_FOLDER_NAME}')")
    print(f"   - JSON_FILENAME (currently: '{JSON_FILENAME}')")
    print(f"   - PLOT_ALL_SIMULATIONS (currently: {PLOT_ALL_SIMULATIONS})")
    print(f"   - SPECIFIC_SIMULATIONS (currently: {SPECIFIC_SIMULATIONS})")
    print()
    print("2. Run the script:")
    print("   python3 3D_trajectory.py")
    print()
    print("3. Optional command line arguments:")
    print("   python3 3D_trajectory.py --list    # List available simulations")
    print("   python3 3D_trajectory.py --help    # Show this help")
    print()
    print("This version uses EXACT cylinder geometry (size & orientation) from evaluation manager")
    print("and includes straight-line distance visualization from start to target.")
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
                print(f"Error: JSON file not found: {json_file_path}")
            sys.exit(0)
    
    # Run main function
    main()