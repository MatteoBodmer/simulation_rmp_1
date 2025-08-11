#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
#exec franka_ros2_ws/src/simulation_rmp/simulation_rmp/run_suite.sh

set -e

NUM_SIMULATIONS=4
PACKAGE_NAME="simulation_rmp"
LAUNCH_FILE="simulation_rmp_launch.py"

# Create a timestamp for the run folder to group all simulations in the same folder
RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
SIMULATION_RMP_DIR="/home/matteo/Simulation_rmp"
RUN_DIR="$SIMULATION_RMP_DIR/Run_${RUN_TIMESTAMP}"

# Create the run directory
mkdir -p "$RUN_DIR"

echo "Running simulations in folder: $RUN_DIR"

for ((i=1; i<=NUM_SIMULATIONS; i++))
do
    echo
    echo "=== Starting simulation $i/$NUM_SIMULATIONS ==="
    echo

    # Pass the run directory to the ROS2 node as an environment variable
    export RUN_DIR="$RUN_DIR"
    
    # Launch the ROS2 simulation
    ros2 launch "$PACKAGE_NAME" "$LAUNCH_FILE"

    echo
    echo "=== Simulation $i finished. Restarting Gazebo & RViz for next simulation ==="
    
    # Kill any existing Gazebo and RViz processes
    pkill -f 'ign gazebo' || true
    sleep 3
done

echo "All simulations completed successfully!"





