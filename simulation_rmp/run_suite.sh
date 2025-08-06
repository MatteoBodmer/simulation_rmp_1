#!/bin/bash

# Exit immediately if a command exits with a non-zero status.

NUM_SIMULATIONS=2
PACKAGE_NAME="simulation_rmp"
LAUNCH_FILE="simulation_rmp_launch.py"

for ((i=1; i<=NUM_SIMULATIONS; i++))
do
    echo
    echo "=== Starting simulation $i/$NUM_SIMULATIONS ==="
    echo
    ros2 launch "$PACKAGE_NAME" "$LAUNCH_FILE"
    echo
    echo "=== Simulation $i finished. Restarting Gazebo & RViz for next simulation ==="
    echo
    pkill -f 'ign gazebo' || true
    sleep 3
done

batch





