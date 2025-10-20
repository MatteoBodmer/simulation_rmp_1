#!/bin/bash

# set -e


#exec franka_ros2_ws/src/simulation_rmp_1/simulation_rmp_1/run_suite_cartesian.sh

NUM_SIMULATIONS=5
PACKAGE_NAME="simulation_rmp_1"                 # <-- change if your package name differs
LAUNCH_FILE="simulation_cartesian_launch.py"

RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
SIMULATION_CARTESIAN_DIR="/home/matteo/Simulation_cartesian"
RUN_DIR="$SIMULATION_CARTESIAN_DIR/Run_${RUN_TIMESTAMP}"

mkdir -p "$RUN_DIR"
echo "Running Cartesian simulations in folder: $RUN_DIR"

for ((i=1; i<=NUM_SIMULATIONS; i++))
do
    echo
    echo "=== Starting Cartesian simulation $i/$NUM_SIMULATIONS ==="
    echo

    export RUN_DIR="$RUN_DIR"

    timeout 60s ros2 launch "$PACKAGE_NAME" "$LAUNCH_FILE" \
        load_gripper:=true \
        franka_hand:='franka_hand'

    echo
    echo "=== Simulation $i finished. Restarting Gazebo & RViz for next simulation ==="

    pkill -f 'ign gazebo' || true
    sleep 3
done

echo "All Cartesian simulations completed successfully!"
