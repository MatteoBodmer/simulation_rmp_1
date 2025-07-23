import subprocess
import time

NUM_SIMULATIONS = 5
LAUNCH_CMD = ["ros2", "launch", "simulation_rmp", "simulation_rmp_launch.py"]

for i in range(NUM_SIMULATIONS):
    print(f"\n=== Starting simulation {i+1}/{NUM_SIMULATIONS} ===\n")
    proc = subprocess.Popen(LAUNCH_CMD)
    proc.wait()
    print(f"\n=== Simulation {i+1} finished. Restarting Gazebo & RViz for next simulation ===\n")
    time.sleep(3)
