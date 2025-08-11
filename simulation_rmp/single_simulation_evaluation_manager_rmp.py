import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import time
import numpy as np
import json
from datetime import datetime
from messages_fr3.msg import ClosestPoint
from geometry_msgs.msg import PoseStamped


class ManualEvaluationManager(Node):
    def __init__(self):
        super().__init__('manual_evaluation_manager')
        self.pose_pub = self.create_publisher(Pose, '/riemannian_motion_policy/reference_pose', 10)
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        
        # Subscribe to end-effector pose from distance calculator (for monitoring only)
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/end_effector_pose',
            self.ee_pose_callback,
            10
        )
        self.current_ee_pose = None
        
        # Subscribe to distance data for safety monitoring
        self.distance_sub = self.create_subscription(
            ClosestPoint,
            '/closest_point',
            self.distance_callback,
            10
        )
        
        self.get_logger().info("Manual Evaluation Manager initialized")
        self.wait_for_dependencies()
        
        # Get user input and run simulation
        obstacle_pos, target_pos = self.get_simulation_from_json()
        self.run_manual_simulation(obstacle_pos, target_pos)
        
        # Keep node alive for observation
        self.get_logger().info("Simulation running. Press Ctrl+C to exit.")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=1.0)

    def ee_pose_callback(self, msg):
        """Store the latest end-effector pose from distance calculator"""
        self.current_ee_pose = msg

    def get_end_effector_position(self):
        """Get end-effector position from distance calculator"""
        if self.current_ee_pose is not None:
            pos = self.current_ee_pose.pose.position
            return np.array([pos.x, pos.y, pos.z])
        else:
            return None

    def distance_callback(self, msg):
        """Callback to monitor distances during simulation (no logging)"""
        # Calculate minimum distances for safety monitoring
        min_distances = {}
        
        links_data = {
            'link2': {'x': msg.frame2x, 'y': msg.frame2y, 'z': msg.frame2z},
            'link3': {'x': msg.frame3x, 'y': msg.frame3y, 'z': msg.frame3z},
            'link4': {'x': msg.frame4x, 'y': msg.frame4y, 'z': msg.frame4z},
            'link5': {'x': msg.frame5x, 'y': msg.frame5y, 'z': msg.frame5z},
            'link6': {'x': msg.frame6x, 'y': msg.frame6y, 'z': msg.frame6z},
            'link7': {'x': msg.frame7x, 'y': msg.frame7y, 'z': msg.frame7z},
            'hand': {'x': msg.framehandx, 'y': msg.framehandy, 'z': msg.framehandz},
            'end_effector': {'x': msg.frameeex, 'y': msg.frameeey, 'z': msg.frameeez}
        }
        
        for link_name, coords in links_data.items():
            if len(coords['x']) > 0:
                distances = []
                for i in range(len(coords['x'])):
                    distance = np.sqrt(coords['x'][i]**2 + coords['y'][i]**2 + coords['z'][i]**2)
                    distances.append(distance)
                min_distances[link_name] = min(distances) if distances else float('inf')
            else:
                min_distances[link_name] = float('inf')
        
        # Find overall minimum distance
        overall_min = min(min_distances.values()) if min_distances else float('inf')
        
        # Log critical distances only
        if overall_min < 0.05:  # Less than 5cm - critical warning
            self.get_logger().warn(f"CRITICAL: Very close approach detected: {overall_min:.3f}m")

    def list_available_json_files(self, base_dir="/home/matteo/Simulation_rmp"):
        """List available JSON files in simulation directories"""
        available_files = []
        
        if not os.path.exists(base_dir):
            return available_files
            
        try:
            for item in os.listdir(base_dir):
                run_dir = os.path.join(base_dir, item)
                if os.path.isdir(run_dir) and item.startswith('Run_'):
                    json_file = os.path.join(run_dir, "evaluation_results_with_distances.json")
                    if os.path.exists(json_file):
                        available_files.append({
                            'run_folder': item,
                            'json_path': json_file,
                            'full_path': json_file
                        })
        except Exception as e:
            print(f"Error scanning directories: {e}")
            
        return available_files

    def list_simulations_in_json(self, json_path):
        """List all simulations in a JSON file with details"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            simulations = []
            for key in data.keys():
                if key.startswith('simulation_'):
                    sim_data = data[key]
                    sim_info = {
                        'key': key,
                        'number': int(key.replace('simulation_', '')),
                        'obstacle_pos': sim_data.get('obstacle_position', []),
                        'target_pos': sim_data.get('target_position', []),
                        'timestamp': sim_data.get('timestamp', 'Unknown'),
                        'goal_reached': sim_data.get('goal_reached', 'Unknown'),
                        'goal_time': sim_data.get('goal_reach_time', None)
                    }
                    simulations.append(sim_info)
                    
            # Sort by simulation number
            simulations.sort(key=lambda x: x['number'])
            return data, simulations
            
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return None, []

    def get_simulation_from_json(self):
        """Get obstacle and target positions from a specific simulation in JSON file"""
        
        print("\n" + "="*80)
        print("SIMULATION RECREATION FROM JSON FILE")
        print("="*80)
        print("This tool will load obstacle and target positions from a previous")
        print("simulation stored in the JSON evaluation results.")
        print("="*80)

        # Step 1: Find and select JSON file
        print("\n📁 STEP 1: Select JSON File")
        print("-" * 40)
        
        available_files = self.list_available_json_files()
        
        if not available_files:
            print("❌ No evaluation JSON files found in /home/matteo/Simulation_rmp/")
            print("Make sure you have run some simulations first with evaluation_manager_rmp.py")
            exit(1)
            
        print("Available JSON files:")
        for i, file_info in enumerate(available_files, 1):
            print(f"  {i}. {file_info['run_folder']}")
            print(f"     Path: {file_info['json_path']}")
        
        # Get JSON file selection
        while True:
            try:
                print(f"\nYou can also specify a custom JSON file path.")
                selection = input(f"Select JSON file (1-{len(available_files)}) or enter full path: ").strip()
                
                if selection.isdigit():
                    idx = int(selection) - 1
                    if 0 <= idx < len(available_files):
                        selected_json = available_files[idx]['json_path']
                        break
                    else:
                        print(f"❌ Invalid selection. Choose 1-{len(available_files)}")
                else:
                    # Custom path provided
                    if os.path.exists(selection):
                        selected_json = selection
                        break
                    else:
                        print(f"❌ File not found: {selection}")
                        
            except ValueError:
                print("❌ Invalid input. Enter a number or file path.")
        
        print(f"✅ Selected: {selected_json}")

        # Step 2: Load and display simulations
        print(f"\n🔍 STEP 2: Select Simulation")
        print("-" * 40)
        
        data, simulations = self.list_simulations_in_json(selected_json)
        
        if not data or not simulations:
            print("❌ No simulations found in the selected JSON file!")
            exit(1)
            
        print(f"Found {len(simulations)} simulations:")
        print()
        
        for sim in simulations:
            print(f"  Simulation {sim['number']}:")
            print(f"    📅 Time: {sim['timestamp'][:19] if sim['timestamp'] != 'Unknown' else 'Unknown'}")
            print(f"    🎯 Goal reached: {sim['goal_reached']}")
            if sim['goal_time'] is not None:
                print(f"    ⏱️  Goal time: {sim['goal_time']:.2f}s")
            print(f"    🚧 Obstacle: [{sim['obstacle_pos'][0]:.3f}, {sim['obstacle_pos'][1]:.3f}, {sim['obstacle_pos'][2]:.3f}]")
            print(f"    🎯 Target:   [{sim['target_pos'][0]:.3f}, {sim['target_pos'][1]:.3f}, {sim['target_pos'][2]:.3f}]")
            
            # Calculate distance between obstacle and target
            if len(sim['obstacle_pos']) == 3 and len(sim['target_pos']) == 3:
                dist = np.linalg.norm(np.array(sim['target_pos']) - np.array(sim['obstacle_pos']))
                print(f"    📏 Distance: {dist:.3f}m")
            print()

        # Step 3: Select specific simulation
        print(f"\n🎮 STEP 3: Choose Simulation to Recreate")
        print("-" * 50)
        
        available_numbers = [sim['number'] for sim in simulations]
        
        while True:
            try:
                sim_number = input(f"Enter simulation number ({available_numbers[0]}-{available_numbers[-1]}): ").strip()
                sim_number = int(sim_number)
                
                if sim_number in available_numbers:
                    break
                else:
                    print(f"❌ Invalid simulation number. Available: {available_numbers}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Find the selected simulation
        selected_sim = None
        for sim in simulations:
            if sim['number'] == sim_number:
                selected_sim = sim
                break
                
        if selected_sim is None:
            print("❌ Simulation not found!")
            exit(1)

        # Step 4: Confirm and extract positions
        print(f"\n✅ SIMULATION {sim_number} SELECTED")
        print("=" * 80)
        print("RECREATION CONFIGURATION:")
        print(f"  📅 Original timestamp: {selected_sim['timestamp'][:19] if selected_sim['timestamp'] != 'Unknown' else 'Unknown'}")
        print(f"  🎯 Original goal reached: {selected_sim['goal_reached']}")
        if selected_sim['goal_time'] is not None:
            print(f"  ⏱️  Original goal time: {selected_sim['goal_time']:.2f}s")
        print(f"  🚧 Obstacle position: [{selected_sim['obstacle_pos'][0]:.3f}, {selected_sim['obstacle_pos'][1]:.3f}, {selected_sim['obstacle_pos'][2]:.3f}]")
        print(f"  🎯 Target position:   [{selected_sim['target_pos'][0]:.3f}, {selected_sim['target_pos'][1]:.3f}, {selected_sim['target_pos'][2]:.3f}]")
        
        # Calculate and show distance
        obstacle_array = np.array(selected_sim['obstacle_pos'])
        target_array = np.array(selected_sim['target_pos'])
        distance = np.linalg.norm(target_array - obstacle_array)
        print(f"  📏 Obstacle-Target distance: {distance:.3f}m")
        print(f"  🔧 Cylinder: height=0.4m, radius=0.12m")
        
        # Safety check
        min_safe_distance = 0.15
        if distance < min_safe_distance:
            print(f"  ⚠️  WARNING: Very challenging configuration (distance < {min_safe_distance}m)")
            
        print("=" * 80)
        
        confirm = input(f"\n🚀 Proceed with recreating Simulation {sim_number}? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Recreation cancelled. Exiting...")
            exit(0)

        print(f"✅ Starting recreation of Simulation {sim_number}...")
        
        return selected_sim['obstacle_pos'], selected_sim['target_pos']

    def wait_for_dependencies(self):
        """Wait for required services and subscribers"""
        self.get_logger().info("Waiting for dependencies...")
        
        # Wait for planning scene subscribers
        while not self.count_subscribers('/planning_scene') > 0:
            self.get_logger().info("Waiting for planning scene subscribers...")
            time.sleep(1.0)
        
        # Wait for RMP subscribers
        while not self.count_subscribers('/riemannian_motion_policy/reference_pose') > 0:
            self.get_logger().info("Waiting for RMP subscribers...")
            time.sleep(1.0)
        
        # Wait for distance calculator (optional for manual mode)
        self.get_logger().info("Waiting for distance calculator...")
        timeout = 5.0
        start_wait = time.time()
        while not self.count_publishers('/closest_point') > 0:
            if time.time() - start_wait > timeout:
                self.get_logger().warn("Distance calculator not detected - safety monitoring disabled")
                break
            time.sleep(0.5)
        
        time.sleep(2.0)
        self.get_logger().info("Dependencies ready.")

    def spawn_obstacle(self, position):
        """Spawn obstacle at specified position"""
        self.clear_obstacles()
        
        obj = CollisionObject()
        obj.header.frame_id = "fr3_link0"
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = "recreation_obstacle"
        
        cylinder = SolidPrimitive()
        cylinder.type = SolidPrimitive.CYLINDER
        cylinder.dimensions = [0.4, 0.12]  # [height, radius] - same as evaluation_manager_rmp.py
        
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.w = 1.0
        
        obj.primitives.append(cylinder)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        
        scene = PlanningScene()
        scene.world.collision_objects.append(obj)
        scene.is_diff = True
        
        # Publish multiple times to ensure it's received
        for i in range(10):
            self.planning_scene_pub.publish(scene)
            time.sleep(0.1)
        
        self.get_logger().info(f"✅ Obstacle recreated at [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")

    def clear_obstacles(self):
        """Clear all obstacles from the scene"""
        obj = CollisionObject()
        obj.header.frame_id = "fr3_link0"
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = "recreation_obstacle"
        obj.operation = CollisionObject.REMOVE
        
        scene = PlanningScene()
        scene.world.collision_objects.append(obj)
        scene.is_diff = True
        
        for _ in range(5):
            self.planning_scene_pub.publish(scene)
            time.sleep(0.1)

    def set_target_pose(self, position):
        """Set target pose for the robot"""
        target = Pose()
        target.position.x = position[0]
        target.position.y = position[1]
        target.position.z = position[2]
        target.orientation.w = 1.0  # Same orientation as evaluation_manager_rmp.py
        
        # Publish multiple times to ensure it's received
        for i in range(10):
            self.pose_pub.publish(target)
            time.sleep(0.1)
        
        self.get_logger().info(f"✅ Target recreated at [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")

    def run_manual_simulation(self, obstacle_pos, target_pos):
        """Run the manual simulation with specified poses"""
        
        print("\n" + "="*80)
        print("🚀 STARTING SIMULATION RECREATION")
        print("="*80)
        
        # Step 1: Spawn obstacle
        self.get_logger().info("Step 1: Recreating obstacle...")
        self.spawn_obstacle(obstacle_pos)
        time.sleep(2.0)  # Allow time for obstacle to be added to planning scene
        
        # Step 2: Set target pose
        self.get_logger().info("Step 2: Setting target...")
        self.set_target_pose(target_pos)
        time.sleep(1.0)
        
        # Step 3: Start monitoring
        self.get_logger().info("Step 3: Recreation active - robot attempting to reach target")
        
        print(f"\n🤖 SIMULATION RECREATION ACTIVE")
        print(f"   🚧 Obstacle: [{obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}, {obstacle_pos[2]:.3f}]")
        print(f"   🎯 Target:   [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"   📺 Watch RViz to see the robot motion")
        print(f"   ⏹️  Press Ctrl+C to stop simulation")
        print(f"   🔍 Compare with original results!")
        print("="*80)
        
        # Goal tracking variables (same as evaluation_manager_rmp.py)
        self.target_position = np.array(target_pos)
        self.goal_tolerance = 0.02  # 2 cm tolerance
        self.goal_reached = False
        self.goal_reach_time = None
        self.start_time = time.time()
        
        # Monitor goal achievement
        last_status_time = time.time()
        status_interval = 5.0  # Print status every 5 seconds
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            current_time = time.time()
            
            # Check if goal is reached
            if not self.goal_reached:
                current_pos = self.get_end_effector_position()
                if current_pos is not None:
                    dist = np.linalg.norm(current_pos - self.target_position)
                    
                    if dist <= self.goal_tolerance:
                        self.goal_reached = True
                        self.goal_reach_time = current_time - self.start_time
                        print(f"\n🎯 GOAL REACHED IN RECREATION!")
                        print(f"   Time: {self.goal_reach_time:.2f}s")
                        print(f"   Final distance: {dist:.3f}m")
                        self.get_logger().info(f"🎯 Goal reached in recreation: {self.goal_reach_time:.2f}s (distance: {dist:.3f}m)")
            
            # Print periodic status
            if current_time - last_status_time > status_interval:
                elapsed = current_time - self.start_time
                current_pos = self.get_end_effector_position()
                if current_pos is not None:
                    dist_to_goal = np.linalg.norm(current_pos - self.target_position)
                    status = "🎯 REACHED" if self.goal_reached else f"🔄 {dist_to_goal:.3f}m to goal"
                    print(f"⏱️  {elapsed:.0f}s elapsed | {status} | Current: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                last_status_time = current_time

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ManualEvaluationManager()
    except KeyboardInterrupt:
        print("\n🛑 Simulation recreation stopped by user")
    except Exception as e:
        print(f"\n❌ Error during recreation: {e}")
    finally:
        rclpy.shutdown()
        print("Simulation recreation ended.")

if __name__ == '__main__':
    main()