import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import time
import numpy as np
import json
from messages_fr3.msg import ClosestPoint
from geometry_msgs.msg import PoseStamped

# ==================== MANUAL CONFIGURATION ====================
# Edit these variables to specify which simulation to recreate:

# JSON file configuration
JSON_FILE_PATH = "/home/matteo/Simulation_rmp/Run_20250814_145854/evaluation_results_with_distances.json"  # Path to the JSON file with simulation data

# Simulation number to recreate
SIMULATION_NUMBER = 1  # Change this to the simulation you want to recreate

# =============================================================


class SingleSimulationEvaluationManager(Node):
    def __init__(self):
        super().__init__('single_simulation_evaluation_manager')
        self.pose_pub = self.create_publisher(Pose, '/riemannian_motion_policy/reference_pose', 10)
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        
        # Subscribe to end-effector pose from distance calculator (for goal monitoring only)
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/end_effector_pose',
            self.ee_pose_callback,
            10
        )
        self.current_ee_pose = None
        
        # Subscribe to distance data for safety monitoring (optional)
        self.distance_sub = self.create_subscription(
            ClosestPoint,
            '/closest_point',
            self.distance_callback,
            10
        )
        
        # Load simulation data from JSON
        self.simulation_data = self.load_simulation_from_json()
        
        # Wait for dependencies and run simulation
        self.wait_for_dependencies()
        self.run_recreation_simulation()

    def load_simulation_from_json(self):
        """Load specific simulation data from JSON file"""
        print(f"\n📁 Loading simulation {SIMULATION_NUMBER} from: {JSON_FILE_PATH}")
        
        # Check if file exists
        if not os.path.exists(JSON_FILE_PATH):
            raise FileNotFoundError(f"JSON file not found: {JSON_FILE_PATH}")
        
        # Load JSON data
        try:
            with open(JSON_FILE_PATH, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise Exception(f"Error reading JSON file: {e}")
        
        # Find the specific simulation
        simulation_key = f'simulation_{SIMULATION_NUMBER}'
        
        if simulation_key not in data:
            available_sims = [key.replace('simulation_', '') for key in data.keys() if key.startswith('simulation_')]
            raise KeyError(f"Simulation {SIMULATION_NUMBER} not found! Available: {sorted(available_sims)}")
        
        sim_data = data[simulation_key]
        
        # Extract obstacle information
        obstacle_positions = sim_data.get('obstacle_positions', [])
        if not obstacle_positions:
            # Fallback to old single obstacle format
            single_obstacle = sim_data.get('obstacle_position', [])
            if single_obstacle:
                obstacle_positions = [single_obstacle]
        
        obstacle_sizes = sim_data.get('obstacle_sizes', [])
        target_pos = sim_data.get('target_position', [])
        
        # Validate essential data
        if not obstacle_positions:
            raise ValueError(f"No obstacle positions found in simulation {SIMULATION_NUMBER}")
        if not target_pos:
            raise ValueError(f"No target position found in simulation {SIMULATION_NUMBER}")
        
        print(f"✅ Loaded simulation {SIMULATION_NUMBER}:")
        print(f"   Obstacles: {len(obstacle_positions)}")
        print(f"   Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        return {
            'obstacle_positions': obstacle_positions,
            'obstacle_sizes': obstacle_sizes,
            'target_position': target_pos,
            'num_obstacles': len(obstacle_positions)
        }

    def ee_pose_callback(self, msg):
        """Store the latest end-effector pose"""
        self.current_ee_pose = msg

    def get_end_effector_position(self):
        """Get end-effector position from distance calculator"""
        if self.current_ee_pose is not None:
            pos = self.current_ee_pose.pose.position
            return np.array([pos.x, pos.y, pos.z])
        return None

    def distance_callback(self, msg):
        """Monitor distances for safety (minimal logging)"""
        # Calculate minimum distance for safety monitoring
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
        
        min_distances = []
        for link_name, coords in links_data.items():
            if len(coords['x']) > 0:
                for i in range(len(coords['x'])):
                    distance = np.sqrt(coords['x'][i]**2 + coords['y'][i]**2 + coords['z'][i]**2)
                    min_distances.append(distance)
        
        if min_distances:
            overall_min = min(min_distances)
            # Only log critical distances
            if overall_min < 0.05:  # Less than 5cm
                self.get_logger().warn(f"Close approach: {overall_min:.3f}m")

    def wait_for_dependencies(self):
        """Wait for required services and subscribers (same as evaluation_manager_rmp.py)"""
        self.get_logger().info("Waiting for dependencies...")
        
        while not self.count_subscribers('/planning_scene') > 0:
            time.sleep(1.0)
        while not self.count_subscribers('/riemannian_motion_policy/reference_pose') > 0:
            time.sleep(1.0)
        
        # Wait for distance calculator
        timeout = 10.0
        start_wait = time.time()
        while not self.count_publishers('/closest_point') > 0:
            if time.time() - start_wait > timeout:
                self.get_logger().warn("Distance calculator not detected")
                break
            time.sleep(0.5)
        
        time.sleep(2.0)
        self.get_logger().info("Dependencies ready.")

    def create_obstacle_orientation(self, orientation_type):
        """Create orientation quaternion based on orientation type (same as evaluation_manager_rmp.py)"""
        pose_orientation = Pose().orientation
        
        if orientation_type == 'vertical':
            pose_orientation.x = 0.0
            pose_orientation.y = 0.0
            pose_orientation.z = 0.0
            pose_orientation.w = 1.0
        elif orientation_type == 'horizontal_x':
            pose_orientation.x = 0.0
            pose_orientation.y = 0.707107
            pose_orientation.z = 0.0
            pose_orientation.w = 0.707107
        elif orientation_type == 'horizontal_y':
            pose_orientation.x = 0.707107
            pose_orientation.y = 0.0
            pose_orientation.z = 0.0
            pose_orientation.w = 0.707107
        else:
            pose_orientation.w = 1.0
            
        return pose_orientation

    def spawn_obstacles_from_data(self):
        """Spawn obstacles from loaded JSON data (similar to spawn_obstacles in evaluation_manager_rmp.py)"""
        self.clear_obstacles()
        
        scene = PlanningScene()
        scene.is_diff = True
        
        obstacle_positions = self.simulation_data['obstacle_positions']
        obstacle_sizes = self.simulation_data['obstacle_sizes']
        
        for i, obs_pos in enumerate(obstacle_positions):
            obj = CollisionObject()
            obj.header.frame_id = "fr3_link0"
            obj.header.stamp = self.get_clock().now().to_msg()
            obj.id = f"recreation_obstacle_{i}"
            
            cylinder = SolidPrimitive()
            cylinder.type = SolidPrimitive.CYLINDER
            
            # Get size information or use defaults
            if i < len(obstacle_sizes):
                size = obstacle_sizes[i]
                height = size.get('height', 0.4)
                radius = size.get('radius', 0.12)
                orientation = size.get('orientation', 'vertical')
            else:
                height = 0.4
                radius = 0.12
                orientation = 'vertical'
            
            cylinder.dimensions = [height, radius]
            
            # Create pose
            pose = Pose()
            pose.position.x = obs_pos[0]
            pose.position.y = obs_pos[1]
            pose.position.z = obs_pos[2]
            pose.orientation = self.create_obstacle_orientation(orientation)
            
            obj.primitives.append(cylinder)
            obj.primitive_poses.append(pose)
            obj.operation = CollisionObject.ADD
            
            scene.world.collision_objects.append(obj)
        
        # Publish scene multiple times
        for _ in range(5):
            self.planning_scene_pub.publish(scene)
            time.sleep(0.1)
        
        self.get_logger().info(f"Spawned {len(obstacle_positions)} obstacles from JSON data")

    def clear_obstacles(self):
        """Clear all obstacles (same as evaluation_manager_rmp.py)"""
        scene = PlanningScene()
        scene.is_diff = True
        
        for i in range(10):
            obj = CollisionObject()
            obj.header.frame_id = "fr3_link0"
            obj.header.stamp = self.get_clock().now().to_msg()
            obj.id = f"recreation_obstacle_{i}"
            obj.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(obj)
        
        for _ in range(3):
            self.planning_scene_pub.publish(scene)
            time.sleep(0.1)

    def set_target_pose_from_data(self):
        """Set target pose from loaded JSON data (similar to set_target_pose in evaluation_manager_rmp.py)"""
        target_pos = self.simulation_data['target_position']
        
        target = Pose()
        target.position.x = target_pos[0]
        target.position.y = target_pos[1]
        target.position.z = target_pos[2]
        target.orientation.w = 1.0
        
        for _ in range(5):
            self.pose_pub.publish(target)
            time.sleep(0.1)
        
        self.get_logger().info(f"Set target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

    def run_recreation_simulation(self):
        """Run the recreation simulation (simplified version of run_single_simulation)"""
        print(f"\n🚀 Starting recreation of simulation {SIMULATION_NUMBER}")
        
        # Spawn obstacles from JSON data
        self.spawn_obstacles_from_data()
        time.sleep(2.0)
        
        # Set target from JSON data
        self.set_target_pose_from_data()
        time.sleep(1.0)
        
        # Goal tracking setup
        target_pos = self.simulation_data['target_position']
        self.target_position = np.array(target_pos)
        self.goal_tolerance = 0.02  # 2 cm tolerance
        self.goal_reached = False
        self.goal_reach_time = None
        self.start_time = time.time()
        
        print(f"✅ Recreation setup complete")
        print(f"   Obstacles: {self.simulation_data['num_obstacles']}")
        print(f"   Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print("   Robot should start moving towards target...")
        print("   Press Ctrl+C to stop")
        
        # Monitor execution (similar to evaluation_manager_rmp.py)
        exec_time = 15.0  # Run for 15 seconds
        end_time = time.time() + exec_time
        
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if goal is reached
            if not self.goal_reached:
                current_pos = self.get_end_effector_position()
                if current_pos is not None:
                    dist = np.linalg.norm(current_pos - self.target_position)
                    
                    if dist <= self.goal_tolerance:
                        self.goal_reached = True
                        self.goal_reach_time = time.time() - self.start_time
                        print(f"\n🎯 GOAL REACHED!")
                        print(f"   Time: {self.goal_reach_time:.2f}s")
                        print(f"   Distance: {dist:.3f}m")
                        break
        
        if not self.goal_reached:
            print(f"\n⏱️ Simulation completed ({exec_time}s)")
            print("   Goal not reached within time limit")
        
        print("Recreation finished.")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SingleSimulationEvaluationManager()
        # Node runs simulation in constructor, so we just need to keep it alive
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Recreation stopped by user")
    except Exception as e:
        print(f"\n❌ Error during recreation: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()