import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
import random
import time
import json
import numpy as np
from datetime import datetime
from messages_fr3.msg import ClosestPoint  # Import your custom message type
from geometry_msgs.msg import PoseStamped


class EvaluationManager(Node):
    def __init__(self):
        super().__init__('evaluation_manager')
        self.pose_pub = self.create_publisher(Pose, '/riemannian_motion_policy/reference_pose', 10)
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        
        # Use run directory from environment variable if available, otherwise create new one
        if 'RUN_DIR' in os.environ and os.path.exists(os.environ['RUN_DIR']):
            self.run_dir = os.environ['RUN_DIR']
            self.get_logger().info(f"Using existing run directory from environment: {self.run_dir}")
        else:
            # Fallback: create timestamped run directory if no environment variable
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.simulation_rmp_dir = "/home/matteo/Simulation_rmp"
            self.run_dir = os.path.join(self.simulation_rmp_dir, f"Run_{self.run_timestamp}")
            os.makedirs(self.run_dir, exist_ok=True)
            self.get_logger().info(f"Created new run directory: {self.run_dir}")
        
        # JSON file path for this run
        self.results_file = os.path.join(self.run_dir, "evaluation_results_with_distances.json")
        
        # Subscribe to closest point distances
        self.distance_sub = self.create_subscription(
            ClosestPoint,
            '/closest_point',
            self.distance_callback,
            10
        )
        
        # Subscribe to end-effector pose from distance calculator
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/end_effector_pose',
            self.ee_pose_callback,
            10
        )
        self.current_ee_pose = None
        
        self.workspace_bounds = {'x': (0.2, 0.8), 'y': (-0.5, 0.5), 'z': (0.2, 0.8)}
        
        # Storage for distance data during simulation
        self.distance_data = []
        self.simulation_start_time = None
        self.is_recording = False
        
        self.wait_for_dependencies()
        self.get_logger().info("Starting single simulation...")
        result = self.run_single_simulation()
        self.save_results(result)
        self.get_logger().info("Simulation complete.")

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
        """Callback to log distance data during simulation"""
        if not self.is_recording:
            return
            
        current_time = time.time()
        relative_time = current_time - self.simulation_start_time if self.simulation_start_time else 0
        
        # Get current end-effector position for trajectory recording
        current_ee_pos = self.get_end_effector_position()
        
        # Calculate minimum distances for each link
        min_distances = {}
        
        # Process each link's repulsion data to find minimum distances
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
            if len(coords['x']) > 0:  # Check if there are obstacles near this link
                # Calculate distances from repulsion vectors (magnitude of repulsion direction)
                distances = []
                for i in range(len(coords['x'])):
                    distance = np.sqrt(coords['x'][i]**2 + coords['y'][i]**2 + coords['z'][i]**2)
                    distances.append(distance)
                min_distances[link_name] = min(distances) if distances else float('inf')
            else:
                min_distances[link_name] = float('inf')  # No obstacles detected
        
        # Find overall minimum distance
        overall_min = min(min_distances.values()) if min_distances else float('inf')
        
        distance_entry = {
            'timestamp': relative_time,
            'overall_min_distance': overall_min,
            'link_distances': min_distances,
            'num_obstacles_detected': len(msg.frame2x),  # Assuming all links detect same obstacles
            # CRITICAL ADDITION: Store end-effector position for trajectory plotting
            'end_effector_position': current_ee_pos.tolist() if current_ee_pos is not None else None
        }
        
        self.distance_data.append(distance_entry)
        
        # Log critical distances
        if overall_min < 0.1:  # Less than 10cm
            self.get_logger().warn(f"Close approach detected: {overall_min:.3f}m at t={relative_time:.2f}s")

    def wait_for_dependencies(self):
        self.get_logger().info("Waiting for dependencies...")
        while not self.count_subscribers('/planning_scene') > 0:
            time.sleep(1.0)
        while not self.count_subscribers('/riemannian_motion_policy/reference_pose') > 0:
            time.sleep(1.0)
        
        # Wait for distance calculator to be ready
        self.get_logger().info("Waiting for distance calculator...")
        timeout = 10.0
        start_wait = time.time()
        while not self.count_publishers('/closest_point') > 0:
            if time.time() - start_wait > timeout:
                self.get_logger().warn("Distance calculator not detected, continuing anyway...")
                break
            time.sleep(0.5)
        
        # Wait for end-effector pose publisher
        self.get_logger().info("Waiting for end-effector pose publisher...")
        start_wait = time.time()
        while not self.count_publishers('/end_effector_pose') > 0:
            if time.time() - start_wait > timeout:
                self.get_logger().warn("End-effector pose publisher not detected, continuing anyway...")
                break
            time.sleep(0.5)
            
        time.sleep(2.0)
        self.get_logger().info("Dependencies ready.")

    def is_point_inside_cylinder(self, point, cylinder_pose, height=0.4, radius=0.12):
        """Check if a point is inside the cylinder"""
        # Vector from cylinder center to point
        dx = point.position.x - cylinder_pose.position.x
        dy = point.position.y - cylinder_pose.position.y
        dz = point.position.z - cylinder_pose.position.z
        
        # Check if within cylinder height
        if abs(dz) > height / 2:
            return False
        
        # Check if within cylinder radius
        radial_distance = np.sqrt(dx**2 + dy**2)
        return radial_distance <= radius

    def geometric_collision_check(self, obstacle_pose):
        """Simplified geometric check - assumes robot base is at origin"""
        # Robot base position (fr3_link0)
        robot_base = np.array([0.0, 0.0, 0.0])
        
        # Approximate robot arm reach (simplified)
        max_reach = 0.8  # meters
        min_safe_distance = 0.3  # minimum safe distance from base
        
        obstacle_pos = np.array([obstacle_pose.position.x, obstacle_pose.position.y, obstacle_pose.position.z])
        distance_from_base = np.linalg.norm(obstacle_pos - robot_base)
        
        # Obstacle should be within reach but not too close to base
        return min_safe_distance < distance_from_base < max_reach

    def generate_random_obstacle_pose(self):
        """Generate obstacle pose ensuring it doesn't collide with robot"""
        for _ in range(100):  # Increased attempts
            pose = Pose()
            pose.position.x = random.uniform(*self.workspace_bounds['x'])
            pose.position.y = random.uniform(*self.workspace_bounds['y'])
            pose.position.z = random.uniform(*self.workspace_bounds['z'])
            pose.orientation.w = 1.0
            
            if self.geometric_collision_check(pose):
                return pose
        
        # Fallback to a known safe position
        self.get_logger().warn("Could not find safe obstacle pose, using fallback")
        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0.3
        pose.position.z = 0.4
        pose.orientation.w = 1.0
        return pose

    def generate_random_target_pose(self, obstacle_pose):
        """Generate target pose ensuring it's not inside the obstacle"""
        min_clearance = 0.15  # 15cm clearance from obstacle surface
        cylinder_height = 0.4
        cylinder_radius = 0.12
        
        for _ in range(100):  # Increased attempts
            target = Pose()
            target.position.x = random.uniform(*self.workspace_bounds['x'])
            target.position.y = random.uniform(*self.workspace_bounds['y'])
            target.position.z = random.uniform(*self.workspace_bounds['z'])
            target.orientation.w = 1.0
            
            # Check if target is inside cylinder
            if self.is_point_inside_cylinder(target, obstacle_pose, cylinder_height, cylinder_radius):
                continue
            
            # Calculate minimum distance to cylinder surface (not just center)
            dx = target.position.x - obstacle_pose.position.x
            dy = target.position.y - obstacle_pose.position.y
            dz = target.position.z - obstacle_pose.position.z
            
            # Distance to curved surface
            radial_distance = np.sqrt(dx**2 + dy**2)
            distance_to_curved_surface = abs(radial_distance - cylinder_radius)
            
            # Distance to top/bottom caps
            distance_to_caps = abs(abs(dz) - cylinder_height/2)
            
            # Use minimum distance to any surface
            min_surface_distance = min(distance_to_curved_surface, distance_to_caps)
            
            if min_surface_distance > min_clearance:
                return target
        
        # Fallback to a known safe position
        self.get_logger().warn("Could not find safe target pose, using fallback")
        target = Pose()
        target.position.x = 0.3
        target.position.y = 0.0
        target.position.z = 0.6
        target.orientation.w = 1.0
        return target

    def wait_and_check_initial_collision(self):
        """Wait for distance data and check if robot starts in collision"""
        self.get_logger().info("Checking initial robot safety...")
        
        # Subscribe temporarily to distance data
        initial_distances = None
        
        def temp_callback(msg):
            nonlocal initial_distances
            initial_distances = msg
        
        temp_sub = self.create_subscription(ClosestPoint, '/closest_point', temp_callback, 10)
        
        # Wait for distance data
        timeout = 5.0
        start_time = time.time()
        while initial_distances is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.destroy_subscription(temp_sub)
        
        if initial_distances is None:
            self.get_logger().warn("Could not get initial distance data")
            return False
        
        # Check if any link is too close to obstacles
        min_safe_distance = 0.1  # 10cm minimum
        
        links_data = {
            'link2': {'x': initial_distances.frame2x, 'y': initial_distances.frame2y, 'z': initial_distances.frame2z},
            'link3': {'x': initial_distances.frame3x, 'y': initial_distances.frame3y, 'z': initial_distances.frame3z},
            'link4': {'x': initial_distances.frame4x, 'y': initial_distances.frame4y, 'z': initial_distances.frame4z},
            'link5': {'x': initial_distances.frame5x, 'y': initial_distances.frame5y, 'z': initial_distances.frame5z},
            'link6': {'x': initial_distances.frame6x, 'y': initial_distances.frame6y, 'z': initial_distances.frame6z},
            'link7': {'x': initial_distances.frame7x, 'y': initial_distances.frame7y, 'z': initial_distances.frame7z},
            'hand': {'x': initial_distances.framehandx, 'y': initial_distances.framehandy, 'z': initial_distances.framehandz},
            'end_effector': {'x': initial_distances.frameeex, 'y': initial_distances.frameeey, 'z': initial_distances.frameeez}
        }
        
        for link_name, coords in links_data.items():
            if len(coords['x']) > 0:
                for i in range(len(coords['x'])):
                    distance = np.sqrt(coords['x'][i]**2 + coords['y'][i]**2 + coords['z'][i]**2)
                    if distance < min_safe_distance:
                        self.get_logger().warn(f"Initial collision risk: {link_name} too close ({distance:.3f}m)")
                        return False
        
        return True

    def spawn_obstacle(self, pose):
        self.clear_obstacles()
        obj = CollisionObject()
        obj.header.frame_id = "fr3_link0"
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = "evaluation_obstacle"
        cylinder = SolidPrimitive()
        cylinder.type = SolidPrimitive.CYLINDER
        cylinder.dimensions = [0.4, 0.12]
        obj.primitives.append(cylinder)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.world.collision_objects.append(obj)
        scene.is_diff = True
        for _ in range(5):
            self.planning_scene_pub.publish(scene)
            time.sleep(0.1)

    def clear_obstacles(self):
        obj = CollisionObject()
        obj.header.frame_id = "fr3_link0"
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = "evaluation_obstacle"
        obj.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.world.collision_objects.append(obj)
        scene.is_diff = True
        for _ in range(3):
            self.planning_scene_pub.publish(scene)
            time.sleep(0.1)

    def set_target_pose(self, pose):
        for _ in range(5):
            self.pose_pub.publish(pose)
            time.sleep(0.1)

    def start_recording(self):
        """Start recording distance data"""
        self.distance_data = []
        self.simulation_start_time = time.time()
        self.is_recording = True
        self.get_logger().info("Started recording distance data")

    def stop_recording(self):
        """Stop recording distance data"""
        self.is_recording = False
        self.get_logger().info(f"Stopped recording. Collected {len(self.distance_data)} distance measurements")

    def run_single_simulation(self):
        # Generate obstacle and target with safety checks
        max_attempts = 10
        for attempt in range(max_attempts):
            obstacle = self.generate_random_obstacle_pose()
            self.spawn_obstacle(obstacle)
            time.sleep(1.0)  # Allow time for obstacle to be added to scene
            
            # Check if robot is safe
            if self.wait_and_check_initial_collision():
                self.get_logger().info(f"Safe obstacle spawn achieved on attempt {attempt+1}")
                break
            else:
                self.get_logger().warn(f"Attempt {attempt+1}: Unsafe spawn, retrying...")
                self.clear_obstacles()
                time.sleep(1.0)
        else:
            self.get_logger().warn("Could not find safe obstacle spawn after maximum attempts")
        
        target = self.generate_random_target_pose(obstacle)
        time.sleep(2.0)
        
        # Start distance recording
        self.start_recording()
        
        # Goal tracking variables
        self.target_position = np.array([target.position.x, target.position.y, target.position.z])
        self.goal_tolerance = 0.02  # 2 cm tolerance
        self.goal_reached = False
        self.goal_reach_time = None
        self.position_check_count = 0
        self.successful_position_checks = 0
        
        # Set target and execute
        self.set_target_pose(target)
        self.get_logger().info(f"Target position: [{target.position.x:.3f}, {target.position.y:.3f}, {target.position.z:.3f}]")
        self.get_logger().info("Waiting briefly for robot state updates...")
        time.sleep(1.0)
        exec_time = 10.0
        self.get_logger().info(f"Monitoring for {exec_time} seconds.")
        
        # Keep spinning to receive distance messages and check goal
        end_time = time.time() + exec_time
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if goal is reached
            if not self.goal_reached:
                self.position_check_count += 1
                current_pos = self.get_end_effector_position()
                if current_pos is not None:
                    self.successful_position_checks += 1
                    dist = np.linalg.norm(current_pos - self.target_position)
                    
                    # Log position every 50 checks for debugging
                    if self.position_check_count % 50 == 0:
                        self.get_logger().info(f"Current pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}], distance to goal: {dist:.3f}m")
                    
                    if dist <= self.goal_tolerance:
                        self.goal_reached = True
                        self.goal_reach_time = time.time() - self.simulation_start_time
                        self.get_logger().info(f"Goal reached at t={self.goal_reach_time:.2f}s (distance {dist:.3f}m)")
        
        # Debug info
        self.get_logger().info(f"Position checks: {self.position_check_count}, successful: {self.successful_position_checks}")
        if self.successful_position_checks == 0:
            self.get_logger().warn("No successful position checks - end-effector pose not available!")
        
        # Stop recording
        self.stop_recording()
        
        # Analyze distance data
        analysis = self.analyze_distance_data()
        
        result = {
            'obstacle_position': [obstacle.position.x, obstacle.position.y, obstacle.position.z],
            'target_position': [target.position.x, target.position.y, target.position.z],
            'execution_time': exec_time,
            'timestamp': datetime.now().isoformat(),
            'distance_analysis': analysis,
            'raw_distance_data': self.distance_data[:100],  # Limit raw data to first 100 points
            # New goal tracking results
            'goal_reached': self.goal_reached,
            'goal_reach_time': self.goal_reach_time,
            'goal_tolerance': self.goal_tolerance,
            'position_check_count': self.position_check_count,
            'successful_position_checks': self.successful_position_checks,
            # Safety information
            'safe_spawn_attempts': attempt + 1 if 'attempt' in locals() else 1
        }
        return result

    def analyze_distance_data(self):
        """Analyze the collected distance data"""
        if not self.distance_data:
            return {'error': 'No distance data collected'}
        
        overall_distances = [entry['overall_min_distance'] for entry in self.distance_data if entry['overall_min_distance'] != float('inf')]
        
        if not overall_distances:
            return {'error': 'No valid distance measurements'}
        
        analysis = {
            'min_distance_achieved': min(overall_distances),
            'max_distance': max(overall_distances),
            'avg_distance': np.mean(overall_distances),
            'std_distance': np.std(overall_distances),
            'num_measurements': len(overall_distances),
            'close_calls': len([d for d in overall_distances if d < 0.05]),  # < 5cm
            'safety_violations': len([d for d in overall_distances if d < 0.02])  # < 2cm
        }
        
        # Per-link analysis
        link_analysis = {}
        for link_name in ['link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'end_effector']:
            link_distances = []
            for entry in self.distance_data:
                if link_name in entry['link_distances'] and entry['link_distances'][link_name] != float('inf'):
                    link_distances.append(entry['link_distances'][link_name])
            
            if link_distances:
                link_analysis[link_name] = {
                    'min_distance': min(link_distances),
                    'avg_distance': np.mean(link_distances)
                }
        
        analysis['per_link_analysis'] = link_analysis
        return analysis

    def save_results(self, result):
        # Load existing data if file exists
        existing_data = {}
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    existing_data = json.load(f)
                self.get_logger().info(f"Loaded existing data with {len(existing_data)} simulations")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.get_logger().warn(f"Could not load existing data: {e}, starting fresh")
                existing_data = {}
        
        # Find the next simulation number
        simulation_numbers = [int(k.replace('simulation_', '')) for k in existing_data.keys() if k.startswith('simulation_')]
        next_sim_num = max(simulation_numbers) + 1 if simulation_numbers else 1
        
        # Add new result with simulation number as key
        existing_data[f'simulation_{next_sim_num}'] = result
        
        # Save updated data
        with open(self.results_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        self.get_logger().info(f"Results saved to {self.results_file} as simulation_{next_sim_num}")
        
        # Print summary including goal achievement
        self.get_logger().info(f"Simulation {next_sim_num} - Goal Achievement Summary:")
        self.get_logger().info(f"  Goal reached: {result.get('goal_reached', False)}")
        if result.get('goal_reach_time') is not None:
            self.get_logger().info(f"  Time to reach goal: {result['goal_reach_time']:.2f}s")
        self.get_logger().info(f"  Goal tolerance: {result.get('goal_tolerance', 'N/A')}m")
        self.get_logger().info(f"  Position checks: {result.get('successful_position_checks', 0)}/{result.get('position_check_count', 0)}")
        
        # Print safety summary
        self.get_logger().info(f"Safety Summary:")
        self.get_logger().info(f"  Safe spawn attempts: {result.get('safe_spawn_attempts', 'N/A')}")
        
        # Print distance summary
        if 'distance_analysis' in result:
            analysis = result['distance_analysis']
            if 'min_distance_achieved' in analysis:
                self.get_logger().info(f"Distance Summary:")
                self.get_logger().info(f"  Min distance: {analysis['min_distance_achieved']:.3f}m")
                self.get_logger().info(f"  Avg distance: {analysis['avg_distance']:.3f}m")
                self.get_logger().info(f"  Close calls (<5cm): {analysis['close_calls']}")
                self.get_logger().info(f"  Safety violations (<2cm): {analysis['safety_violations']}")

def main(args=None):
    rclpy.init(args=args)
    node = EvaluationManager()
    node.destroy_node()
    rclpy.shutdown()
    print("Evaluation completed and node shut down.")
    
if __name__ == '__main__':
    main()

# test if we need tokens 