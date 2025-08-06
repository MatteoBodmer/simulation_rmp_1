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

class EvaluationManager(Node):
    def __init__(self):
        super().__init__('evaluation_manager')
        self.pose_pub = self.create_publisher(Pose, '/riemannian_motion_policy/reference_pose', 10)
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        
        # Subscribe to closest point distances
        self.distance_sub = self.create_subscription(
            ClosestPoint,
            '/closest_point',
            self.distance_callback,
            10
        )
        
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

    def distance_callback(self, msg):
        """Callback to log distance data during simulation"""
        if not self.is_recording:
            return
            
        current_time = time.time()
        relative_time = current_time - self.simulation_start_time if self.simulation_start_time else 0
        
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
            'num_obstacles_detected': len(msg.frame2x)  # Assuming all links detect same obstacles
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
            
        time.sleep(2.0)
        self.get_logger().info("Dependencies ready.")

    def generate_random_obstacle_pose(self):
        pose = Pose()
        pose.position.x = random.uniform(*self.workspace_bounds['x'])
        pose.position.y = random.uniform(*self.workspace_bounds['y'])
        pose.position.z = random.uniform(*self.workspace_bounds['z'])
        pose.orientation.w = 1.0
        return pose

    def generate_random_target_pose(self, obstacle_pose):
        min_distance = 0.15
        for _ in range(50):
            target = Pose()
            target.position.x = random.uniform(*self.workspace_bounds['x'])
            target.position.y = random.uniform(*self.workspace_bounds['y'])
            target.position.z = random.uniform(*self.workspace_bounds['z'])
            target.orientation.w = 1.0
            dist = np.linalg.norm([
                target.position.x - obstacle_pose.position.x,
                target.position.y - obstacle_pose.position.y,
                target.position.z - obstacle_pose.position.z
            ])
            if dist > min_distance:
                return target
        target = Pose()
        target.position.x = 0.3
        target.position.y = 0.0
        target.position.z = 0.6
        target.orientation.w = 1.0
        return target

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
        # Setup simulation
        obstacle = self.generate_random_obstacle_pose()
        self.spawn_obstacle(obstacle)
        target = self.generate_random_target_pose(obstacle)
        time.sleep(2.0)
        
        # Start distance recording
        self.start_recording()
        
        # Set target and execute
        self.set_target_pose(target)
        self.get_logger().info("Waiting briefly for robot state updates...")
        time.sleep(1.0)
        exec_time = 10.0
        self.get_logger().info(f"Monitoring for {exec_time} seconds.")
        
        # Keep spinning to receive distance messages
        end_time = time.time() + exec_time
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
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
            'raw_distance_data': self.distance_data[:100]  # Limit raw data to first 100 points
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/matteo/evaluation_results_with_distances_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({'result': result}, f, indent=2)
        self.get_logger().info(f"Results saved to {filename}")
        
        # Print summary
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