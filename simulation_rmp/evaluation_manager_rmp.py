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

class EvaluationManager(Node):
    def __init__(self):
        super().__init__('evaluation_manager')
        self.pose_pub = self.create_publisher(Pose, '/riemannian_motion_policy/reference_pose', 10)
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.workspace_bounds = {'x': (0.2, 0.8), 'y': (-0.5, 0.5), 'z': (0.2, 0.8)}
        self.wait_for_dependencies()
        self.get_logger().info("Starting single simulation...")
        result = self.run_single_simulation()
        self.save_results(result)
        self.get_logger().info("Simulation complete.")
        

    def wait_for_dependencies(self):
        self.get_logger().info("Waiting for dependencies...")
        while not self.count_subscribers('/planning_scene') > 0:
            time.sleep(1.0)
        while not self.count_subscribers('/riemannian_motion_policy/reference_pose') > 0:
            time.sleep(1.0)
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

    def run_single_simulation(self):
        obstacle = self.generate_random_obstacle_pose()
        self.spawn_obstacle(obstacle)
        target = self.generate_random_target_pose(obstacle)
        time.sleep(2.0)
        self.set_target_pose(target)
        exec_time = 10.0
        self.get_logger().info(f"Monitoring for {exec_time} seconds.")
        time.sleep(exec_time)
        result = {
            'obstacle_position': [obstacle.position.x, obstacle.position.y, obstacle.position.z],
            'target_position': [target.position.x, target.position.y, target.position.z],
            'execution_time': exec_time,
            'timestamp': datetime.now().isoformat()
        }
        return result

    def save_results(self, result):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/matteo/evaluation_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({'result': result}, f, indent=2)
        self.get_logger().info(f"Results saved to {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = EvaluationManager()
    node.destroy_node()
    rclpy.shutdown()
    print("Evaluation completed and node shut down.")
    
if __name__ == '__main__':
    main()
