import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import GetPlanningScene
from shape_msgs.msg import SolidPrimitive
import random
import time
import json
import numpy as np
from datetime import datetime
from messages_fr3.msg import ClosestPoint 
from geometry_msgs.msg import PoseStamped
import threading


from moveit_msgs.srv import GetStateValidity, ApplyPlanningScene
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker  #Add this import for the target marker


class EvaluationManager(Node):
    def __init__(self):
        super().__init__('evaluation_manager')
        self.pose_pub = self.create_publisher(Pose, '/riemannian_motion_policy/reference_pose', 10)
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

            # Add marker publisher for target visualization
        self.target_marker_pub = self.create_publisher(Marker, '/target_marker', 10)
        
        # MoveIt planning scene service client for scene/state retrieval
        self.planning_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')

        # >>> NEW: service clients for validity check and synchronous scene apply <<<
        self.state_validity_client = self.create_client(GetStateValidity, '/check_state_validity')
        self.apply_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        
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
        
        # NEW: Subscribe to joint states for velocity logging
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.current_joint_state = None
        
        self.workspace_bounds = {'x': (0.2, 0.8), 'y': (-0.5, 0.5), 'z': (0.2, 0.8)}
        
        # MODIFIED: Random number of cylinders within range
        self.min_cylinders = 2
        self.max_cylinders = 6
        self.num_cylinders = random.randint(self.min_cylinders, self.max_cylinders)
        self.get_logger().info(f"Will spawn {self.num_cylinders} cylinders this simulation")
        
        # NEW: Cylinder size limits for random generation
        self.max_cylinder_height = 0.7  # maximum height (meters)
        self.max_cylinder_radius = 0.14 #0.12 maximum radius (meters)
        self.min_cylinder_height = 0.35   # minimum height
        self.min_cylinder_radius = 0.08  #0.08 minimum radius

        # Valid spawn zone constraints (derived from geometric_collision_check)
        self.min_distance_from_base = 0.40  # minimum safe distance from base
        self.max_distance_from_base = 1.3   # maximum reach distance
        
        # >>> NEW: Gaussian sampling controls <<<
        self.use_gaussian_sampling = True  # set False to revert to uniform
        # Radial distance r ~ truncated Normal(μ=0.50, σ=0.15) clipped to [min_distance_from_base, max_distance_from_base]
        self.gauss_r_mu = 0.50
        self.gauss_r_sigma = 0.15
        # Height z ~ truncated Normal within your z constraints (keep reasonable defaults)
        z_low = max(0.0, self.workspace_bounds['z'][0])
        z_high = min(0.7, self.workspace_bounds['z'][1])  # matches your 0.7 constraint below
        self.gauss_z_mu = 0.5 * (z_low + z_high)
        self.gauss_z_sigma = 0.12
        # Angle θ: keep uniform by default
        self.theta_distribution = 'uniform'  # or 'von_mises'
        self.theta_mu = 0.0
        self.theta_kappa = 4.0
        
        # NEW: Storage for obstacle information including sizes
        self.obstacle_info = []  # Will store [pose, height, radius, orientation] for each obstacle
        
        # MODIFIED: Enhanced distance data storage with regular sampling
        self.distance_data = []
        self.simulation_start_time = None
        self.is_recording = False
        
        # NEW: Regular sampling configuration
        self.sampling_rate = 20.0  # Hz (20 samples per second)
        self.max_samples = 500     # Increased from 100 to capture full trajectory
        self.sampling_timer = None
        
        # NEW: Latest distance data storage for regular sampling
        self.latest_distance_msg = None
        self.distance_data_lock = threading.Lock()
        
        # NEW: Path tracking variables for distance metrics
        self.start_ee_position = None
        self.end_ee_position = None
        self.previous_ee_position = None
        self.total_distance_traveled = 0.0
        self.path_positions = []  # Store all positions for debugging
        
        # NEW: Curvature tracking variables
        self.total_curvature = 0.0
        self.curvature_values = []  # Store individual curvature values for debugging
        self.previous_direction = None  # Store previous direction vector for curvature calculation
        
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
        """Store the latest distance message for regular sampling"""
        with self.distance_data_lock:
            self.latest_distance_msg = msg

    def joint_state_callback(self, msg):
        """Store the latest joint state data"""
        with self.distance_data_lock:
            self.current_joint_state = msg

    def calculate_curvature_between_points(self, p1, p2, p3):
        """
        Calculate curvature at point p2 given three consecutive points p1, p2, p3.
        Uses the formula: curvature = |angle between vectors| / average_segment_length
        """
        v1 = p2 - p1
        v2 = p3 - p2
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-6 or len2 < 1e-6:
            return 0.0
        v1_normalized = v1 / len1
        v2_normalized = v2 / len2
        dot_product = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
        angle = np.arccos(dot_product)
        average_length = (len1 + len2) / 2.0
        curvature = angle / average_length if average_length > 1e-6 else 0.0
        return curvature

    def regular_sampling_callback(self):
        """Regular timer callback to sample data at fixed intervals"""
        if not self.is_recording:
            return
            
        current_time = time.time()
        relative_time = current_time - self.simulation_start_time if self.simulation_start_time else 0
        
        current_ee_pos = self.get_end_effector_position()

     # Calculate distance to target
        distance_to_target = None
        if current_ee_pos is not None and hasattr(self, 'target_position'):
            distance_to_target = float(np.linalg.norm(current_ee_pos - self.target_position))
        
        if current_ee_pos is not None:
            if self.start_ee_position is None:
                self.start_ee_position = current_ee_pos.copy()
                self.get_logger().info(f"Start position recorded: [{self.start_ee_position[0]:.3f}, {self.start_ee_position[1]:.3f}, {self.start_ee_position[2]:.3f}]")
            if self.previous_ee_position is not None:
                distance_increment = np.linalg.norm(current_ee_pos - self.previous_ee_position)
                self.total_distance_traveled += distance_increment
            if len(self.path_positions) >= 2:
                p1 = self.path_positions[-2]
                p2 = self.path_positions[-1]
                p3 = current_ee_pos
                curvature = self.calculate_curvature_between_points(p1, p2, p3)
                self.total_curvature += curvature
                self.curvature_values.append(curvature)
                if curvature > 10.0:
                    self.get_logger().debug(f"High curvature detected: {curvature:.3f} rad/m at t={relative_time:.2f}s")
            self.previous_ee_position = current_ee_pos.copy()
            self.path_positions.append(current_ee_pos.copy())
            self.end_ee_position = current_ee_pos.copy()
        
        #Get joint velocities
        joint_velocities = None
        joint_names = None
        with self.distance_data_lock:
            if self.current_joint_state is not None:
                joint_names = list(self.current_joint_state.name)
                joint_velocities = list(self.current_joint_state.velocity)
        
        with self.distance_data_lock:
            msg = self.latest_distance_msg
        
        min_distances = {}
        overall_min = float('inf')
        num_obstacles_detected = 0
        
        if msg is not None:
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
            overall_min = min(min_distances.values()) if min_distances else float('inf')
            num_obstacles_detected = len(msg.frame2x)
        else:
            for link_name in ['link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'end_effector']:
                min_distances[link_name] = float('inf')
        
        distance_entry = {
            'timestamp': relative_time,
            'overall_min_distance': overall_min,
            'link_distances': min_distances,
            'num_obstacles_detected': num_obstacles_detected,
            'end_effector_position': current_ee_pos.tolist() if current_ee_pos is not None else None,
            'distance_to_target': distance_to_target, 
            'joint_names': joint_names,
            'joint_velocities': joint_velocities
        }
        self.distance_data.append(distance_entry)
        if overall_min < 0.1:
            self.get_logger().warn(f"Close approach detected: {overall_min:.3f}m at t={relative_time:.2f}s")
        if len(self.distance_data) >= self.max_samples:
            self.get_logger().info(f"Reached maximum samples ({self.max_samples}), stopping regular sampling")
            if self.sampling_timer:
                self.sampling_timer.cancel()

    def calculate_joint_velocity_metrics(self):
        """Calculate joint velocity statistics"""
        if not self.distance_data:
            return {'error': 'No distance data collected'}
        
        # Collect all joint velocity data
        all_joint_velocities = []
        joint_names = None
        timestamps = []
        
        for entry in self.distance_data:
            if entry.get('joint_velocities') is not None and entry.get('joint_names') is not None:
                if joint_names is None:
                    joint_names = entry['joint_names']
                all_joint_velocities.append(entry['joint_velocities'])
        
        if not all_joint_velocities:
            return {'error': 'No joint velocity data collected'}
        
        velocities_array = np.array(all_joint_velocities)  # Shape: (num_samples, num_joints)
        
        metrics = {
            'joint_names': joint_names,
            'num_samples': len(all_joint_velocities),
            'timestamps': timestamps,  # Include timestamps for each sample
            'raw_velocity_data': all_joint_velocities,  # NEW: Include all raw velocity data
            'per_joint_metrics': {},
            'per_joint_raw_data': {}  # Per-joint raw data for easier access
        }
        
        # Calculate metrics for each joint
        for i, joint_name in enumerate(joint_names):
            joint_vels = velocities_array[:, i]
            joint_vels_list = joint_vels.tolist()  # Convert to list for JSON serialization
            
            metrics['per_joint_metrics'][joint_name] = {
                'max_velocity': float(np.max(np.abs(joint_vels))),
                'avg_abs_velocity': float(np.mean(np.abs(joint_vels))),
                'std_velocity': float(np.std(joint_vels)),
                'max_positive': float(np.max(joint_vels)),
                'max_negative': float(np.min(joint_vels))
            }
            
            # NEW: Store raw velocity data for each joint
            metrics['per_joint_raw_data'][joint_name] = {
                'velocities': joint_vels_list,
                'timestamps': timestamps
            }
        
        # Overall metrics
        metrics['overall_max_velocity'] = float(np.max(np.abs(velocities_array)))
        metrics['overall_avg_velocity'] = float(np.mean(np.abs(velocities_array)))
        
        return metrics

    def wait_for_dependencies(self):
        self.get_logger().info("Waiting for dependencies...")
        while not self.count_subscribers('/planning_scene') > 0:
            time.sleep(1.0)
        while not self.count_subscribers('/riemannian_motion_policy/reference_pose') > 0:
            time.sleep(1.0)
        
        self.get_logger().info("Waiting for distance calculator...")
        timeout = 10.0
        start_wait = time.time()
        while not self.count_publishers('/closest_point') > 0:
            if time.time() - start_wait > timeout:
                self.get_logger().warn("Distance calculator not detected, continuing anyway...")
                break
            time.sleep(0.5)
        
        self.get_logger().info("Waiting for end-effector pose publisher...")
        start_wait = time.time()
        while not self.count_publishers('/end_effector_pose') > 0:
            if time.time() - start_wait > timeout:
                self.get_logger().warn("End-effector pose publisher not detected, continuing anyway...")
                break
            time.sleep(0.5)
        
        self.get_logger().info("Waiting for planning scene service...")
        start_wait = time.time()
        while not self.planning_scene_client.wait_for_service(timeout_sec=1.0):
            if time.time() - start_wait > timeout:
                self.get_logger().warn("Planning scene service not available, collision checking may not work properly")
                break
            self.get_logger().info("Planning scene service not available, waiting...")

        # >>> NEW: wait for /check_state_validity and /apply_planning_scene <<<
        self.get_logger().info("Waiting for /check_state_validity and /apply_planning_scene ...")
        start_wait = time.time()
        while (not self.state_validity_client.wait_for_service(timeout_sec=1.0) or
               not self.apply_scene_client.wait_for_service(timeout_sec=1.0)):
            if time.time() - start_wait > timeout:
                self.get_logger().warn("State validity or apply planning scene service not available.")
                break
            self.get_logger().info("...still waiting")
            
        time.sleep(2.0)
        self.get_logger().info("Dependencies ready.")

    def generate_random_cylinder_size(self):
        """Generate random height and radius for cylinder within limits"""
        height = random.uniform(self.min_cylinder_height, self.max_cylinder_height)
        radius = random.uniform(self.min_cylinder_radius, self.max_cylinder_radius)
        return height, radius

    def sample_valid_position_directly(self):
        """
        Directly sample a position that satisfies all constraints without iteration.
        Uses cylindrical coordinates to ensure distance constraints from robot base are met.
        Now uses cylindrical volumes instead of circular areas.
        """
        cylinder_height = 0.7
        min_volume = np.pi * self.min_distance_from_base**2 * cylinder_height
        max_volume = np.pi * self.max_distance_from_base**2 * cylinder_height
        random_volume = random.uniform(min_volume, max_volume)
        radius = np.sqrt(random_volume / (np.pi * cylinder_height))
        theta = random.uniform(0, 2 * np.pi)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        max_z_constraint = min(cylinder_height, self.workspace_bounds['z'][1])
        min_z_constraint = max(0.0, self.workspace_bounds['z'][0])
        z = random.uniform(min_z_constraint, max_z_constraint)
        x = np.clip(x, *self.workspace_bounds['x'])
        y = np.clip(y, *self.workspace_bounds['y'])
        z = np.clip(z, min_z_constraint, max_z_constraint)
        return x, y, z

    # >>> NEW: Gaussian sampling helpers and sampler <<<

    def _truncated_normal(self, mu, sigma, low, high, max_tries=100):
        """Sample a normal clipped to [low, high] via rejection; fall back to clip."""
        v = mu
        for _ in range(max_tries):
            v = random.gauss(mu, sigma)
            if low <= v <= high:
                return v
        return float(np.clip(v, low, high))

    def _sample_theta(self):
        if self.theta_distribution == 'von_mises':
            return float(np.random.vonmises(self.theta_mu, self.theta_kappa))
        return random.uniform(0.0, 2.0 * np.pi)

    def sample_valid_position_gaussian(self):
        """
        Sample (x,y,z) using truncated Gaussians in cylindrical coordinates,
        respecting workspace bounds and your min/max distance from base.
        """
        cylinder_height = 0.7
        min_z_constraint = max(0.0, self.workspace_bounds['z'][0])
        max_z_constraint = min(cylinder_height, self.workspace_bounds['z'][1])

        for _ in range(200):
            r = self._truncated_normal(self.gauss_r_mu, self.gauss_r_sigma,
                                       self.min_distance_from_base, self.max_distance_from_base)
            theta = self._sample_theta()
            z = self._truncated_normal(self.gauss_z_mu, self.gauss_z_sigma,
                                       min_z_constraint, max_z_constraint)

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Enforce workspace bounds strictly
            if (self.workspace_bounds['x'][0] <= x <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= y <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= z <= self.workspace_bounds['z'][1]):
                return x, y, z

        self.get_logger().warn("Gaussian sampler failed after 200 tries; falling back to uniform sampler.")
        return self.sample_valid_position_directly()

    # >>> NEW HELPERS: synchronous ApplyPlanningScene & RobotState retrieval <<<

    def _apply_scene(self, scene_diff: PlanningScene) -> bool:
        """Apply a PlanningScene diff synchronously via /apply_planning_scene."""
        try:
            req = ApplyPlanningScene.Request()
            req.scene = scene_diff
            future = self.apply_scene_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            ok = bool(future.result() and future.result().success)
            if not ok:
                self.get_logger().warn("ApplyPlanningScene failed.")
            return ok
        except Exception as e:
            self.get_logger().error(f"ApplyPlanningScene error: {e}")
            return False

    def _get_current_robot_state(self) -> RobotState:
        """Fetch the current RobotState from the planning scene."""
        try:
            req = GetPlanningScene.Request()
            req.components.components = (req.components.ROBOT_STATE |
                                         req.components.ROBOT_STATE_ATTACHED_OBJECTS)
            future = self.planning_scene_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() is None:
                self.get_logger().warn("Could not fetch current RobotState; assuming default.")
                return RobotState()
            return future.result().scene.robot_state
        except Exception as e:
            self.get_logger().error(f"GetPlanningScene for RobotState failed: {e}")
            return RobotState()

    # >>> REPLACED: real MoveIt collision check using GetStateValidity <<<
    def check_obstacle_collision_with_moveit(self, obstacle_pose, height, radius, orientation_type):
        """
        Returns True if the obstacle would collide with the current robot state.
        """
        try:
            # Unique temp id to avoid clashes
            temp_id = f"temp_collision_check_obstacle_{int(time.time()*1e6)}"

            temp_obj = CollisionObject()
            temp_obj.header.frame_id = "fr3_link0"   # Ensure this matches your planning frame
            temp_obj.header.stamp = self.get_clock().now().to_msg()
            temp_obj.id = temp_id

            cylinder = SolidPrimitive()
            cylinder.type = SolidPrimitive.CYLINDER
            cylinder.dimensions = [height, radius]  # [HEIGHT, RADIUS]
            temp_obj.primitives.append(cylinder)
            temp_obj.primitive_poses.append(obstacle_pose)
            temp_obj.operation = CollisionObject.ADD

            scene_diff = PlanningScene()
            scene_diff.is_diff = True
            scene_diff.world.collision_objects.append(temp_obj)

            # Apply ADD
            if not self._apply_scene(scene_diff):
                # Fail safe: treat as collision so we re-sample
                return True

            # Let monitor tick (usually not needed with Apply, but harmless)
            rclpy.spin_once(self, timeout_sec=0.05)

            # Ask validity
            robot_state = self._get_current_robot_state()
            sv_req = GetStateValidity.Request()
            sv_req.robot_state = robot_state
            sv_req.group_name = ""  # Entire robot; set your group if desired e.g. "arm"
            sv_future = self.state_validity_client.call_async(sv_req)
            rclpy.spin_until_future_complete(self, sv_future, timeout_sec=2.0)

            if sv_future.result() is not None:
                collision_detected = not sv_future.result().valid
            else:
                self.get_logger().warn("GetStateValidity failed; treating as collision to be safe.")
                collision_detected = True

            # Remove the temp object
            temp_obj.operation = CollisionObject.REMOVE
            scene_remove = PlanningScene()
            scene_remove.is_diff = True
            scene_remove.world.collision_objects.append(temp_obj)
            self._apply_scene(scene_remove)

            return collision_detected

        except Exception as e:
            self.get_logger().error(f"Error in MoveIt collision checking: {e}")
            # Fail safe
            return True

    def check_target_sphere_collision_with_moveit(self, obstacle_pose, height, radius, orientation_type, target_position, target_clearance=0.13):
    # """
    # Check if an obstacle would collide with a sphere around the target position.
    # Returns True if collision is detected (obstacle too close to target).
    
    # Args:
    #     obstacle_pose: Pose of the obstacle cylinder
    #     height, radius: Cylinder dimensions
    #     orientation_type: Orientation of the cylinder
    #     target_position: numpy array [x, y, z] of target position
    #     target_clearance: radius of sphere around target (default 13cm)
    
    # Returns:
    #     bool: True if obstacle is too close to target (collision detected)
    # """
        try:
            # Create unique IDs to avoid clashes with existing objects
            obstacle_id = f"temp_target_check_obstacle_{int(time.time()*1e6)}"
            sphere_id = f"temp_target_check_sphere_{int(time.time()*1e6)}"

            # Step 1: Create the obstacle cylinder
            obstacle_obj = CollisionObject()
            obstacle_obj.header.frame_id = "fr3_link0"
            obstacle_obj.header.stamp = self.get_clock().now().to_msg()
            obstacle_obj.id = obstacle_id

            cylinder = SolidPrimitive()
            cylinder.type = SolidPrimitive.CYLINDER
            cylinder.dimensions = [height, radius]  # [HEIGHT, RADIUS]
            obstacle_obj.primitives.append(cylinder)
            obstacle_obj.primitive_poses.append(obstacle_pose)
            obstacle_obj.operation = CollisionObject.ADD

            # Step 2: Create sphere around target position
            sphere_obj = CollisionObject()
            sphere_obj.header.frame_id = "fr3_link0"
            sphere_obj.header.stamp = self.get_clock().now().to_msg()
            sphere_obj.id = sphere_id

            sphere = SolidPrimitive()
            sphere.type = SolidPrimitive.SPHERE
            sphere.dimensions = [target_clearance]  # Only radius needed for sphere
            sphere_obj.primitives.append(sphere)
            
            # Create pose for sphere at target position
            sphere_pose = Pose()
            sphere_pose.position.x = target_position[0]
            sphere_pose.position.y = target_position[1]
            sphere_pose.position.z = target_position[2]
            sphere_pose.orientation.w = 1.0  # No rotation needed for sphere
            sphere_obj.primitive_poses.append(sphere_pose)
            sphere_obj.operation = CollisionObject.ADD

            # Step 3: Apply both objects to planning scene
            scene_diff = PlanningScene()
            scene_diff.is_diff = True
            scene_diff.world.collision_objects.append(obstacle_obj)
            scene_diff.world.collision_objects.append(sphere_obj)

            if not self._apply_scene(scene_diff):
                self.get_logger().warn("Failed to apply scene for target sphere collision check")
                return True  # Fail safe: treat as collision so we retry

            # Step 4: Small delay to ensure scene is properly updated
            rclpy.spin_once(self, timeout_sec=0.05)

            # Step 5: Calculate geometric overlap (simplified collision detection)
            obstacle_pos = np.array([obstacle_pose.position.x, obstacle_pose.position.y, obstacle_pose.position.z])
            center_distance = np.linalg.norm(obstacle_pos - target_position)
            
            # Calculate minimum distance from cylinder surface to sphere surface
            if orientation_type == 'vertical':
                # Cylinder axis along Z
                height_diff = abs(obstacle_pose.position.z - target_position[2])
                radial_distance = np.sqrt((obstacle_pose.position.x - target_position[0])**2 + 
                                        (obstacle_pose.position.y - target_position[1])**2)
                
                if height_diff <= height/2:
                    # Target is within cylinder height range
                    surface_distance = max(0, radial_distance - radius)
                else:
                    # Target is above/below cylinder
                    edge_height_distance = height_diff - height/2
                    if radial_distance <= radius:
                        surface_distance = edge_height_distance
                    else:
                        surface_distance = np.sqrt(edge_height_distance**2 + (radial_distance - radius)**2)
            else:
                # For horizontal cylinders, use simplified calculation
                surface_distance = max(0, center_distance - radius)
            
            # Collision detected if surface distance is less than target clearance
            collision_detected = surface_distance < target_clearance

            # Step 6: IMPORTANT - Clean up both temporary objects immediately
            obstacle_obj.operation = CollisionObject.REMOVE
            sphere_obj.operation = CollisionObject.REMOVE
            scene_remove = PlanningScene()
            scene_remove.is_diff = True
            scene_remove.world.collision_objects.append(obstacle_obj)
            scene_remove.world.collision_objects.append(sphere_obj)
            self._apply_scene(scene_remove)

            return collision_detected

        except Exception as e:
            self.get_logger().error(f"Error in target sphere collision checking: {e}")
            return True  # Fail safe: treat as collision

    def generate_random_orientation(self):
        """Generate random orientation for cylinder"""
        orientation_type = random.choice(['vertical', 'horizontal_x', 'horizontal_y'])
        
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
        
        return pose_orientation, orientation_type

    def generate_multiple_obstacle_poses_with_moveit_collision_check(self):
        """
        Generate multiple obstacle poses using MoveIt collision checking.
        Now also ensures obstacles are at least 13cm away from target pose using sphere collision detection.
        """
        obstacle_poses = []
        self.obstacle_info = []
        max_attempts_per_obstacle = 100
        min_distance_to_target = 0.13  # 13cm minimum distance to target
        
        self.get_logger().info("Using MoveIt collision checking for obstacle placement...")
        
        # STEP 1: Generate target position early for distance checking
        target = self.sample_valid_target_directly([])  # Empty list since no obstacles exist yet
        target_position = np.array([target.position.x, target.position.y, target.position.z])
        self.get_logger().info(f"Target position for obstacle distance checking: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        self.get_logger().info(f"Will check {min_distance_to_target*100:.0f}cm clearance around target using MoveIt sphere collision detection")
        
        # STEP 2: Generate obstacles with both robot collision and target distance checking
        for cylinder_idx in range(self.num_cylinders):
            self.get_logger().info(f"Generating obstacle {cylinder_idx + 1}/{self.num_cylinders}...")
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts_per_obstacle:
                attempts += 1
                height, radius = self.generate_random_cylinder_size()
                temp_pose = Pose()
                orientation, orientation_type = self.generate_random_orientation()
                temp_pose.orientation = orientation

                # Generate position
                if self.use_gaussian_sampling:
                    x, y, z = self.sample_valid_position_gaussian()
                else:
                    x, y, z = self.sample_valid_position_directly()

                temp_pose.position.x = x
                temp_pose.position.y = y
                temp_pose.position.z = z
                
                # STEP 3: Check if obstacle is too close to target using sphere collision
                target_collision_detected = self.check_target_sphere_collision_with_moveit(
                    temp_pose, height, radius, orientation_type, target_position, min_distance_to_target
                )
                
                if target_collision_detected:
                    if attempts % 20 == 0:
                        self.get_logger().debug(f"  Attempt {attempts}: Too close to target (sphere collision detected), retrying...")
                    continue
                
                # STEP 4: Check robot collision (existing check)
                robot_collision_detected = self.check_obstacle_collision_with_moveit(
                    temp_pose, height, radius, orientation_type
                )
                
                if not robot_collision_detected:
                    obstacle_poses.append(temp_pose)
                    self.obstacle_info.append({
                        'pose': temp_pose,
                        'height': height,
                        'radius': radius,
                        'orientation': orientation_type
                    })
                    orientation_str = orientation_type.replace('_', ' ')
                    self.get_logger().info(f"  ✓ Obstacle {cylinder_idx + 1} placed at [{x:.3f}, {y:.3f}, {z:.3f}] ({orientation_str})")
                    self.get_logger().info(f"    Size: height={height:.3f}m, radius={radius:.3f}m")
                    self.get_logger().info(f"    Target clearance: OK (>= {min_distance_to_target:.3f}m via MoveIt sphere check)")
                    self.get_logger().info(f"    Attempts needed: {attempts}")
                    placed = True
                else:
                    if attempts % 20 == 0:
                        self.get_logger().debug(f"  Attempt {attempts}: MoveIt detected robot collision, retrying...")
                        
            if not placed:
                self.get_logger().error(f"Failed to place obstacle {cylinder_idx + 1} after {max_attempts_per_obstacle} attempts")
                self.get_logger().error("Consider adjusting constraints, cylinder sizes, or target distance requirement")
                continue
        
        self.get_logger().info(f"Successfully placed {len(obstacle_poses)}/{self.num_cylinders} obstacles")
        self.get_logger().info("All obstacles respect 13cm clearance from target pose (verified via MoveIt sphere collision)")
        
        # STEP 5: Store the target for later use in simulation
        self.precomputed_target = target
        
        return obstacle_poses

    def sample_valid_target_directly(self, obstacle_poses):
        """
        Directly sample a valid target position that avoids obstacles.
        Now uses actual cylinder sizes from obstacle_info
        """
        min_clearance = 0.15
        max_attempts = 50
        
        for attempt in range(max_attempts):
            target = Pose()
            target.position.x = random.uniform(*self.workspace_bounds['x'])
            target.position.y = random.uniform(*self.workspace_bounds['y'])
            target.position.z = random.uniform(*self.workspace_bounds['z'])
            target.orientation.w = 1.0
            
            if self.is_point_inside_any_cylinder_with_sizes(target):
                continue
            
            valid_target = True
            for i, obstacle_pose in enumerate(obstacle_poses):
                obstacle_info = self.obstacle_info[i]
                dx = target.position.x - obstacle_pose.position.x
                dy = target.position.y - obstacle_pose.position.y
                dz = target.position.z - obstacle_pose.position.z
                center_distance = np.sqrt(dx**2 + dy**2 + dz**2)
                required_distance = min_clearance + obstacle_info['radius']
                if center_distance < required_distance:
                    valid_target = False
                    break
            
            if valid_target:
                return target
        
        self.get_logger().warn("Using corner fallback target position")
        target = Pose()
        target.position.x = self.workspace_bounds['x'][0] + 0.1
        target.position.y = self.workspace_bounds['y'][0] + 0.1
        target.position.z = self.workspace_bounds['z'][1] - 0.1
        target.orientation.w = 1.0
        return target

    def is_point_inside_any_cylinder_with_sizes(self, point):
        """Check if a point is inside any of the cylinders using actual sizes"""
        for i, obstacle_info in enumerate(self.obstacle_info):
            if self.is_point_inside_cylinder(
                point, 
                obstacle_info['pose'], 
                obstacle_info['height'], 
                obstacle_info['radius']
            ):
                return True
        return False

    def is_point_inside_cylinder(self, point, cylinder_pose, height, radius):
        """Check if a point is inside the cylinder - updated to handle different orientations and sizes"""
        dx = point.position.x - cylinder_pose.position.x
        dy = point.position.y - cylinder_pose.position.y
        dz = point.position.z - cylinder_pose.position.z
        
        qx = cylinder_pose.orientation.x
        qy = cylinder_pose.orientation.y
        qz = cylinder_pose.orientation.z
        qw = cylinder_pose.orientation.w
        
        if abs(qx) > 0.5 or abs(qy) > 0.5:
            if abs(qx) > 0.5:
                radial_distance = np.sqrt(dx**2 + dz**2)
                axial_distance = abs(dy)
            else:
                radial_distance = np.sqrt(dy**2 + dz**2)
                axial_distance = abs(dx)
            return radial_distance <= radius and axial_distance <= height / 2
        else:
            if abs(dz) > height / 2:
                return False
            radial_distance = np.sqrt(dx**2 + dy**2)
            return radial_distance <= radius

    def wait_and_check_initial_collision(self):
        """Wait for distance data and check if robot starts in collision"""
        self.get_logger().info("Checking initial robot safety...")
        initial_distances = None
        def temp_callback(msg):
            nonlocal initial_distances
            initial_distances = msg
        temp_sub = self.create_subscription(ClosestPoint, '/closest_point', temp_callback, 10)
        timeout = 5.0
        start_time = time.time()
        while initial_distances is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(temp_sub)
        if initial_distances is None:
            self.get_logger().warn("Could not get initial distance data")
            return True
        min_safe_distance = 0.05
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

    def spawn_obstacles(self, poses):
        """Spawn multiple obstacles with their individual sizes"""
        self.clear_obstacles()
        scene = PlanningScene()
        scene.is_diff = True
        for i, pose in enumerate(poses):
            obj = CollisionObject()
            obj.header.frame_id = "fr3_link0"
            obj.header.stamp = self.get_clock().now().to_msg()
            obj.id = f"evaluation_obstacle_{i}"
            cylinder = SolidPrimitive()
            cylinder.type = SolidPrimitive.CYLINDER
            obstacle_height = self.obstacle_info[i]['height']
            obstacle_radius = self.obstacle_info[i]['radius']
            cylinder.dimensions = [obstacle_height, obstacle_radius]
            obj.primitives.append(cylinder)
            obj.primitive_poses.append(pose)
            obj.operation = CollisionObject.ADD
            scene.world.collision_objects.append(obj)
        # >>> NEW: apply synchronously <<<
        if not self._apply_scene(scene):
            self.get_logger().error("Failed to spawn obstacles via ApplyPlanningScene")
        else:
            self.get_logger().info(f"Spawned {len(poses)} obstacles with individual sizes")

    def clear_obstacles(self):
        """Clear all obstacles"""
        scene = PlanningScene()
        scene.is_diff = True
        for i in range(10):
            obj = CollisionObject()
            obj.header.frame_id = "fr3_link0"
            obj.header.stamp = self.get_clock().now().to_msg()
            obj.id = f"evaluation_obstacle_{i}"
            obj.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(obj)
        temp_obj = CollisionObject()
        temp_obj.header.frame_id = "fr3_link0"
        temp_obj.header.stamp = self.get_clock().now().to_msg()
        temp_obj.id = "temp_collision_check_obstacle"
        temp_obj.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(temp_obj)
        # >>> NEW: apply synchronously <<<
        if not self._apply_scene(scene):
            self.get_logger().warn("Clear obstacles ApplyPlanningScene failed")

    def set_target_pose(self, pose):
        for _ in range(5):
            self.pose_pub.publish(pose)
            time.sleep(0.1)

    def publish_target_marker(self, target_pose):
        """Publish a red sphere marker at the target position for RViz visualization"""
        marker = Marker()
        
        # Header information
        marker.header.frame_id = "fr3_link0"  # Same frame as your robot
        marker.header.stamp = self.get_clock().now().to_msg()
        
        # Marker properties
        marker.ns = "target_pose"  # Namespace
        marker.id = 0  # Unique ID
        marker.type = Marker.SPHERE  # Sphere shape
        marker.action = Marker.ADD  # Add/update the marker
        
        # Position (from target pose)
        marker.pose.position.x = target_pose.position.x
        marker.pose.position.y = target_pose.position.y
        marker.pose.position.z = target_pose.position.z
        marker.pose.orientation.w = 1.0  # No rotation needed for sphere
        
        # Size of the sphere (diameter in meters)
        marker.scale.x = 0.05  # 5cm diameter
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Color (red sphere)
        marker.color.r = 1.0  # Full red
        marker.color.g = 0.0  # No green
        marker.color.b = 0.0  # No blue
        marker.color.a = 0.8  # 80% opacity (slightly transparent)
        
        # Publish the marker
        self.target_marker_pub.publish(marker)
        self.get_logger().info(f"Published target marker at [{target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f}]")

    def start_recording(self):
        """Start recording distance data with regular sampling"""
        self.distance_data = []
        self.simulation_start_time = time.time()
        self.is_recording = True
        self.start_ee_position = None
        self.end_ee_position = None
        self.previous_ee_position = None
        self.total_distance_traveled = 0.0
        self.path_positions = []
        self.total_curvature = 0.0
        self.curvature_values = []
        self.previous_direction = None
        sampling_interval = 1.0 / self.sampling_rate
        self.sampling_timer = self.create_timer(sampling_interval, self.regular_sampling_callback)
        self.get_logger().info(f"Started recording distance data at {self.sampling_rate} Hz (max {self.max_samples} samples)")

    def stop_recording(self):
        """Stop recording distance data"""
        self.is_recording = False
        if self.sampling_timer:
            self.sampling_timer.cancel()
            self.sampling_timer = None
        self.get_logger().info(f"Stopped recording. Collected {len(self.distance_data)} distance measurements")

    def wait_for_obstacles_to_be_active(self):
        """Wait for obstacles to be properly loaded and detected by the distance calculator."""
        self.get_logger().info("Waiting for obstacles to be active and detectable...")
        timeout = 10.0
        start_wait_time = time.time()
        obstacles_detected = False
        def temp_distance_callback(msg):
            nonlocal obstacles_detected
            total_obstacles = 0
            links_to_check = [
                msg.frame2x, msg.frame3x, msg.frame4x, msg.frame5x,
                msg.frame6x, msg.frame7x, msg.framehandx, msg.frameeex
            ]
            for link_data in links_to_check:
                total_obstacles += len(link_data)
            if total_obstacles > 0:
                obstacles_detected = True
                self.get_logger().info(f"Obstacles detected! Total obstacle interactions: {total_obstacles}")
        temp_sub = self.create_subscription(ClosestPoint, '/closest_point', temp_distance_callback, 10)
        while not obstacles_detected and (time.time() - start_wait_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = time.time() - start_wait_time
            if int(elapsed) % 2 == 0 and elapsed > 0:
                remaining = timeout - elapsed
                if remaining > 0:
                    self.get_logger().info(f"Still waiting for obstacle detection... ({remaining:.1f}s remaining)")
        self.destroy_subscription(temp_sub)
        if obstacles_detected:
            self.get_logger().info("✓ Obstacles successfully loaded and detected by distance calculator")
            time.sleep(0.5)
            return True
        else:
            self.get_logger().warn(f" Timeout waiting for obstacle detection after {timeout}s")
            self.get_logger().warn("  Continuing anyway - obstacles may not be properly loaded")
            return False

    def calculate_path_metrics(self):
        """Calculate path efficiency metrics including curvature"""
        if self.start_ee_position is None or self.end_ee_position is None:
            return {
                'total_distance_traveled': None,
                'straight_line_distance': None,
                'path_efficiency_ratio': None,
                'total_curvature': None,
                'average_curvature': None,
                'max_curvature': None,
                'curvature_per_distance': None,
                'error': 'Start or end position not recorded'
            }
        straight_line_distance = np.linalg.norm(self.end_ee_position - self.start_ee_position)
        path_efficiency_ratio = None
        if straight_line_distance > 0:
            path_efficiency_ratio = self.total_distance_traveled / straight_line_distance
        average_curvature = None
        max_curvature = None
        curvature_per_distance = None
        normalized_total_curvature = None
        curvature_per_sampled_point = None 
        if len(self.curvature_values) > 0:
            average_curvature = np.mean(self.curvature_values)
            max_curvature = np.max(self.curvature_values)
            if self.total_distance_traveled > 0:
                curvature_per_distance = self.total_curvature / self.total_distance_traveled
            if straight_line_distance > 0:
                normalized_total_curvature = self.total_curvature / straight_line_distance
        total_sampled_points = len(self.distance_data)
        if total_sampled_points > 0:
            curvature_per_sampled_point = self.total_curvature / total_sampled_points
        return {
            'total_distance_traveled': self.total_distance_traveled,
            'straight_line_distance': straight_line_distance,
            'path_efficiency_ratio': path_efficiency_ratio,
            'start_position': self.start_ee_position.tolist(),
            'end_position': self.end_ee_position.tolist(),
            'num_path_points': len(self.path_positions),
            'total_sampled_points': total_sampled_points,
            'total_curvature': self.total_curvature,
            'average_curvature': average_curvature,
            'max_curvature': max_curvature,
            'curvature_per_distance': curvature_per_distance,
            'normalized_total_curvature': normalized_total_curvature,
            'curvature_per_sampled_point': curvature_per_sampled_point,
            'num_curvature_points': len(self.curvature_values)
        }

    def run_single_simulation(self):
        obstacles = self.generate_multiple_obstacle_poses_with_moveit_collision_check()
        self.spawn_obstacles(obstacles)
        
        self.get_logger().info("Obstacles spawned, now waiting for them to be properly loaded...")
        obstacles_ready = self.wait_for_obstacles_to_be_active()
        
        self.get_logger().info("✓ Started sampling data - obstacles are confirmed active and simulation ready!")
        
        is_safe = self.wait_and_check_initial_collision()
        if not is_safe:
            self.get_logger().warn("Initial collision detected but continuing (this should be rare with MoveIt collision checking)")
        
        # STEP 3: Use the precomputed target (no sphere around it anymore)
        if hasattr(self, 'precomputed_target'):
            target = self.precomputed_target
            self.get_logger().info("Using precomputed target that respects 13cm clearance from all obstacles")
            self.get_logger().info("Target sphere was used only for obstacle placement and has been removed")
        else:
            # Fallback to old method if something went wrong
            target = self.sample_valid_target_directly(obstacles)
            self.get_logger().warn("Fallback: using newly sampled target")

        time.sleep(2.0)
        
        self.target_position = np.array([target.position.x, target.position.y, target.position.z])
        self.goal_tolerance = 0.02
        self.goal_reached = False
        self.goal_reach_time = None
        self.position_check_count = 0
        self.successful_position_checks = 0

        self.set_target_pose(target)
        #Publish target marker for RViz visualization
        self.publish_target_marker(target)

        self.get_logger().info(f"Target position: [{target.position.x:.3f}, {target.position.y:.3f}, {target.position.z:.3f}]")
        
        self.set_target_pose(target)
        self.get_logger().info(f"Target position: [{target.position.x:.3f}, {target.position.y:.3f}, {target.position.z:.3f}]")
        self.get_logger().info("Robot can now freely approach target - no collision sphere around target")
        self.get_logger().info("Waiting briefly for robot state updates...")
        time.sleep(1.0)
        self.start_recording()
        exec_time = 15.0
        self.get_logger().info(f"Monitoring for {exec_time} seconds.")
        
        end_time = time.time() + exec_time
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.goal_reached:
                self.position_check_count += 1
                current_pos = self.get_end_effector_position()
                if current_pos is not None:
                    self.successful_position_checks += 1
                    dist = np.linalg.norm(current_pos - self.target_position)
                    if self.position_check_count % 50 == 0:
                        self.get_logger().info(f"Current pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}], distance to goal: {dist:.3f}m")
                    if dist <= self.goal_tolerance:
                        self.goal_reached = True
                        self.goal_reach_time = time.time() - self.simulation_start_time
                        self.get_logger().info(f"Goal reached at t={self.goal_reach_time:.2f}s (distance {dist:.3f}m)")
        self.get_logger().info(f"Position checks: {self.position_check_count}, successful: {self.successful_position_checks}")
        if self.successful_position_checks == 0:
            self.get_logger().warn("No successful position checks - end-effector pose not available!")
        
        self.stop_recording()
        path_metrics = self.calculate_path_metrics()
        analysis = self.analyze_distance_data()
        joint_velocity_metrics = self.calculate_joint_velocity_metrics()
        obstacle_positions = [[obs.position.x, obs.position.y, obs.position.z] for obs in obstacles]
        obstacle_sizes = []
        for obs_info in self.obstacle_info:
            obstacle_sizes.append({
                'height': obs_info['height'],
                'radius': obs_info['radius'],
                'orientation': obs_info['orientation']
            })
        result = {
            'obstacle_positions': obstacle_positions,
            'obstacle_sizes': obstacle_sizes,
            'num_obstacles': len(obstacles),
            'target_position': [target.position.x, target.position.y, target.position.z],
            'execution_time': exec_time,
            'timestamp': datetime.now().isoformat(),
            'distance_analysis': analysis,
            'raw_distance_data': self.distance_data,
            'goal_reached': self.goal_reached,
            'goal_reach_time': self.goal_reach_time,
            'goal_tolerance': self.goal_tolerance,
            'position_check_count': self.position_check_count,
            'successful_position_checks': self.successful_position_checks,
            'moveit_collision_checking_used': True,
            'initial_safety_check': is_safe,
            'sampling_rate': self.sampling_rate,
            'total_samples_collected': len(self.distance_data),
            'max_samples_configured': self.max_samples,
            'obstacles_properly_loaded': obstacles_ready,
            'path_metrics': path_metrics,
            'joint_velocity_analysis': joint_velocity_metrics
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
            'close_calls': len([d for d in overall_distances if d < 0.05]),
            'safety_violations': len([d for d in overall_distances if d < 0.02])
        }
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
        existing_data = {}
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    existing_data = json.load(f)
                self.get_logger().info(f"Loaded existing data with {len(existing_data)} simulations")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.get_logger().warn(f"Could not load existing data: {e}, starting fresh")
                existing_data = {}
        simulation_numbers = [int(k.replace('simulation_', '')) for k in existing_data.keys() if k.startswith('simulation_')]
        next_sim_num = max(simulation_numbers) + 1 if simulation_numbers else 1
        existing_data[f'simulation_{next_sim_num}'] = result
        with open(self.results_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        self.get_logger().info(f"Results saved to {self.results_file} as simulation_{next_sim_num}")
        self.get_logger().info(f"Simulation {next_sim_num} - Goal Achievement Summary:")
        self.get_logger().info(f"  Goal reached: {result.get('goal_reached', False)}")
        if result.get('goal_reach_time') is not None:
            self.get_logger().info(f"  Time to reach goal: {result['goal_reach_time']:.2f}s")
        self.get_logger().info(f"  Goal tolerance: {result.get('goal_tolerance', 'N/A')}m")
        self.get_logger().info(f"  Position checks: {result.get('successful_position_checks', 0)}/{result.get('position_check_count', 0)}")
        if 'path_metrics' in result and result['path_metrics'].get('total_distance_traveled') is not None:
            path_metrics = result['path_metrics']
            self.get_logger().info(f"Path Efficiency Summary:")
            self.get_logger().info(f"  Total distance traveled: {path_metrics['total_distance_traveled']:.3f}m")
            self.get_logger().info(f"  Straight-line distance: {path_metrics['straight_line_distance']:.3f}m")
            if path_metrics['path_efficiency_ratio'] is not None:
                self.get_logger().info(f"  Path efficiency ratio: {path_metrics['path_efficiency_ratio']:.2f}")
            self.get_logger().info(f"  Path points recorded: {path_metrics['num_path_points']}")
            if path_metrics.get('total_curvature') is not None:
                self.get_logger().info(f"Trajectory Curvature Summary:")
                self.get_logger().info(f"  Total curvature: {path_metrics['total_curvature']:.3f} rad")
                if path_metrics.get('average_curvature') is not None:
                    self.get_logger().info(f"  Average curvature: {path_metrics['average_curvature']:.3f} rad/m")
                if path_metrics.get('max_curvature') is not None:
                    self.get_logger().info(f"  Maximum curvature: {path_metrics['max_curvature']:.3f} rad/m")
                if path_metrics.get('curvature_per_distance') is not None:
                    self.get_logger().info(f"  Curvature per distance: {path_metrics['curvature_per_distance']:.3f} rad/m")
                if path_metrics.get('normalized_total_curvature') is not None:
                    self.get_logger().info(f"  Normalized total curvature: {path_metrics['normalized_total_curvature']:.3f} rad/m")
                if path_metrics.get('curvature_per_sampled_point') is not None:
                    self.get_logger().info(f"  Curvature per sampled point: {path_metrics['curvature_per_sampled_point']:.4f} rad/point")
                if path_metrics.get('total_sampled_points') is not None:
                    self.get_logger().info(f"  Total sampled points: {path_metrics['total_sampled_points']}")
                self.get_logger().info(f"  Curvature points calculated: {path_metrics.get('num_curvature_points', 0)}")
        else:
            self.get_logger().warn("Path metrics not available - end-effector position tracking failed")
        
        # NEW: Joint velocity summary logging
        if 'joint_velocity_analysis' in result and 'per_joint_metrics' in result['joint_velocity_analysis']:
            joint_metrics = result['joint_velocity_analysis']
            self.get_logger().info(f"Joint Velocity Summary:")
            self.get_logger().info(f"  Overall max velocity: {joint_metrics.get('overall_max_velocity', 'N/A'):.3f} rad/s")
            self.get_logger().info(f"  Overall avg velocity: {joint_metrics.get('overall_avg_velocity', 'N/A'):.3f} rad/s")
            self.get_logger().info(f"  Samples collected: {joint_metrics.get('num_samples', 'N/A')}")
            
            for joint_name, metrics in joint_metrics['per_joint_metrics'].items():
                self.get_logger().info(f"  {joint_name}:")
                self.get_logger().info(f"    Max |velocity|: {metrics['max_velocity']:.3f} rad/s")
                self.get_logger().info(f"    Avg |velocity|: {metrics['avg_abs_velocity']:.3f} rad/s")
        else:
            self.get_logger().warn("Joint velocity metrics not available")
        
        self.get_logger().info(f"Obstacle Summary:")
        self.get_logger().info(f"  Number of obstacles: {result.get('num_obstacles', 'N/A')}")
        self.get_logger().info(f"  MoveIt collision checking used: {result.get('moveit_collision_checking_used', False)}")
        self.get_logger().info(f"  Obstacles properly loaded: {result.get('obstacles_properly_loaded', 'Unknown')}")
        if 'obstacle_positions' in result and 'obstacle_sizes' in result:
            for i, (pos, size) in enumerate(zip(result['obstacle_positions'], result['obstacle_sizes'])):
                self.get_logger().info(f"  Obstacle {i+1}:")
                self.get_logger().info(f"    Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                self.get_logger().info(f"    Size: h={size['height']:.3f}m, r={size['radius']:.3f}m ({size['orientation']})")
        self.get_logger().info(f"Sampling Summary:")
        self.get_logger().info(f"  Sampling rate: {result.get('sampling_rate', 'N/A')} Hz")
        self.get_logger().info(f"  Total samples collected: {result.get('total_samples_collected', 'N/A')}")
        self.get_logger().info(f"  Max samples configured: {result.get('max_samples_configured', 'N/A')}")
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
