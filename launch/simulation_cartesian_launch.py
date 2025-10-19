# simulation_cartesian_launch.py
import os
import xacro
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription,
    RegisterEventHandler, Shutdown, AppendEnvironmentVariable, ExecuteProcess, TimerAction
)
from launch.event_handlers import OnProcessExit, OnProcessStart, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def get_robot_description(context, arm_id, load_gripper, franka_hand):
    arm_id_str = context.perform_substitution(arm_id)
    load_gripper_str = context.perform_substitution(load_gripper)
    franka_hand_str = context.perform_substitution(franka_hand)

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        arm_id_str,
        arm_id_str + '.urdf.xacro'
    )

    robot_description_config = xacro.process_file(
        franka_xacro_file,
        mappings={
            'arm_id': arm_id_str,
            'hand': load_gripper_str,
            'ros2_control': 'true',
            'gazebo': 'true',
            'ee_id': franka_hand_str,
            'gazebo_effort': 'true'
        }
    )
    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[('joint_states', '/gazebo/joint_states')],
    )

    return [robot_state_publisher]


def generate_launch_description():
    # --- Common arguments
    load_gripper_arg = DeclareLaunchArgument('load_gripper', default_value='true')
    franka_hand_arg = DeclareLaunchArgument('franka_hand', default_value='franka_hand')
    arm_id_arg = DeclareLaunchArgument('arm_id', default_value='fr3')
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value='sim', description='Hostname or IP of robot.')
    use_fake_hw_arg = DeclareLaunchArgument('use_fake_hardware', default_value='true')
    fake_sensor_cmd_arg = DeclareLaunchArgument('fake_sensor_commands', default_value='true')
    db_arg = DeclareLaunchArgument('db', default_value='False')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    # --- Launch configurations
    load_gripper = LaunchConfiguration('load_gripper')
    franka_hand = LaunchConfiguration('franka_hand')
    arm_id = LaunchConfiguration('arm_id')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # --- Gazebo
    set_gz_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('franka_description'))
    )

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': 'empty.sdf -r'}.items(),
    )

    # --- Robot description & publisher
    robot_state_publisher = OpaqueFunction(
        function=get_robot_description,
        args=[arm_id, load_gripper, franka_hand]
    )

    # --- Spawn robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # --- Controller spawners
    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '30'
        ],
        output='screen',
    )

    cic_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'cartesian_impedance_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '30'
        ],
        output='screen',
    )

    # --- MoveIt
    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'),
        'srdf', 'fr3_arm.srdf.xacro')

    robot_description_semantic_str = Command([
        FindExecutable(name='xacro'), ' ', franka_semantic_xacro_file, ' hand:=true'
    ])

    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(
            robot_description_semantic_str,
            value_type=str)
    }

    kinematics_yaml = load_yaml('franka_fr3_moveit_config', 'config/kinematics.yaml')
    ompl_planning_yaml = load_yaml('franka_fr3_moveit_config', 'config/ompl_planning.yaml')
    ompl_planning_pipeline_config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)

    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }

    moveit_controllers_yaml = load_yaml('franka_fr3_moveit_config', 'config/fr3_controllers.yaml')
    moveit_controllers = {
        'moveit_simple_controller_manager': moveit_controllers_yaml,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'publish_robot_description_semantic' : True,
        'monitor_octomap': False,
    }

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
        ],
    )

    # --- RViz
    rviz_config = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'),
        'rviz', 'moveit.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config],
        parameters=[
            {'use_sim_time': use_sim_time},
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
        ],
    )

    # --- Distance Calculator
    distance_calculator = Node(
        package='motion_planning_mt',
        executable='distance_calculator',
        name='distance_calculator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # --- CRITICAL: Evaluation Manager (THIS WAS MISSING!)
    evaluation_manager = Node(
        package='simulation_rmp_1',
        executable='evaluation_manager_cartesian',
        name='evaluation_manager',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # --- Shutdown handling
    shutdown_on_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=evaluation_manager,
            on_exit=[Shutdown()]
        )
    )

    kill_gui_on_shutdown = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                ExecuteProcess(cmd=['pkill', '-f', 'ign gazebo'], output='screen'),
                ExecuteProcess(cmd=['pkill', '-f', 'gz sim'], output='screen'),
                ExecuteProcess(cmd=['pkill', '-f', 'rviz2'], output='screen'),
            ]
        )
    )

    # --- Event chain
    chain_spawn_to_delay = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[TimerAction(period=3.0, actions=[jsb_spawner])])
    )

    chain_jsb_to_cic = RegisterEventHandler(
        OnProcessExit(target_action=jsb_spawner, on_exit=[cic_spawner])
    )

    chain_cic_to_move_group = RegisterEventHandler(
        OnProcessExit(target_action=cic_spawner, on_exit=[move_group])
    )

    start_distance_after_move_group_starts = RegisterEventHandler(
        OnProcessStart(target_action=move_group, on_start=[TimerAction(period=1.0, actions=[distance_calculator])])
    )

    start_eval_after_distance_starts = RegisterEventHandler(
        OnProcessStart(target_action=distance_calculator, on_start=[TimerAction(period=1.0, actions=[evaluation_manager])])
    )

    return LaunchDescription([
        # Args
        load_gripper_arg, franka_hand_arg, arm_id_arg,
        robot_ip_arg, use_fake_hw_arg, fake_sensor_cmd_arg, db_arg,
        use_sim_time_arg,

        # Gazebo + env path
        set_gz_resources,
        gazebo_world,

        # Robot description + spawn
        robot_state_publisher,
        spawn_robot,

        # Chain
        chain_spawn_to_delay,
        chain_jsb_to_cic,
        chain_cic_to_move_group,
        start_distance_after_move_group_starts,
        start_eval_after_distance_starts,

        # Visualization
        rviz_node,

        # Shutdown
        shutdown_on_exit,
        kill_gui_on_shutdown,
    ])