from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Get the package share directory for rehab_robot_bayes_ros2
    # pkg_share_dir = get_package_share_directory('rehab_robot_bayes_ros2')
    # pkg_dir = os.path.abspath(os.path.join(pkg_share_dir, "..", "..", "..", ".."))
    # rviz_config_path = os.path.join(pkg_dir, 'config', 'simulation_config.rviz')
    rviz_config_path = '/home/awear/ros2_ws/src/rehab_robot_bayes_ros2/config/simulation_config.rviz'


    return LaunchDescription([
        # Launch bayes_opt_node with required arguments
        Node(
            package='rehab_robot_bayes_ros2',
            executable='bayes_opt_node',
            # name='bayesian_optimization',
            output='screen',
            arguments=[
                '--freq', '10.0',  # Frequency in Hz
                '--show-plot',     # Enable plotting
                '--max-runs', '10' # Maximum number of runs
            ],
        ),
        # Launch env_node with required arguments
        Node(
            package='rehab_robot_bayes_ros2',
            executable='env_node',
            # name='environment_simulation',
            output='screen',
            arguments=[
                '--freq', '50.0',      # Simulation frequency in Hz
                '--frame-id', 'world'  # Frame ID for RViz visualization
            ],
        ),
        # Launch RViz2 with the specified configuration file
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            # arguments=[
            #     '-d', 'install/rehab_robot_bayes_ros2/share/rehab_robot_bayes_ros2/config/simulation_config.rviz'
            # ],
            arguments=[
                '-d', rviz_config_path
            ],
            output='screen',
        ),
    ])
