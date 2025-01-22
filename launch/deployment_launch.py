from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Example of how to get the path to an RViz config file if needed
    # pkg_share_dir = get_package_share_directory('rehab_robot_bayes_ros2')
    # rviz_config_file = os.path.join(pkg_share_dir, 'config', 'simulation_config.rviz')
    rviz_config_path = '/home/awear/ros2_ws/src/rehab_robot_bayes_ros2/config/simulation_config.rviz'

    return LaunchDescription([
        # 1) Launch the Bayesian Optimization node
        Node(
            package='rehab_robot_bayes_ros2',
            executable='bayes_opt_node',
            output='screen',
            arguments=[
                '--freq', '10.0',   # Frequency in Hz
                '--show-plot',      # Enable plotting
                '--max-runs', '10'  # Maximum number of runs
            ],
        ),

        # 2) Launch the simple Viz node (viz_node) to show /mes_pos and /ref_pos in RViz
        Node(
            package='rehab_robot_bayes_ros2',
            executable='viz_node',
            output='screen',
            arguments=[
                '--freq', '10.0',       # Marker update frequency in Hz
                '--frame-id', 'world'   # Frame ID for the markers
            ],
        ),

        # 3) Launch RViz2 with a specified config file
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=[
                '-d', rviz_config_path
                # or '-d', rviz_config_file if using the path above
            ],
            output='screen',
        ),
    ])
