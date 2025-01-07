import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/awear/ros2_ws/src/rehab_robot_bayes_ros2/install/rehab_robot_bayes_ros2'
