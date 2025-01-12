from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'rehab_robot_bayes_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',         # Required for Python package installation
        'rclpy',              # ROS 2 Python client library
        'std_msgs',           # Standard ROS 2 messages (e.g., Int32)
        'visualization_msgs', # For RViz marker messages
        'geometry_msgs',      # For marker scale (Vector3)
        'numpy',              # Numerical computations
        'matplotlib',         # Real-time plotting
        'scipy',              # Statistical functions (e.g., norm)
        'scikit-learn',       # Gaussian Process Regressor and kernels
    ],
    zip_safe=True,
    maintainer='jinu',
    maintainer_email='realfcn@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'env_node = rehab_robot_bayes_ros2.env_node:main',
            'bayes_opt_node = rehab_robot_bayes_ros2.bayes_opt_node:main',
            'viz_node = rehab_robot_bayes_ros2.viz_node:main',
        ],
    },
)
