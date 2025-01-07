# Rehablitation Robot Bayesian Optimization ROS2 Package

This package demonstrates Bayesian optimization for a rehabilitation robot in ROS 2, designed for deployment in real systems. It aims to find the optimal rehabilitation assistive and resistive forces tailored for patients.

Currently under development for a cable-driven system, this package requires ROS 2 installation as a prerequisite.

## Features
- Linear environment simulation
- Bayesian optimization using Gaussian processes
- Real-time RViz visualization

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/rehab_robot_bayes_ros2.git
```

2. Install dependencies:
```bash
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the workspace:
```bash
colcon build
```