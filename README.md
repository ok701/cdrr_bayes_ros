# Rehablitation Robot Bayesian Optimization ROS 2 Package

This package demonstrates Bayesian optimization for a rehabilitation robot in ROS 2, designed for deployment in real systems. It aims to find the optimal rehabilitation assistive and resistive forces tailored for patients.

## Deployment Constraints
The high-level control logic is implemented in Python within a ROS 2 framework, while LabVIEW is used for hardware interfacing and the graphical user interface. This means that the system cannot be deployed as a standalone Python application. If you require LabVIEW integration, you can use the `cdrr_control_supine` repository. Alternatively, if you prefer a different setup, you are free to create hardware settings tailored to your needs.

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