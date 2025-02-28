# Bayesian Optimization ROS 2 Package<br>for Rehablitation Robot 

This package demonstrates Bayesian optimization for a rehabilitation robot in ROS 2, designed for deployment in real systems. It aims to find the optimal constant rehabilitation assistive and resistive forces tailored for patients.

## Deployment Constraints
The high-level control logic is implemented in Python within a ROS 2 framework, while LabVIEW is used for hardware interfacing and the graphical user interface. This means that the system cannot be deployed as a standalone Python application. If you require LabVIEW integration, you can use the `cdrr_control` repository. Alternatively, if you prefer a different setup, you are free to create hardware settings tailored to your needs.

### Main Topics
[rosgraph_deploy](./assets/rosgraph_deploy.png)
[rosgraph_sim](./assets/rosgraph_sim.png)

- `/mes_pos`
- `/ref_pos`
- `/ref_force`


### Usage
To simulate uzing RViz:
```bash
roslaunch 
```
[visual](./assets/visual.png)


[bayes](./assets/bayes.png)
