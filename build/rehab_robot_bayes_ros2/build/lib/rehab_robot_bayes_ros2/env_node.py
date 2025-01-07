#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Vector3

import sys
import argparse


class LinearEnvSimVizNode(Node):
    """
    A ROS2 node that:
      - Simulates a 1D linear environment in mm in real-time.
      - Accepts an external force in Newtons via /manipulated_var.
      - Visualizes the environment in RViz (converted to meters for the marker).
      - Resets the simulation on a /trigger signal:
          - 0 -> 1: Start the simulation.
          - 1 -> 0: Reset the simulation and go back to x = 0.
    """

    def __init__(self, freq=50.0, frame_id='world'):
        super().__init__('linear_env_sim_viz')

        # Simulation parameters (in mm, mm/s, mm/s^2, but mass in kg, force in N)
        self.mass = 50.0         # Mass in kg
        self.b = 0.1               # Damping, units = N / (mm/s)
        self.L = 300.0           # Max length in mm
        self.T = 5.0             # Total time in seconds
        self.freq = freq
        self.dt = 1.0 / freq     # Time step in seconds
        self.frame_id = frame_id
        self.Kp = 0.1

        # ROS subscriptions and publications
        self.trigger_sub = self.create_subscription(
            Int32, '/trigger', self.trigger_callback, 10
        )
        self.manipulated_var_sub = self.create_subscription(
            Int32, '/manipulated_var', self.manipulated_var_callback, 10
        )
        self.ee_error_pub = self.create_publisher(Int32, '/ee_error', 10)
        self.env_x_pub = self.create_publisher(Int32, '/env_x', 10)
        self.env_xref_pub = self.create_publisher(Int32, '/env_xref', 10)

        self.mass_marker_pub = self.create_publisher(Marker, '/mass_marker', 10)
        self.xref_marker_pub = self.create_publisher(Marker, '/xref_marker', 10)

        # Internal state
        self.trigger_state = 0
        self.simulation_active = False
        self.cable_force = 0.0  # Force in N
        self.x = 0.0            # Position in mm
        self.x_dot = 0.0        # Velocity in mm/s
        self.time_sim = 0.0

        self.reset_simulation()
        self.timer = self.create_timer(self.dt, self.update_and_visualize)

        self.get_logger().info(f"LinearEnvSimVizNode started at {freq} Hz.")

    def reset_simulation(self, go_to_start=False):
        """
        Reset simulation variables for a new iteration.
        If go_to_start=True, set x and x_dot to 0.
        """
        self.time_sim = 0.0
        self.cable_force = 0.0
        self.x_dot = 0.0
        if go_to_start:
            self.x = 0.0
            self.x_dot = 0.0
        self.simulation_active = False

    def trigger_callback(self, msg: Int32):
        """
        /trigger:
          - 0->1: Start simulation
          - 1->0: Reset simulation
        """
        new_trigger = msg.data
        if new_trigger == 1 and self.trigger_state == 0:
            self.get_logger().info("Trigger 0->1: Starting simulation.")
            self.simulation_active = True

        elif new_trigger == 0 and self.trigger_state == 1:
            self.get_logger().info("Trigger 1->0: Resetting to start position.")
            self.reset_simulation(go_to_start=True)
            self.publish_marker(self.mass_marker_pub, self.x / 1000.0, 0.05, (1.0, 0.0, 0.0))
            self.publish_marker(self.xref_marker_pub, self.x / 1000.0, 0.05, (0.0, 1.0, 0.0))

        self.trigger_state = new_trigger

    def manipulated_var_callback(self, msg: Int32):
        """
        /manipulated_var:
        Interpreted directly as a force in N.
        E.g., msg.data = 2 => 2 N
        """
        self.cable_force = float(msg.data)
        self.get_logger().info(f"Updated cable force to {self.cable_force:.2f} N")

    def update_and_visualize(self):
        """
        1. Update the simulation dynamics (in mm, mm/s, mm/s^2).
        2. Publish state in mm.
        3. Visualize in RViz (convert mm to m for markers).
        """
        if not self.simulation_active:
            return

        x_ref = self.target_trajectory(self.time_sim)
        human_force = self.Kp * (x_ref - self.x)

        # Net force in N = external cable_force - damping * velocity
        # (b has units N/(mm/s), x_dot in mm/s => b*x_dot in N)
        net_force = self.cable_force - (self.b * self.x_dot) + human_force

        # x_ddot in mm/s^2 = 1000 * ( net_force / mass in kg )
        x_ddot = 1000.0 * (net_force / self.mass)

        # Integrate
        self.x_dot += x_ddot * self.dt       # mm/s
        self.x += self.x_dot * self.dt       # mm

        # Clamp [0, L]
        if self.x < 0.0:
            self.x = 0.0
            self.x_dot = 0.0
        elif self.x > self.L:
            self.x = self.L
            self.x_dot = 0.0

        self.time_sim += self.dt

        # Stop simulation if we pass T
        if self.time_sim >= self.T:
            self.simulation_active = False
            self.get_logger().info("Simulation complete. Waiting for next trigger.")

        # Compute reference in mm
        # x_ref = self.target_trajectory(self.time_sim)

        # Publish states (Int32 in mm)
        self.env_x_pub.publish(Int32(data=int(self.x)))
        self.env_xref_pub.publish(Int32(data=int(x_ref)))
        self.ee_error_pub.publish(Int32(data=int(self.x - x_ref)))

        # RViz Markers: positions must be in meters
        self.publish_marker(self.mass_marker_pub, self.x / 1000.0, 0.05, (1.0, 0.0, 0.0))
        self.publish_marker(self.xref_marker_pub, x_ref / 1000.0, 0.05, (0.0, 1.0, 0.0))

    def publish_marker(self, topic, position_m, scale_m, color):
        """
        Publish a sphere Marker in RViz (all distances in meters here).
        """
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "simulation_ns"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(position_m)
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Scale in meters
        marker.scale = Vector3(x=scale_m, y=scale_m, z=scale_m)

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        topic.publish(marker)

    def target_trajectory(self, t):
        """
        Minimum-jerk trajectory in mm, from 0 to L over T seconds.
        """
        if t >= self.T:
            return self.L
        tau = t / self.T
        return self.L * (10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)


def main(args=None):
    # Create an argument parser and define known arguments
    parser = argparse.ArgumentParser(description="1D Linear Environment with Visualization in RViz")
    parser.add_argument("--freq", type=float, default=50.0, help="Simulation frequency")
    parser.add_argument("--frame-id", type=str, default="world", help="Frame ID for markers")

    # Parse both known (custom) and unknown (ROS 2) arguments
    known_args, _ = parser.parse_known_args()

    # Initialize ROS with the full argument list
    rclpy.init(args=sys.argv)

    # Instantiate your environment node using the parsed arguments
    node = LinearEnvSimVizNode(
        freq=known_args.freq,
        frame_id=known_args.frame_id
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down LinearEnvSimVizNode.")
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
