#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Vector3

import sys
import argparse


class LinearEnvSimVizNode(Node):
    def __init__(self, freq=50.0, frame_id='world'):
        super().__init__('linear_env_sim_viz')

        # Simulation parameters
        self.mass = 50.0       # kg
        self.b = 0.1           # N/(mm/s)
        self.L = 300.0         # mm
        self.T = 5.0           # s
        self.freq = freq
        self.dt = 1.0 / freq
        self.frame_id = frame_id
        self.Kp = 0.1          # "Human" spring gain

        # ROS subscriptions and publications
        self.trigger_sub = self.create_subscription(
            Int32, '/trigger', self.trigger_callback, 10
        )

        self.ref_force_sub = self.create_subscription(
            Int32, '/ref_force', self.ref_force_callback, 10
        )

        self.mes_pos_pub = self.create_publisher(Int32, '/mes_pos', 10)
        self.ref_pos_pub = self.create_publisher(Int32, '/ref_pos', 10)

        # For RViz visualization
        self.mes_marker_pub = self.create_publisher(Marker, '/mes_marker', 10)
        self.ref_marker_pub = self.create_publisher(Marker, '/ref_marker', 10)

        # Internal state
        self.trigger_state = 0
        self.simulation_active = False
        self.cable_force = 0.0
        self.x = 0.0
        self.x_dot = 0.0
        self.time_sim = 0.0

        self.reset_simulation()
        self.timer = self.create_timer(self.dt, self.update_and_visualize)
        self.get_logger().info(f"LinearEnvSimVizNode started at {freq} Hz.")

    def reset_simulation(self, go_to_start=False):
        self.time_sim = 0.0
        self.cable_force = 0.0
        self.x_dot = 0.0
        if go_to_start:
            self.x = 0.0
            self.x_dot = 0.0
        self.simulation_active = False

    def trigger_callback(self, msg: Int32):
        new_trigger = msg.data
        if new_trigger == 1 and self.trigger_state == 0:
            self.get_logger().info("Trigger 0->1: Starting simulation.")
            self.simulation_active = True

        elif new_trigger == 0 and self.trigger_state == 1:
            self.get_logger().info("Trigger 1->0: Resetting to start position.")
            self.reset_simulation(go_to_start=True)
            self.publish_marker(self.mes_marker_pub, self.x / 1000.0, 0.05, (1.0, 0.0, 0.0))
            self.publish_marker(self.ref_marker_pub, self.x / 1000.0, 0.05, (0.0, 1.0, 0.0))

        self.trigger_state = new_trigger

    def ref_force_callback(self, msg: Int32):
        self.cable_force = float(msg.data)
        self.get_logger().info(f"Updated cable force to {self.cable_force:.2f} N")

    def update_and_visualize(self):
        if not self.simulation_active:
            return

        # Compute reference in mm
        x_ref = self.target_trajectory(self.time_sim)

        # "Human" spring force
        human_force = self.Kp * (x_ref - self.x)

        # Net force = cable_force - b*x_dot + human_force
        net_force = self.cable_force - (self.b * self.x_dot) + human_force

        # Acceleration in mm/s^2
        x_ddot = 1000.0 * (net_force / self.mass)

        # Integrate
        self.x_dot += x_ddot * self.dt
        self.x += self.x_dot * self.dt

        # Clamp
        if self.x < 0.0:
            self.x = 0.0
            self.x_dot = 0.0
        elif self.x > self.L:
            self.x = self.L
            self.x_dot = 0.0

        self.time_sim += self.dt

        # Stop after T
        if self.time_sim >= self.T:
            self.simulation_active = False
            self.get_logger().info("Simulation complete. Waiting for next trigger.")

        # Publish mes_pos and ref_pos
        self.mes_pos_pub.publish(Int32(data=int(self.x)))
        self.ref_pos_pub.publish(Int32(data=int(x_ref)))

        # Publish markers in meters
        self.publish_marker(self.mes_marker_pub, self.x / 1000.0, 0.05, (1.0, 0.0, 0.0))
        self.publish_marker(self.ref_marker_pub, x_ref / 1000.0, 0.05, (0.0, 1.0, 0.0))

    def publish_marker(self, topic, position_m, scale_m, color):
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

        marker.scale = Vector3(x=scale_m, y=scale_m, z=scale_m)

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        topic.publish(marker)

    def target_trajectory(self, t):
        if t >= self.T:
            return self.L
        tau = t / self.T
        return self.L * (10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)


def main(args=None):
    parser = argparse.ArgumentParser(description="1D Linear Environment with Visualization in RViz")
    parser.add_argument("--freq", type=float, default=50.0, help="Simulation frequency")
    parser.add_argument("--frame-id", type=str, default="world", help="Frame ID for markers")

    known_args, _ = parser.parse_known_args()
    rclpy.init(args=sys.argv)

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
