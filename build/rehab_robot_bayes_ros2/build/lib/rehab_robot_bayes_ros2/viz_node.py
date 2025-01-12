#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Vector3

import sys
import argparse


class LinearVizNode(Node):
    """
    A ROS2 node that:
      - Subscribes to /mes_pos and /ref_pos (in mm).
      - Visualizes these two positions in RViz (converted to meters).
    """

    def __init__(self, freq=10.0, frame_id='world'):
        super().__init__('linear_viz_node')

        # Frequency (Hz) at which we publish the markers
        self.freq = freq
        self.dt = 1.0 / freq
        self.frame_id = frame_id

        # Internal storage of positions (in mm)
        self.mes_pos = None
        self.ref_pos = None

        # ROS subscriptions
        self.mes_pos_sub = self.create_subscription(
            Int32,
            '/mes_pos',
            self.mes_pos_callback,
            10
        )
        self.ref_pos_sub = self.create_subscription(
            Int32,
            '/ref_pos',
            self.ref_pos_callback,
            10
        )

        # ROS publishers for RViz markers
        self.mes_marker_pub = self.create_publisher(Marker, '/mes_marker', 10)
        self.ref_marker_pub = self.create_publisher(Marker, '/ref_marker', 10)

        # Create a timer to periodically update RViz
        self.timer = self.create_timer(self.dt, self.update_markers)

        self.get_logger().info(
            f"LinearVizNode started at {freq} Hz, listening to /mes_pos and /ref_pos."
        )

    def mes_pos_callback(self, msg: Int32):
        """Store the latest measured position (in mm)."""
        self.mes_pos = float(msg.data)

    def ref_pos_callback(self, msg: Int32):
        """Store the latest reference position (in mm)."""
        self.ref_pos = float(msg.data)

    def update_markers(self):
        """
        Publish two spheres in RViz:
          - Red sphere at mes_pos
          - Green sphere at ref_pos
        Positions must be converted from mm to meters for RViz.
        """
        # We only publish markers if both positions are known
        if self.mes_pos is None or self.ref_pos is None:
            return

        # Convert mm -> m
        mes_m = self.mes_pos / 1000.0
        ref_m = self.ref_pos / 1000.0

        # Publish the measured-position (red) marker
        self.publish_marker(
            publisher=self.mes_marker_pub,
            position_m=mes_m,
            color=(1.0, 0.0, 0.0),  # red
            marker_id=1,
            ns='mes_ns'
        )

        # Publish the reference-position (green) marker
        self.publish_marker(
            publisher=self.ref_marker_pub,
            position_m=ref_m,
            color=(0.0, 1.0, 0.0),  # green
            marker_id=2,
            ns='ref_ns'
        )

    def publish_marker(self, publisher, position_m, color, marker_id, ns):
        """
        Helper to publish a sphere Marker in RViz.
        :param publisher: Marker publisher
        :param position_m: float, X position in meters
        :param color: tuple (r, g, b)
        :param marker_id: int, unique ID for this marker
        :param ns: str, namespace for this marker
        """
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = position_m
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Scale (in meters)
        marker.scale = Vector3(x=0.05, y=0.05, z=0.05)

        # Color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # fully opaque

        publisher.publish(marker)


def main(args=None):
    parser = argparse.ArgumentParser(description="Minimal Node for Visualizing mes_pos and ref_pos in RViz")
    parser.add_argument("--freq", type=float, default=10.0, help="Marker update frequency in Hz")
    parser.add_argument("--frame-id", type=str, default="world", help="Frame ID for the markers in RViz")

    known_args, _ = parser.parse_known_args()

    rclpy.init(args=sys.argv)

    node = LinearVizNode(
        freq=known_args.freq,
        frame_id=known_args.frame_id
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down LinearVizNode.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
