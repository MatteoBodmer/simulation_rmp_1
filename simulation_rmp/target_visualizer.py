import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

class TargetVisualizer(Node):
    def __init__(self):
        super().__init__('target_visualizer')
        self.marker_pub = self.create_publisher(Marker, '/target_marker', 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/target_pose_visualization', self.pose_callback, 10)
        self.get_logger().info("Target visualizer started")

    def pose_callback(self, msg):
        marker = Marker()
        marker.header = msg.header
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = msg.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.id = 0
        marker.ns = "target_position"
        self.marker_pub.publish(marker)
        
        # Also publish an arrow for orientation
        arrow_marker = Marker()
        arrow_marker.header = msg.header
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.pose = msg.pose
        arrow_marker.scale.x = 0.15  # Length
        arrow_marker.scale.y = 0.02  # Width
        arrow_marker.scale.z = 0.02  # Height
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 0.8
        arrow_marker.id = 1
        arrow_marker.ns = "target_orientation"
        self.marker_pub.publish(arrow_marker)

def main():
    rclpy.init()
    node = TargetVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()