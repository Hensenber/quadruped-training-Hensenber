"""
本步需要三个导入：
内容	             来源	                     用途
rclpy	            rclpy	                    初始化、处理回调和关闭 ROS 2
Node	            rclpy.node	                作为你的节点类的父类
MotorCommandArray	quadruped_interfaces.msg	接收整组电机指令
"""
from rclpy import rclpy     #初始化、处理回调和关闭 ROS 2
from rclpy.node import Node #作为你的节点类的父类
from quadruped_interfaces.msg import MotorCommandArray #接收整组电机指令
