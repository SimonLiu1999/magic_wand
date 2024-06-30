import smbus2
import time
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# BMX160 的 I2C 地址
BMX160_I2C_ADDR = 0x68

# BMX160 寄存器地址
BMX160_ACCEL_X_LSB = 0x12
BMX160_ACCEL_X_MSB = 0x13
BMX160_ACCEL_Y_LSB = 0x14
BMX160_ACCEL_Y_MSB = 0x15
BMX160_ACCEL_Z_LSB = 0x16
BMX160_ACCEL_Z_MSB = 0x17
BMX160_CMD_REG = 0x7E
BMX160_ACCEL_CONFIG = 0x40
BMX160_ACCEL_RANGE = 0x41

# 初始化 I2C 总线
bus = smbus2.SMBus(1)

def init_bmx160():
    # 复位 BMX160
    bus.write_byte_data(BMX160_I2C_ADDR, BMX160_CMD_REG, 0xB6)
    time.sleep(0.1)


    # 复位 BMX160
    bus.write_byte_data(BMX160_I2C_ADDR, BMX160_CMD_REG, 0x11)
    time.sleep(0.1)

    # 配置加速度计范围为 ±2G
    bus.write_byte_data(BMX160_I2C_ADDR, BMX160_ACCEL_RANGE, 0x03)
    time.sleep(0.1)

    # 配置加速度计输出数据率和带宽
    bus.write_byte_data(BMX160_I2C_ADDR, BMX160_ACCEL_CONFIG, 0x28)
    time.sleep(0.1)

# 初始化 BMX160
init_bmx160()

def read_accel_data():
    # 读取加速度数据
    data = bus.read_i2c_block_data(BMX160_I2C_ADDR, BMX160_ACCEL_X_LSB, 6)

    # 将数据转换为 16 位有符号整数
    accel_x = (data[1] << 8) | data[0]
    accel_y = (data[3] << 8) | data[2]
    accel_z = (data[5] << 8) | data[4]

    # 处理负值
    if accel_x > 32767:
        accel_x -= 65536
    if accel_y > 32767:
        accel_y -= 65536
    if accel_z > 32767:
        accel_z -= 65536

    return accel_x, accel_y, accel_z

def calculate_angles(accel_x, accel_y, accel_z):
    # 将加速度转换为 m/s^2
    accel_x = accel_x / 16384.0
    accel_y = accel_y / 16384.0
    accel_z = accel_z / 16384.0
    
    pitch = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)) * 180.0 / math.pi
    roll = math.atan2(-accel_x, accel_z) * 180.0 / math.pi
    print(pitch,roll)
    return pitch, roll

# 定义细长长方体的顶点
vertices = np.array([[-10, -1, -1],
                     [ 10, -1, -1],
                     [ 10,  1, -1],
                     [-10,  1, -1],
                     [-10, -1,  1],
                     [ 10, -1,  1],
                     [ 10,  1,  1],
                     [-10,  1,  1]])

# 定义长方体的面
faces = [[0, 1, 5, 4],
         [1, 2, 6, 5],
         [2, 3, 7, 6],
         [3, 0, 4, 7],
         [0, 1, 2, 3],
         [4, 5, 6, 7]]

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建 Poly3DCollection 对象
poly3d = [vertices[face] for face in faces]
collection = Poly3DCollection(poly3d, alpha=0.25, linewidths=1, edgecolors='r')
ax.add_collection3d(collection)

# 设置轴的限制
ax.set_xlim([-12, 12])
ax.set_ylim([-12, 12])
ax.set_zlim([-12, 12])

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置各轴的比例相同
ax.set_box_aspect([1, 1, 1])

def update(num):
    # 读取加速度数据
    accel_x, accel_y, accel_z = read_accel_data()
    
    # 计算姿态角
    pitch, roll = calculate_angles(accel_x, accel_y, accel_z)
    
    # 计算旋转矩阵
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                   [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
    
    Ry = np.array([[np.cos(roll_rad), 0, np.sin(roll_rad)],
                   [0, 1, 0],
                   [-np.sin(roll_rad), 0, np.cos(roll_rad)]])
    
    # 总旋转矩阵
    R = np.dot(Ry, Rx)
    
    # 旋转长方体的顶点
    rotated_vertices = np.dot(vertices, R.T)
    
    # 更新 Poly3DCollection 对象
    poly3d = [rotated_vertices[face] for face in faces]
    collection.set_verts(poly3d)
    
    return collection,

# 创建动画
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

# 显示图形
plt.show()