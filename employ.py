import smbus2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# BMX160 的 I2C 地址
BMX160_I2C_ADDR = 0x68

# 初始化 I2C 总线
bus = smbus2.SMBus(1)

# 常量
GRAVITY = 9.80665  # 重力加速度，单位：m/s^2
ACCEL_SCALE_MODIFIER = 16384.0  # 对于加速度计 ±2G 范围
GYRO_SCALE_MODIFIER = 16.4  # 对于陀螺仪 ±2000°/s 范围

# 手势类型
GESTURES = ["triangle", "circle", "cross"]

def init_bmx160():
    try:
        # 复位 BMX160
        bus.write_byte_data(BMX160_I2C_ADDR, 0x7E, 0xB6)
        time.sleep(0.1)

        # 设置加速度计为正常模式
        bus.write_byte_data(BMX160_I2C_ADDR, 0x7E, 0x11)  # 加速度计正常模式
        time.sleep(0.1)

        # 设置陀螺仪为正常模式
        bus.write_byte_data(BMX160_I2C_ADDR, 0x7E, 0x15)  # 陀螺仪正常模式
        time.sleep(0.1)

        # 配置加速度计范围为 ±2G
        bus.write_byte_data(BMX160_I2C_ADDR, 0x41, 0x03)
        time.sleep(0.1)

        # 配置加速度计输出数据率和带宽
        bus.write_byte_data(BMX160_I2C_ADDR, 0x40, 0x28)
        time.sleep(0.1)

        # 配置陀螺仪范围为 ±2000DPS
        bus.write_byte_data(BMX160_I2C_ADDR, 0x43, 0x00)
        time.sleep(0.1)

        # 配置陀螺仪输出数据率和带宽
        bus.write_byte_data(BMX160_I2C_ADDR, 0x42, 0x28)
        time.sleep(0.1)

        print("传感器初始化成功")
    except Exception as e:
        print(f"传感器初始化失败: {e}")

def read_sensor_data():
    try:
        # 读取加速度数据
        accel_data = bus.read_i2c_block_data(BMX160_I2C_ADDR, 0x12, 6)
        # 读取陀螺仪数据
        gyro_data = bus.read_i2c_block_data(BMX160_I2C_ADDR, 0x0C, 6)

        # 处理加速度数据
        accel_x = (accel_data[1] << 8) | accel_data[0]
        accel_y = (accel_data[3] << 8) | accel_data[2]
        accel_z = (accel_data[5] << 8) | accel_data[4]

        if accel_x > 32767:
            accel_x -= 65536
        if accel_y > 32767:
            accel_y -= 65536
        if accel_z > 32767:
            accel_z -= 65536

        # 转换加速度为 m/s^2
        accel_x = (accel_x / ACCEL_SCALE_MODIFIER) * GRAVITY
        accel_y = (accel_y / ACCEL_SCALE_MODIFIER) * GRAVITY
        accel_z = (accel_z / ACCEL_SCALE_MODIFIER) * GRAVITY

        # 处理陀螺仪数据
        gyro_x = (gyro_data[1] << 8) | gyro_data[0]
        gyro_y = (gyro_data[3] << 8) | gyro_data[2]
        gyro_z = (gyro_data[5] << 8) | gyro_data[4]

        if gyro_x > 32767:
            gyro_x -= 65536
        if gyro_y > 32767:
            gyro_y -= 65536
        if gyro_z > 32767:
            gyro_z -= 65536

        # 转换角速度为 °/s
        gyro_x = gyro_x / GYRO_SCALE_MODIFIER
        gyro_y = gyro_y / GYRO_SCALE_MODIFIER
        gyro_z = gyro_z / GYRO_SCALE_MODIFIER

        return [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    except Exception as e:
        print(f"传感器数据读取失败: {e}")
        return None

# 定义手势检测模型
class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 128, 128)
        self.fc2 = nn.Linear(128, len(GESTURES))
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将形状 (batch_size, 128, 6) 转换为 (batch_size, 6, 128)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def detect_gesture(model, data):
    model.eval()
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # 添加 batch 维度
        output = model(data)
        _, predicted = torch.max(output, 1)
        return GESTURES[predicted.item()]

def main():
    # 初始化传感器
    init_bmx160()

    # 加载训练好的模型
    model = GestureCNN()
    model.load_state_dict(torch.load('gesture_cnn.pth'))
    model.eval()
    print("模型加载成功")

    while True:
        input("按下 Enter 键开始手势检测...")

        # 读取 128 个数据点
        data = []
        for _ in range(128):
            sensor_data = read_sensor_data()
            if sensor_data:
                data.append(sensor_data)
            time.sleep(1/50.0)  # 50Hz 读取数据

        if len(data) == 128:
            gesture = detect_gesture(model, data)
            print(f"检测到的手势: {gesture}")
        else:
            print("数据采集失败，请重试")

if __name__ == "__main__":
    main()