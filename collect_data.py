import smbus2
import time
import numpy as np
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

        return (accel_x, accel_y, accel_z), (gyro_x, gyro_y, gyro_z)
    except Exception as e:
        print(f"读取传感器数据失败: {e}")
        return None, None

def collect_motion_data(sample_rate_hz=50, num_samples=128):
    data = []

    for _ in range(num_samples):
        accel, gyro = read_sensor_data()
        if accel and gyro:
            # 将加速度和角速度合并为六维向量
            data_point = accel + gyro
            data.append(data_point)
        else:
            print("无法读取传感器数据")

        # 等待下一个采样时间
        time.sleep(1.0 / sample_rate_hz)

    return np.array(data)

def save_data(gesture, data, output_dir="gesture_data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{gesture}_{int(time.time())}.npy"
    filepath = os.path.join(output_dir, filename)
    
    np.save(filepath, data)
    
    print(f"数据已保存到 {filepath}")

def main():
    # 初始化传感器
    init_bmx160()

    while True:
        print("请选择一个手势进行数据采集：")
        for i, gesture in enumerate(GESTURES):
            print(f"{i + 1}. {gesture}")
        
        choice = input("输入手势编号 (或输入 'q' 退出): ").strip()
        
        if choice.lower() == 'q':
            print("退出程序")
            break
        
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(GESTURES):
            print("无效输入，请重新选择")
            continue
        
        gesture = GESTURES[int(choice) - 1]
        print(f"开始采集手势: {gesture}")

        # 收集动作数据
        motion_data = collect_motion_data()

        # 保存数据
        save_data(gesture, motion_data)

        print(f"{gesture} 手势数据采集完成\n")

if __name__ == "__main__":
    main()