import smbus2
import time

# 设置I2C总线
bus = smbus2.SMBus(1)
DEVICE_ADDRESS = 0x68  # BMX160的I2C地址

# BMX160寄存器地址
ACC_X_LSB_REG = 0x12
ACC_X_MSB_REG = 0x13
ACC_Y_LSB_REG = 0x14
ACC_Y_MSB_REG = 0x15
ACC_Z_LSB_REG = 0x16
ACC_Z_MSB_REG = 0x17
CMD_REG = 0x7E

# 初始化BMX160
def init_bmx160():
    bus.write_byte_data(DEVICE_ADDRESS, CMD_REG, 0x11)  # 设置加速度计为正常模式
    time.sleep(0.1)

def twos_complement(val, bits):
    """计算二进制补码"""
    if val & (1 << (bits - 1)):
        val -= 1 << bits
    return val

def read_bmx160():
    data_x = bus.read_i2c_block_data(DEVICE_ADDRESS, ACC_X_LSB_REG, 2)
    data_y = bus.read_i2c_block_data(DEVICE_ADDRESS, ACC_Y_LSB_REG, 2)
    data_z = bus.read_i2c_block_data(DEVICE_ADDRESS, ACC_Z_LSB_REG, 2)

    acc_x = (data_x[1] << 8) | data_x[0]
    acc_y = (data_y[1] << 8) | data_y[0]
    acc_z = (data_z[1] << 8) | data_z[0]

    acc_x = twos_complement(acc_x, 16)
    acc_y = twos_complement(acc_y, 16)
    acc_z = twos_complement(acc_z, 16)

    # 转换为g单位
    scale_factor = 2 / 32768.0
    acc_x *= scale_factor
    acc_y *= scale_factor
    acc_z *= scale_factor

    return acc_x, acc_y, acc_z

if __name__ == "__main__":
    init_bmx160()
    while True:
        acc_x, acc_y, acc_z = read_bmx160()
        print(f"Accelerometer Data: X={acc_x:.3f}g, Y={acc_y:.3f}g, Z={acc_z:.3f}g")
        time.sleep(1)