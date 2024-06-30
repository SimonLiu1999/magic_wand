# magic_wand
a magic wand using BMX160

# 目前开源了以下文件：
bmx160.py 可以读取BMX160的数据，用于检测连接是否正确

euler_angle.py 通过BMX160的数据计算欧拉角，并显示出魔杖实时的姿态

collect_data.py 可以收集动作数据，用于数据集标注

train_model.py 使用收集到的动作数据训练一个三层卷积神经网络并保存用于后续识别

employ.py 将训练好的神经网络读取并部署，实现手势检测

# 本项目还在持续更新中，欢迎加入QQ群关注：210150550