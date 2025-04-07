import time
from pymycobot import MyCobot280
# 默认使用9000端口
#其中"172.20.10.14"为机械臂IP，请自行输入你的机械臂IP
mc = MyCobot280('COM7', 1000000)

# angels = mc.get_angles()

# 标定姿态
angels = [-83.58, -52.73, 150, -99.05, 5.36, -2.81] 
# 原点姿态
angels_0 = [0, 0, 0, 0, 0, 0]

# mc.send_angles(angels_0, 50)

# time.sleep(2)
mc.send_angles(angels, 50)
time.sleep(2)










