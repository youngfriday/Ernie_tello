# 这段代码主要是由文心一言生成的，按照教程中是问答方式即可得到，但TELLO往往坚持不到最后自动执行这段代码，会自动断联，尝试使用官方心跳包，但显示消息无效
# 所以当TELLO断联时，需要手动执行文心一言返回的代码，即可找到目标物体

import threading
import time
#初始化无人机sdk
from tello2 import *
tello = TelloWrapper()
## 开一个线程，持续运行 tello.get_state()，这里写了心跳包，但无效
# def update_state():
#     while True:
#         tello.get_state()
#         time.sleep(5)  # 状态更新间隔
       
# # 开启状态更新线程
# state_thread = threading.Thread(target=update_state, daemon=True)
# state_thread.start()
# 起飞无人机
tello.takeoff()
target = "apple"
while True:
    # 获取当前视野中的物体列表
    obj_name_list = tello.get_objects()
    #打印物体列表
    print(obj_name_list) 
    # 检查"target"是否在物体列表中
    if target in obj_name_list:
        # 如果找到"apple"，获取其位置信息
        final_result = tello.ob_objects_llm(target)
        #打印位置信息
        print(final_result)
        print("找到了taget")

        # 遍历位置信息，找到"target"的距离和角度
        for obj_name, distance, degrees in final_result:
            if obj_name == target:
                # 根据得到的角度调整无人机的方向
                tello.turn(degrees)

                # 根据得到的距离让无人机前进到目标位置
                tello.forward(distance)

                # 找到并飞向"target"后，退出循环
                break
        break
    else:
        # 如果没有找到"target"，则让无人机旋转90度后继续搜索
        tello.turn(55)
        print("未找到target")

# 如果需要，可以在此处添加降落无人机的命令
tello.land()