#初始化无人机大模型
import ernie_airsim
import threading
import time
example_msg = [
            {
                "role": "user",
                "content": "向前移动 1 m"
            },
            {
                "role": "assistant",
                "content": """```python
        tello.forward(1)
        ```
        此代码使用“forward()”函数将无人机移动到距当前位置 1m的新位置。 """}
        ]
my_ernie_airsim = ernie_airsim.ErnieAirSim(system_prompts='system_prompts/airsim_objectnavigation_cn.txt', prompt='prompts/try.txt',example_msg=example_msg)

#初始化无人机sdk
from tello_wrapper_ob import *
tello = TelloWrapper(my_ernie_airsim)
## 开一个线程，持续运行 tello.get_state()，更新无人机状态
def update_state():
    while True:
        tello.get_state()
        time.sleep(3)  # 状态更新间隔
        # # 按下ctrl+c退出时，关闭线程
        # if tello.is_stop_thread:
        #     break


# 开启状态更新线程
state_thread = threading.Thread(target=update_state, daemon=True)
state_thread.start()


while True:
    command = input("请输入指令：")
    python_code, response = my_ernie_airsim.process(command)
    exec(python_code)

    
# command list
# 起飞
# 降落
# 向前移动 1 m
# 起飞，并请告诉我图片中有什么
# 请告诉我图片中所有物体的具体位置
# 找到门到并飞向它，之后你可以再向前移动 1 m，这样就可以穿过门了
# 我想吃水果，帮我选择它，然后飞到它的位置
