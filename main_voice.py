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
my_ernie_airsim = ernie_airsim.ErnieAirSim(system_prompts='system_prompts/my_system_prompt.txt', prompt='prompts/try.txt',example_msg=example_msg)

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



import pyaudio
import vosk
import json
import time
from langchain_community.llms import QianfanLLMEndpoint
import os

os.environ["QIANFAN_AK"] = 'PTZ99RaR29OWrKSgHqtsto1F'
os.environ["QIANFAN_SK"] = 'R1EpnazaRzyjp5KFnHLX23BPM0hTm3WC'
llm = QianfanLLMEndpoint(temperature=0.9)
# 加载 VOSK 模型
model = vosk.Model("vosk-model-small-cn-0.22")  # 替换为模型文件夹的实际路径

flag = 0  # 录音结束标志
history = []


while (1):
    # 配置音频参数
    RATE = 16000  # 采样率
    CHUNK = 8000  # 缓冲区大小
    RECORD_SECONDS = 8  # 录音时长（秒）

    results = []
    result = []
    # 初始化 PyAudio
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    # 实时识别
    rec = vosk.KaldiRecognizer(model, RATE)
    print("开始实时语音识别，每次录音 5 秒，请讲话...")
    try:
        frames = []
        # 录制 5 秒音频
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # 处理录制的音频数据
        audio_data = b''.join(frames)
        if rec.AcceptWaveform(audio_data):
            result = json.loads(rec.Result())
            results.append(result['text'])
            history.append(("human", result['text']))
            if '结束' in result['text']:
                flag = 1

            else:
                print(rec.PartialResult())

            # 可选：输出识别中间结果的延迟
        time.sleep(0.1)

    except KeyboardInterrupt:
        print("停止实时语音识别")

    finally:
        # 结束流并释放资源
        stream.stop_stream()
        stream.close()
        audio.terminate()
    command = result
    result = result['text']
    print()
    print()
    print("指令为：", result)
    print()
    print()
    python_code, response = my_ernie_airsim.process(result)
    exec(python_code)

    
# command list
# 起飞
# 降落
# 向前移动 1 m
# 起飞，并请告诉我图片中有什么
# 请告诉我图片中所有物体的具体位置
# 找到门到并飞向它，之后你可以再向前移动 1 m，这样就可以穿过门了
# 我想吃水果，帮我选择它，然后飞到它的位置
