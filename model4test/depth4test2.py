

from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image



# 加载本地图像
img_path = "1.jpg"
image = Image.open(img_path)

# 加载本地模型
processor = GLPNImageProcessor.from_pretrained("./glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("./glpn-nyu")

model = model.to("cuda")


# 准备图像以供模型使用
inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():# 关闭梯度计算
    #时间戳
    import time
    start = time.time()
    outputs = model(**inputs)
    end = time.time()
    print("Time: ", end - start)
    predicted_depth = outputs.predicted_depth # 每个像素点距离摄像头的距离

# 插值到原始大小
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# 可视化预测结果
output = prediction.squeeze().cpu().numpy() # 深度图
formatted = (output * 255 / np.max(output)).astype("uint8") # 格式化深度图
depth = Image.fromarray(formatted) # 深度图

# 每个像素点距离摄像头的距离
print(predicted_depth)

# 保存或显示深度图
depth.save("depth_map1.png")
depth.show()