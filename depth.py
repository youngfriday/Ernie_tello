# 这是
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#使用GPU

# 加载本地模型
processor = GLPNImageProcessor.from_pretrained("./glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("./glpn-nyu")

def estimate_depth(image):
        #把图像传到GPU
        # image = image.to(device)

        # 加载本地图像
        # image = Image.open(image_path)
        # 准备图像以供模型使用
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():  # 关闭梯度计算
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # 每个像素点距离摄像头的距离

        # 插值到原始大小
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        return predicted_depth