import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
import transformers
from transformers import BertTokenizer
BERT_PATH = './uncased_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
# bert = BertModel.from_pretrained(BERT_PATH)#这个模型是给出图片中的所有物品名称

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image = "2_ramtestfig.jpg" 


# im = Image.open(image)
# plt.imshow(im)
# plt.show()

#这个模型就是给出图片中的所有物品名称：

#######load model,模型自行确定下载的位置
model = ram_plus(pretrained='./pretrained/ram_plus_swin_large_14m.pth',
                         image_size=384,
                         vit='swin_l')
model.eval()

model = model.to(device)
# 定义transform
transform = get_transform(image_size=384)

image = transform(Image.open(image)).unsqueeze(0).to(device) #把图片转换成tensor

res = inference(image, model)
print("Image Tags: ", res[0])
print("图像标签: ", res[1])



# # 下面是nino的模型，这个模型是给出图片中的物体的位置
# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import cv2
# tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)

# #注意确定文件位置
# model = load_model("./GroundingDINO/config/GroundingDINO_SwinT_OGC.py", "./pretrained/groundingdino_swint_ogc.pth")
# IMAGE_PATH = "2_ramtestfig.jpg"
# TEXT_PROMPT = res[0]
# BOX_TRESHOLD = 0.35
# TEXT_TRESHOLD = 0.25

# image_source, image = load_image(IMAGE_PATH)

# boxes, logits, phrases = predict(
#     model=model,
#     image=image,
#     caption=TEXT_PROMPT,
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )

# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_image.jpg", annotated_frame)

# pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

# plt.imshow(pil_image)
# plt.show()