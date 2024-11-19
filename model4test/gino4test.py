import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from transformers import BertTokenizer
BERT_PATH = './uncased_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
#注意确定文件位置
model = load_model( "pretrained/groundingdino_swint_ogc.pth")#"groundingdino\config\GroundingDINO_SwinT_OGC.py",
IMAGE_PATH = "2_ramtestfig.jpg"
res = ['TABLE']
TEXT_PROMPT = res[0]
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)

pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

plt.imshow(pil_image)
plt.show()