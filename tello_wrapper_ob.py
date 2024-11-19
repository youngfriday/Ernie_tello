# -*- coding: utf-8 -*-
# @Time    : 2023/12/23  22:24
# @Author  : mariswang@rflysim
# @File    : tello_wrapper.py
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-

import sys
import threading
import time


from djitellopy import Tello
import math
import numpy as np
import io
import cv2

import random
import torch
from torchvision.ops import box_convert
from PIL import Image
sys.path.append('./recognize-anything')
from ram.models import ram_plus
# import ram.models as ram_models
from ram import inference_ram as inference
from ram import get_transform
from transformers import BertTokenizer
BERT_PATH = './uncased_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)

sys.path.append('./GroundingDINO/groundingdino')
from groundingdino.util.inference import load_model, load_image, predict, annotate

from transformers import pipeline
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import depth

BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25

class TelloWrapper:
    def __init__(self, llm=None):
        self.llm = llm
        
        self.client = Tello()
        self.client.connect()  # 连接tello

        #启动视频流
        self.client.streamon()  # 开启视频传输
        t = threading.Thread(target=self.get_stream)
        t.setDaemon(True)
        t.start()

        self.head_img = None #tello 前置摄像头，图像初始化
        self.final_result = None #最终结果
        self.obj_name_list = None #目标名称列表
        self.depth = depth
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#使用GPU
        print(torch.cuda.is_available())
        #输出当前设备
        print("device:",device)
        #强制使用GPU
        # device = torch.device('cuda')
        
         #目标识别
        ram_model = ram_plus(pretrained='./pretrained/ram_plus_swin_large_14m.pth',
                         image_size=384,
                         vit='swin_l')
        ram_model.eval()#模型设为评估模式


        self.device = device
        self.ram_model = ram_model.to(device)#模型加载到设备上
    
        #目标检测
        self.dino_model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "E:/air/pretrained/groundingdino_swint_ogc.pth")
        self.dino_model = self.dino_model.to(device)
        
        #深度视觉估算
        # checkpoint = "./glpn-nyu"
        # self.depth_estimator = pipeline("depth-estimation", model=checkpoint)
        # self.depth_estimator=depth.estimate_depth() #深度视觉估算

    def get_stream(self):
        while True:
            self.head_img = self.client.get_frame_read().frame
            self.head_img = cv2.cvtColor(self.head_img, cv2.COLOR_BGR2RGB)
            time.sleep(0.01)

    def takeoff(self):
        self.client.takeoff()

    def land(self):
        self.client.land()

    # def turn_left(self,degree=90):
    #     """
    #     左转degree度
    #     :return:
    #     """
    #     degree = 10*degree #tello base is 1度
    #     self.client.rotate_counter_clockwise(int(degree))
    #
    # def turn_right(self,degree=90):
    #
    #     """
    #     右转degree度
    #     :return:
    #     """
    #     degree = 10*degree #tello base is 1度
    #     self.client.rotate_clockwise(int(degree))

    def turn(self, degree):
        self.client.rotate_clockwise(int(degree))
            
    def forward(self, distance):
        """
        向前移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = int(distance*100) # tello base is 1cm
        self.client.move_forward(distance) #向前移动50cm

    def back(self, distance):
        """
        向后移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = int(distance*100) # tello base is 1cm
        self.client.move_back(distance) #向前移动50cm

    def up(self, distance):
        """
        向上移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = int(distance*100) # tello base is 1cm
        self.client.move_up(distance) #向前移动50cm

    def down(self, distance):
        """
        向下移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = int(distance*100) # tello base is 1cm
        self.client.move_down(distance) #向前移动50cm
        
    def get_image(self):
        """
        获得前置摄像头渲染图像
        :return:
        """
        return self.head_img

    def get_drone_state(self):
        """
        获得无人机状态,
        :return:{'pitch': int, 'roll': int, 'yaw': int}
        """
        return self.client.query_attitude()

    def connect(self):
        self.client.connect()


    def get_depth_estimator(self, img):
        """
        在图像 img 上运行深度视觉预估
        :param img:cv2的图片
        :return:predicted_depth # 图片上像素点距离无人机的距离
        """
        img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #这是一个pil图片
        img_pil = Image.fromarray(img_pil)
        predictions = self.depth.estimate_depth(img_pil)

        return predictions    #["predicted_depth"]
        
    def get_objects(self):
        # """
        # 在图像 img 上运行对象检测模型
        # :param img:
        # :return:obj_list # 图片上的目标名称列表
        # """
        #目标检测
        #opencv图片转bytes,pil可直接读取
        img = self.head_img
        
        imgbytes = cv2.imencode(".jpg", img)[1].tobytes()
        byte_stream = io.BytesIO(imgbytes)
        
        transform = get_transform(image_size=384)
        image = transform(Image.open(byte_stream)).unsqueeze(0).to(self.device)#转换为tensor++
        res = inference(image, self.ram_model)
        obj_name_list = res[0].split(" | ") #res[0] 英文，res[1]中文
        
        self.obj_name_list = obj_name_list
        
        prompt = "请告诉我图里有以下物体" + res[1]
        self.llm.ask(prompt)
        return obj_name_list
    
    def get_state(self) :
        """获取当前状态
        返回值:
        
        """
        # self.client.send_keepalive()
        return self.client.get_current_state()
    
    def ob_objects(self,obj_name_list):
        # """
        # 注意需要先执行get_image，
        # 在图像 img 上运行对象检测模型，获得目标列表 [ <对象名称、距离、角度（以度为单位）>,...]
        # :return:对象名称列表、对象信息列表、bbox图
        # """

        TEXT_PROMPT = " | ".join(obj_name_list)
        #目标检测
        imgbytes = cv2.imencode(".jpg", self.head_img)[1].tobytes()
        byte_stream = io.BytesIO(imgbytes)
        
    
        image_source, image = load_image(byte_stream)
        # 把图片加载到GPU上
        image = image.to(self.device)
        
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image, 
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        h, w, _ = image_source.shape
        boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        
        #[xmin, ymin, xmax, ymax]
        obj_locs = boxes_xyxy

        #深度预测
        img_camera_distance = self.get_depth_estimator(self.head_img)  #相机距离

        
        final_obj_list = [] #最终结果列表
        #构建目标结果
        index = 0
        for bbox in obj_locs:
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            # 去掉第一个维度，使 img_camera_distance 成为二维数组
            img_camera_distance = img_camera_distance.squeeze()

            # 然后再进行索引访问
            camera_distance = img_camera_distance[center_y, center_x]#相机距离
            # depth_distance = img_camera_distance[h//2, w//2] #相平面距离
                     # 目标物体相对于相机中心的水平偏移角度

            # if depth_distance / camera_distance>1:
            #     print(depth_distance / camera_distance)
            #     obj_name =  phrases[index]#获得目标名称，可能有多个
            #     angle = 0
            #     angle_degree = angle  # 以度为单位
                
            #     obj_info = (obj_name, camera_distance, angle_degree)
            #     final_obj_list.append(obj_info)
            #     index = index + 1
            #     continue
            
            #     # if depth_distance / camera_distance>1.2:
            #     #     raise ValueError("depth_distance / camera_distance>1.1")
            
            #求角度
            angle = (center_x - (w / 2)) / w * 60
            angle_degree = angle  # 以度为单位

            # # 判断正负，左边为负，右边为正，只看偏航角
            # if center_x < self.head_img.shape[1] / 2:
            #     # 如果目标在图像的左侧，向左转，degree 为负数
            #     angle_degree = -1 * angle_degree


            obj_name =  phrases[index]#获得目标名称，可能有多个

            obj_info = (obj_name, camera_distance, angle_degree)
            final_obj_list.append(obj_info)
            index = index + 1

        #画框
        #annotated_frame：cv2的图片，image_source：pil图片
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        return phrases, final_obj_list, annotated_frame

    def ob_objects_llm(self):
        # """
        # 注意需要先执行get_image，为llm提供观测结果
        # 在图像 img 上运行对象检测模型，获得目标列表 [ <对象名称、距离、角度（以度为单位）>,...] , 给到llm用于推理
        # :return:[ <对象名称、距离、角度（以度为单位）>,...] 如[(门，0.53，22)，(椅子，4.84，-21)]
        # """
        #获得识别结果
        
        ob_list, final_obj_list, annotated_frame = self.ob_objects(self.obj_name_list)
        #显示图片
        cv2.imshow("ob_objects", annotated_frame)
        cv2.waitKey(0)
        
        final_result = []

        for obj_info in final_obj_list:
            item = (obj_info[0], obj_info[1], obj_info[2]) #obj_name, camera_distance, angle_degree
            final_result.append(item)
        print()
        print("final_result:",final_result)
        print("str_final_result:",str(final_result))
        print()
        self.final_result = final_result
        # 图中物体的<对象名称、距离、角度（以度为单位）>如后文所示，对应关系。
        self.llm.ask("以下是我们得到的环境中物体各自的标签、到无人机的距离，以及无人机为了正对目标需要旋转的角度（向左为负，向右为正），例如(门,0.53,22)，意味着现在无人机需要向右旋转22度，然后向前飞行0.53m就可以到达目标“门”的附近" + str(final_result))
        return final_result
