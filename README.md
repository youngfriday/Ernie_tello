# Drone Voice Interaction System Based on ERNIE Bot LLM and Multimodal Models

[our demo](https://bhpan.buaa.edu.cn/link/AADFAF9062597C4C29858163ABCFAA95F3)

## Project Overview

This project leverages the Baidu [`ERNIE Bot`](https://yiyan.baidu.com/) LLM to control a [`TELLO`](https://github.com/damiafuentes/DJITelloPy) drone for object-finding tasks and other operations. The modules are outlined as follows:

- **Speech Recognition Module:** Pending upload
- **Object Detection and Recognition Module:** Utilizing the multimodal models [`RAM`](https://github.com/xinyu1205/recognize-anything) and [`Grounding DINO`](https://github.com/IDEA-Research/GroundingDINO)
- **Monocular Depth Estimation Module:** Based on [`GLPN`](https://huggingface.co/vinvino02/glpn-nyu)


1. Create and activate a Conda environment: (python: recommend 3.8 or 3.9 (This project was set by 3.9))

   ```bash
   conda create -n tello python=3.9 -y
   conda activate tello
   ```

2. git clone our project:

   ```bash
   git clone https://github.com/youngfriday/ernie_tello.git
   ```

3. Deploy and test each model, we have download them in the link below( for some Internet reasons), 

   https://bhpan.buaa.edu.cn/link/AAD4F562ABBE5B4648A81BE2FF50DD18C3

   just download and unzip them , then put the 5 folders in your workplace folder.

   or you can deploy them from the scratch. (if you have a good Internet or a stable source)Whatever, you should carefully follow their professional setting up instructions:

   - [`RAM` ](https://github.com/xinyu1205/recognize-anything)
   - [`Grounding DINO`](https://github.com/IDEA-Research/GroundingDINO) **(if you want to use cuda, do pay attention on its instruction!)**

   You can find `ram4test.py` , `dino4test.py` ,`depth4test.py`  if you download all given.

   > Make sure that you can run them successfully, which means all models have been deployed successfully in your computer, then try to run the `main.py` .

##  Acknowledgements

Thanks to their extraordinary work:

- [RAM ](https://github.com/xinyu1205/recognize-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) 
- [GLPN](https://huggingface.co/vinvino02/glpn-nyu)
- [GPT for Robotics](https://www.microsoft.com/en-us/research/articles/chatgpt-for-robotics/)
- [【Hackathon 5th】提示词的魅力：用文心大模型飞飞机！](https://aistudio.baidu.com/projectdetail/7158159?searchKeyword=%E7%94%A8%E6%96%87%E5%BF%83%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A3%9E%E9%A3%9E%E6%9C%BA&searchTab=ALL)
- If you can read Chinese, we highly recommend you to read the last one instruction series, without which we have no way to finish this job.

