<<<<<<< HEAD
# CV_For_Autonomous_Driving
Computer Vision for Autonomous Driving

## structure

        CV_For_Autonomous_Driving
        ├── config
        │   ├── data
        │   ├── model
=======
# BEVfor3DOD
BEV Representation For 3D Object Detection Using Bipartite matching
BEVfor3DOD는 현재 3DOD에서 활용하지 않는 이미지 레벨의 분석(depth, segmentation)을 같이 학습하여 3DOD에서 필요한 정보를 feature map를 담을 수 있게 하였다. 그리고 position embedding으로 BEV 좌표계의 정보를 투영하고 1번의 attention을 진행하여 50*50 BEV Representation을 만들었다. 그리고 BEV의 특성인 각각의 object는 겹치지 않는 것을 활용하여 Detectin head로 활용되는 DETR의 transformer를 사용하지 않고 각각의 그리드 마다 predict하여 그 결과를 Bipatite matching으로 학습하여 실시간성 모델을 만들어냈다. NDS 31.6, mAP 27.6, RTX 3090기준 FPS 19.9으로 기존 트렌드 모델들과 경쟁력있는 결과를 얻었다.
![image](https://github.com/neonkill/BEVfor3DOD/assets/72084525/5132d304-0f6f-43c4-98f9-727799134863)

# training
Environments
- Linux, Python==3.10, CUDA == 11.3, pytorch == 1.12.0, mmdet3d == 0.17.1

- python script/train.py +experiment=~~~
dataset : nuscene
GPU : 8(RTX3090)
training time : 1d 7h
# ablation Result
1) 이미지에 대한 분석 유무
![image](https://github.com/neonkill/BEVfor3DOD/assets/72084525/314a42f7-7897-4abb-bb19-69472dcca9a2)

2) Detection방식
   
![image](https://github.com/neonkill/BEVfor3DOD/assets/72084525/f4e92df2-f00f-4fba-b970-14c7732a1eec)
   ![image](https://github.com/neonkill/BEVfor3DOD/assets/72084525/57bb304e-57d7-4f19-a5ee-b21e77dadd8e)

# prdiction visualization
![image](https://github.com/neonkill/BEVfor3DOD/assets/72084525/5df129a2-6f06-48cf-996c-5432f6192f05)

![image](https://github.com/neonkill/BEVfor3DOD/assets/72084525/eeb27fda-59c5-475b-9c45-0a7e2586c733)

## structure

        BEVfor3DOD
        ├── config
        │   ├── data
        │   ├── model
        │   ├── loss
        │   ├── metrics
>>>>>>> 1783aa04ca18792310298f0c12a333c87fc1c892
        │   ├── experiment
        │   ├── default_config.yaml
        │
        ├── data_module
        │   ├── lightning_data_module.py
<<<<<<< HEAD
=======
        │   ├── transforms.py
>>>>>>> 1783aa04ca18792310298f0c12a333c87fc1c892
        │   ├── augmentation.py
        │   ├── dataset
        │       ├── nuscenes_dataset.py
        │
        ├── model_module
<<<<<<< HEAD
=======
        │   ├── model
        │       ├── detection
        │       ├── ...   
>>>>>>> 1783aa04ca18792310298f0c12a333c87fc1c892
        │   ├── loss.py
        │   ├── metric.py
        │   ├── lightning_model_module.py
        │   ├── model
        |
        ├── scripts
        │   ├── train.py
        │   ├── utils.py
        │   ├── test.py
        │
        ├── .gitignore
        ├── LICENSE
        ├── README.md

## Run Docker Container

1. docker image pull 하기

        cd ~/ws
<<<<<<< HEAD
        docker pull yelin2/rml_pytorch_lightning:v0.0
=======
        docker pull neonklll/pytorch_lightning:v0.0
>>>>>>> 1783aa04ca18792310298f0c12a333c87fc1c892

        docker images
        
        # docker images cmd 입력 해서 아래와 같이 docker pull 한 이미지 있는지 확인
        REPOSITORY                    TAG   IMAGE ID      CREATED      SIZE
<<<<<<< HEAD
        yelin2/rml_pytorch_lightning  v0.0  5a63efefb75c  8 hours ago  41.7GB
=======
        neonklll/pytorch_lightning  v0.0  5a63efefb75c  8 hours ago  41.7GB
>>>>>>> 1783aa04ca18792310298f0c12a333c87fc1c892


2. pull 한 docker image로 container 실행하기

<<<<<<< HEAD
        sudo nvidia-docker run --name cv --ipc=host --gpus all -it yelin2/rml_pytorch_lightning:v0.0
=======
        sudo nvidia-docker run --name bev --ipc=host --gpus all -it neonklll/pytorch_lightning:v0.0
>>>>>>> 1783aa04ca18792310298f0c12a333c87fc1c892


3. container attach

- terminal에서 attach 하기

<<<<<<< HEAD
        docker attach cv      
=======
        docker attach bev      
>>>>>>> 1783aa04ca18792310298f0c12a333c87fc1c892


        
- vscode에서 attach 하기
        
1. 좌측 메뉴에서 docker 선택
<<<<<<< HEAD
2. CONTANERS에서 실행 중인 yelin2/rml_pytorch_lightning container 우클릭
3. attach visual studio code 선택

[ ] vscode에서 attach 하는 방법 나중에 사진 첨부해서 설명할 것
=======
2. CONTANERS에서 실행 중인 neonklll/pytorch_lightning:v0.0 우클릭
3. attach visual studio code 선택

[ ] vscode에서 attach 하는 방법 나중에 사진 첨부해서 설명할 것
>>>>>>> 1783aa04ca18792310298f0c12a333c87fc1c892
