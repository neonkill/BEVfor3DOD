# CV_For_Autonomous_Driving
Computer Vision for Autonomous Driving

## structure

        CV_For_Autonomous_Driving
        ├── config
        │   ├── data
        │   ├── model
        │   ├── experiment
        │   ├── default_config.yaml
        │
        ├── data_module
        │   ├── lightning_data_module.py
        │   ├── augmentation.py
        │   ├── dataset
        │       ├── nuscenes_dataset.py
        │
        ├── model_module
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
        docker pull yelin2/rml_pytorch_lightning:v0.0

        docker images
        
        # docker images cmd 입력 해서 아래와 같이 docker pull 한 이미지 있는지 확인
        REPOSITORY                    TAG   IMAGE ID      CREATED      SIZE
        yelin2/rml_pytorch_lightning  v0.0  5a63efefb75c  8 hours ago  41.7GB


2. pull 한 docker image로 container 실행하기

        sudo nvidia-docker run --name cv --ipc=host --gpus all -it yelin2/rml_pytorch_lightning:v0.0


3. container attach

- terminal에서 attach 하기

        docker attach cv      


        
- vscode에서 attach 하기
        
1. 좌측 메뉴에서 docker 선택
2. CONTANERS에서 실행 중인 yelin2/rml_pytorch_lightning container 우클릭
3. attach visual studio code 선택

[ ] vscode에서 attach 하는 방법 나중에 사진 첨부해서 설명할 것