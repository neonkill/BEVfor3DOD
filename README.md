# CV_For_Autonomous_Driving
Computer Vision for Autonomous Driving

## structure

        CV_For_Autonomous_Driving
        ├── configs
        │   ├── data
        │   ├── model
        │   ├── config.yaml
        │
        ├── data_modules
        │   ├── data_module.py
        │   ├── augmentations.py
        │   ├── datasets
        │       ├── nuscenes_depth.py
        │       ├── nuscenes_multiple.py
        │       ├── nuscenes_object_detection.py
        │       ├── nuscenes_mapview_segmentation.py
        │       ├── nuscenes_semantic_segmentation.py
        │
        ├── model_modules
        │   ├── loss.py
        │   ├── metrics.py
        │   ├── model_module.py
        │   ├── models
        │       ├── nn.module classes
        |
        ├── scripts
        │   ├── train.py
        │   ├── utils.py
        │   ├── test.py
        │
        ├── .gitignore
        ├── LICENSE
        ├── README.md