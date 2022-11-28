# Yolov5 + Deep Sort + 3d position/speed estimation with PyTorch

<div align="center">
<p>
<img src="demo/Highway_Free_footage.gif" width="400"/> <img src="demo/inha_backgate.gif" width="400"/> 
</p>
</div>

## Project Co-authored
[JinHee Lee](https://github.com/happinis)<br>
Zeyi Lin

## Introduction

This repository is base on [YOLOv5](https://github.com/ultralytics/yolov5) and [Deep Sort](https://github.com/ZQPei/deep_sort_pytorch) to detect the speed estimation.


## Before you run the code
1. check you Python >= 3.7, torch >= 1.7
2. install all the things in requirements.txt

`pip install -r requirements.txt`
(If you're using Anconda using `conda install` can avoid lots of problems.)


## Download weight

Our models pretrained on the [Aihub Dataset(South Korea)](https://aihub.or.kr/aidata/34120)<br>
[best_yolov5s_aihub_finetune.pt](https://drive.google.com/file/d/1G4jLCseNlvQYFseYRbC1Ot6DHY-8QGTj/view?usp=sharing)<br>
[best_yolov5l_aihub_finetune.pt](https://drive.google.com/file/d/1tQmM9XHImSe89QSn9OGdL0ufHvH6ootO/view?usp=share_link)<br>
You can also download yolov5 pretrained weight at [yolov5 releases](https://github.com/ultralytics/yolov5/releases)

## run code

```bash


$ python track.py --source test.mp4 --yolo_weights weights/best_yolov5s_aihub_finetune.pt --img 640
                                                   weights/best_yolov5l_aihub_finetune.pt
                                                   
```
You can also run on image source, webcam, YouTube etc.

If you want to see implent just use ```--show-vid```

```bash


$ python track.py --source test.mp4 --yolo_weights weights/best_yolov5s_aihub_finetune.pt --show-vid
                                                   
```

And use ```--save-vid```
Can be saved to `inference/output` by 

```bash
python3 track.py --source ... --yolo_weights ... --save-vid
```

## References
https://github.com/ultralytics/yolov5<br>
https://github.com/ZQPei/deep_sort_pytorch<br>
https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch<br>
https://www.scirp.org/pdf/JCC_2019052714144662.pdf<br>
https://ieeexplore.ieee.org/document/7101268<br>
