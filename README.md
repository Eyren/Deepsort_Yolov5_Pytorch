# yolov5 + deepsort
Using yolov5 and deepsort to detect project, which you can choose any class(any class in coco dateset)

# Install
ubuntu 16.04    python >=3.7    cuda 10.0   cudnn 7.6.5     opencv-python 4.2.0.34  pytorch 1.5.1   torchvision 0.6.1

```
conda create -n yolov5 python=3.7
conda activate yolov5
git clone https://github.com/Eyren/Deepsort_Yolov5_Pytorch.git Deepsort_yolov5
cd Deepsort_Yolov5_Pytorch-master
pip install -r requirements.txt
```

# Test

- put the video or image sequence folder you want to test in demo folders.

```
python yolov5_deepsort.py demo/*.mp4(demo/img) --clsnum 2)
```
tips: clsnum default is car(2), you can check configs/coco.names to select class name, it count begin 0.(person is 0, bicycle is 1...)

# Result


# Contributers
dhpdong, eyren