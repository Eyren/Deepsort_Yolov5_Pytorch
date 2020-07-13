import argparse
import numpy

import torch.backends.cudnn as cudnn

from detector.yolov5.utils import google_utils
from detector.yolov5.utils.datasets import *
from detector.yolov5.utils.utils import *


def bbox_r(width, height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


class YOLOv5(object):
    def __init__(self, weights, view_img, namesfile, imgsz, augment, class_id,
                 device, classes, agnostic_nms, conf_thres, iou_thres):
        self.weights = weights
        self.view_img = view_img
        self.imgsz = imgsz
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.class_names = self.load_class_names(namesfile)
        self.augment = augment
        self.class_id = class_id

    def model_load(self):
        # Initialize
        device = torch_utils.select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = torch.load(self.weights, map_location=device)['model'].float()  # load FP32 model
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        return model

    def __call__(self, img, model):
        device = torch_utils.select_device(self.device)
        half = device.type != 'cpu'
        imgsz = check_img_size(self.imgsz, s=model.stride.max())  # check img_size
        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(img, img_size=self.imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        # t0 = torch_utils.time_synchronized()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        for img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)
            t2 = torch_utils.time_synchronized()
            # print(pred)
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            bbox_xywh = []
            confs = []
            clas = []
            for i, det in enumerate(pred):  # detections per image
                s, im0 = '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # print("\n det is:", det)
                    # Write results
                    for *xyxy, conf, cls in det:
                        # print(*xyxy)
                        # print(conf.item())
                        # print(cls.item())
                        img_h, img_w, _ = im0.shape  # get image shape

                        x_c, y_c, bbox_w, bbox_h = bbox_r(img_w, img_h, *xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        if cls == self.class_id:
                            bbox_xywh.append(obj)
                            confs.append(conf.item())
                            clas.append(cls.item())
                            # print(bbox_xywh)
            t3 = torch_utils.time_synchronized()
            # print(bbox_xywh)
            # print(confs)
            # print(clas)
            return numpy.array(bbox_xywh), confs, clas


    def load_class_names(self, namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names
