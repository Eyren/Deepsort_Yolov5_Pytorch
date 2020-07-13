from detector.yolov5.detect import YOLOv5


__all__ = ['build_detector']


def build_detector(arg):
    return YOLOv5(weights=arg.weights, view_img=arg.view_img, namesfile=arg.classname,
                  imgsz=arg.img_size, device=arg.device, classes=arg.classes,
                  agnostic_nms=arg.agnostic_nms, augment=arg.augment, class_id=arg.class_id,
                  conf_thres=arg.conf_thres, iou_thres=arg.iou_thres)
