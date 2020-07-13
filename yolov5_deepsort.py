import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from detector.yolov5.utils.torch_utils import time_synchronized


class VideoTracker(object):
    def __init__(self, cfg, args, path):
        self.cfg = cfg
        self.args = args
        self.path = path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        elif os.path.isfile(self.path):
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(args)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.model = self.detector.model_load()

    def __enter__(self):
        video_name = None
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        elif os.path.isfile(self.path):
            self.vdo.open(self.path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frames = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
            video_name = str(self.path.split('/')[-1].split('.')[0])
            assert self.vdo.isOpened()
        elif os.path.isdir(self.path):
            pic_dir = sorted(os.listdir(self.path))
            pic = os.path.join(self.path, pic_dir[0])
            fir_pic = cv2.imread(pic)
            self.im_width = fir_pic.shape[1]
            self.im_height = fir_pic.shape[0]
            self.frames = len(pic_dir)
            video_name = self.path.split('/')[-3]

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, video_name+"_results.avi")
            self.save_results_path = os.path.join(self.args.save_path, video_name+"_results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        frames = self.frames
        model = self.model
        all_time = time_synchronized()
        # print(self.path)
        if os.path.isfile(self.path):
            while self.vdo.grab():
                idx_frame += 1
                if idx_frame % self.args.frame_interval:
                    continue

                start = time_synchronized()
                _, ori_im = self.vdo.retrieve()
                # im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

                # do detection
                bbox_xywh, cls_conf, cls_ids = self.detector(ori_im, model)
                # if not len(cls_ids):
                #     end = time_synchronized()
                #     outputs = []
                #     self.logger.info("time: {:.03f}s, fps: {:.03f}, frame: {}/{}, tracking numbers: {}"
                #                      .format(end - start, 1 / (end - start), idx_frame, frames, len(outputs)))
                #     print("Total:  Time: {:.03f}s    Fps: {:.03f}".format(end - all_time, idx_frame / (end - all_time)))
                #     continue

                # select person class
                # mask = cls_ids == 0

                # bbox_xywh = bbox_xywh[mask]
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                # bbox_xywh[:, 3:] *= 1.2
                # cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    results.append((idx_frame - 1, bbox_tlwh, identities))

                end = time_synchronized()

                if self.args.display:
                    cv2.imshow("test", ori_im)
                    cv2.waitKey(1)

                if self.args.save_path:
                    self.writer.write(ori_im)

                # save results
                write_results(self.save_results_path, results, 'mot')

                # logging
                self.logger.info("time: {:.03f}s, fps: {:.03f}, frame: {}/{}, tracking numbers: {}"
                                 .format(end - start, 1 / (end - start), idx_frame, frames, len(outputs)))

                print("Total:  Time: {:.03f}s    Fps: {:.03f}".format(end - all_time, idx_frame / (end - all_time)))

        elif os.path.isdir(self.path):
            pic_dir = sorted(os.listdir(self.path))
            for i in range(len(pic_dir)):
                start = time_synchronized()
                pic = os.path.join(self.path, pic_dir[i])
                ori_im = cv2.imread(pic)
                bbox_xywh, cls_conf, cls_ids = self.detector(ori_im, model)
                outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    results.append((i, bbox_tlwh, identities))

                end = time_synchronized()

                if self.args.display:
                    cv2.imshow("test", ori_im)
                    cv2.waitKey(1)

                if self.args.save_path:
                    self.writer.write(ori_im)

                # save results
                write_results(self.save_results_path, results, 'mot')

                # logging
                self.logger.info("time: {:.03f}s, fps: {:.03f}, frame: {}/{}, tracking numbers: {}"
                                 .format(end - start, 1 / (end - start), i + 1, frames, len(outputs)))

                print("Total:  Time: {:.03f}s    Fps: {:.03f}".format(end - all_time, (i + 1) / (end - all_time)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH", type=str)
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classname', type=str, default='./configs/coco.names', help='update all models')
    parser.add_argument('--class_id', type=int, default=0, help='all class id please refer to ./configs/coco.names')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, path=args.PATH) as vdo_trk:
        vdo_trk.run()
