#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   darknetTR.py.py
@Contact :   JZ

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/6/12 14:40   JZ      1.0         None
"""

from ctypes import *
import cv2
import numpy as np
import argparse
import os
from threading import Thread
import time
from queue import Queue

from config import INPUT_SIZE, N_CLASSES


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("cl", c_int),
                ("bbox", BOX),
                ("prob", c_float),
                ("name", c_char * 20),
                ]


BATCH_SIZE = 8

lib = CDLL("./build/libdarknetTR.so", RTLD_GLOBAL)

load_network = lib.load_network
load_network.argtypes = [c_char_p, c_int, c_int]
load_network.restype = c_void_p

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

do_inference = lib.do_inference
do_inference.argtypes = [c_void_p, IMAGE * BATCH_SIZE]

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_float, c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

frames_queue = Queue(maxsize=400)


def reader(video_path):
    stream = cv2.VideoCapture(video_path)
    stream.set(cv2.CAP_PROP_FPS, 120)
    ret = True
    while ret:
        ret, image = stream.read()
        frames_queue.put(image)


def detect_images(net, darknet_images, thresh=.5):
    num = c_int(0)

    pnum = pointer(num)
    seq = IMAGE * BATCH_SIZE
    do_inference(net, seq(*darknet_images), BATCH_SIZE)
    res = [[] for _ in range(BATCH_SIZE)]
    for batch_idx in range(BATCH_SIZE):
        dets = get_network_boxes(net, thresh, batch_idx, pnum)
        for i in range(pnum[0]):
            b = dets[i].bbox
            res[batch_idx].append((dets[i].name.decode("ascii"), (b.x, b.y, b.w, b.h)))
    return res


def loop_detect(detect_m):
    start = time.time()
    cnt = 0
    while True:
        images = []
        for i in range(BATCH_SIZE):
            image = frames_queue.get()
            if image is None:
                exit(0)
            image = cv2.resize(image,
                               (INPUT_SIZE, INPUT_SIZE),
                               interpolation=cv2.INTER_LINEAR)
            images.append(image)
        _ = detect_m.detect(images, need_resize=False)
        cnt += 1
        end = time.time()
        print("frame:{},time:{:.3f},FPS:{:.2f}".format(cnt * BATCH_SIZE, end - start, cnt * BATCH_SIZE / (end - start)))


class YOLO4RT(object):
    def __init__(self,
                 input_size=INPUT_SIZE,
                 weight_file="./yolo4_fp16.trt",
                 conf_thres=0.3):
        self.input_size = input_size
        self.metaMain = None
        self.model = load_network(weight_file.encode("ascii"), N_CLASSES, BATCH_SIZE)
        self.darknet_images = [make_image(input_size, input_size, 3) for _ in range(BATCH_SIZE)]
        self.thresh = conf_thres

    def detect(self, images, need_resize=True):
        for i, image in enumerate(images):
            if need_resize:
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(frame_rgb,
                                   (self.input_size, self.input_size),
                                   interpolation=cv2.INTER_LINEAR)
            frame_data = image.ctypes.data_as(c_char_p)
            copy_image_from_bytes(self.darknet_images[i], frame_data)

        detections = detect_images(self.model, self.darknet_images, thresh=self.thresh)
        # for i, image in enumerate(images):
        #     for det_class, det_bbox in detections[i]:
        #         tlx, tly, w, h = det_bbox
        #         image = cv2.rectangle(image, (int(tlx), int(tly)), (int(tlx + w), int(tly + h)), (255, 0, 0), 2)
        #
        #     cv2.imshow("image", image)
        #     cv2.waitKey(1)
        return detections


def parse_args():
    parser = argparse.ArgumentParser(description='tkDNN detect')
    parser.add_argument('weight', help='rt file path')
    parser.add_argument('--video', type=str, help='video path')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    r = Thread(target=reader, args=(args.video,), daemon=True)
    r.start()
    detect_m = YOLO4RT(weight_file=args.weight)
    t = Thread(target=loop_detect, args=(detect_m,), daemon=True)
    t.start()
    r.join()
    t.join()
