# tkDNN

tkDNN is a Deep Neural Network library built with cuDNN and TensorRT primitives. The main goal of this project is to
exploit NVIDIA boards as much as possible to obtain the best inference performance. It does not allow training.

## Dependencies

This branch works on every NVIDIA GPU that supports the dependencies:

* CUDA 10.0
* CUDNN 7.603
* TENSORRT 6.01
* OPENCV 3.4
* yaml-cpp 0.5.2 (sudo apt install libyaml-cpp-dev)

## About OpenCV

OpenCV is necessary to compile this repository. You will probably have it installed, if not follow the steps defined
in `ai-frame-manager` repository.

When using OpenCV not compiled with contrib, comment the definition of OPENCV_CUDACONTRIBCONTRIB in
include/tkDNN/DetectionNN.h. When commented, the preprocessing of the networks is computed on the CPU, otherwise on the
GPU. In the latter case some milliseconds are saved in the end-to-end latency.

## How to compile this repo

Build with cmake.

```
git clone https://github.com/medeasolution/tkDNN-python
cd tkDNN-python
mkdir build
cd build
cmake .. 
make -j8
```

## Building an engine for inference

Check `docs/build_engine.md`

## Demo

CHeck `docs/demo.md`

## PYTHON

The most important files are:

- `demo/darknetTR.cpp` and its headers. There are defined (as `extern`) the functions that will be called from Python.
- `darknetTR.py`: Where the structure to use this C functions is defined.

Also, to run the object detection demo with python:

```
python darknetTR.py build/yolo4_fp16.rt --video=demo/yolo_test.mp4
```

## FPS Results

Inference FPS of YOLOv4 with tkDNN, average of 1200 images with the same dimension as the input size, on

* RTX 2080Ti (CUDA 10.2, TensorRT 7.0.0, Cudnn 7.6.5);
* Xavier AGX, Jetpack 4.3 (CUDA 10.0, CUDNN 7.6.3, tensorrt 6.0.1 );
* Tx2, Jetpack 4.2 (CUDA 10.0, CUDNN 7.3.1, tensorrt 5.0.6 );
* Jetson Nano, Jetpack 4.4  (CUDA 10.2, CUDNN 8.0.0, tensorrt 7.1.0 ).

|  Platform  |  Network   | FP32, B=1 | FP32, B=4 | FP16, B=1 | FP16, B=4 | INT8, B=1 | INT8, B=4 |
|:----------:|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| RTX 2080Ti | yolo4  320 |  118,59   |  237,31   |  207,81   |  443,32   |  262,37   |  530,93   |
| RTX 2080Ti | yolo4  416 |  104,81   |  162,86   |  169,06   |  293,78   |  206,93   |  353,26   |
| RTX 2080Ti | yolo4  512 |   92,98   |  132,43   |  140,36   |  215,17   |  165,35   |  254,96   |
| RTX 2080Ti | yolo4  608 |   63,77   |   81,53   |  111,39   |  152,89   |  127,79   |  184,72   |

## MAP Results

Results for COCO val 2017 (5k images), on RTX 2080Ti, with conf threshold=0.001

|                      |    CodaLab    |  CodaLab  |    CodaLab    |   CodaLab   |   tkDNN map   | tkDNN map |
|----------------------|:-------------:|:---------:|:-------------:|:-----------:|:-------------:|:---------:|
|                      |   **tkDNN**   | **tkDNN** |  **darknet**  | **darknet** |   **tkDNN**   | **tkDNN** |
|                      | MAP(0.5:0.95) |   AP50    | MAP(0.5:0.95) |    AP50     | MAP(0.5:0.95) |   AP50    |
| Yolov3 (416x416)     |     0.381     |   0.675   |     0.380     |    0.675    |     0.372     |   0.663   |
| yolov4 (416x416)     |     0.468     |   0.705   |     0.471     |    0.710    |     0.459     |   0.695   |
| yolov3tiny (416x416) |     0.096     |   0.202   |     0.096     |    0.201    |     0.093     |   0.198   |
| yolov4tiny (416x416) |     0.202     |   0.400   |     0.201     |    0.400    |     0.197     |   0.395   |
| Cnet-dla34 (512x512) |     0.366     |   0.543   |      \-       |     \-      |     0.361     |   0.535   |
| mv2SSD (512x512)     |     0.226     |   0.381   |      \-       |     \-      |     0.223     |   0.378   |

## Darknet Parser

tkDNN implement and easy parser for darknet cfg files, a network can be converted with *tk::dnn::darknetParser*:

```
// example of parsing yolo4
tk::dnn::Network *net = tk::dnn::darknetParser("yolov4.cfg", "yolov4/layers", "coco.names");
net->print();
```

All models from darknet are now parsed directly from cfg, you still need to export the weights with the descripted tools
in the previus section.
<details>
  <summary>Supported layers</summary>
  convolutional
  maxpool
  avgpool
  shortcut
  upsample
  route
  reorg
  region
  yolo
</details>
<details>
  <summary>Supported activations</summary>
  relu
  leaky
  mish
</details>

## Existing tests and supported networks

| Test Name         | Network                                       | Dataset                                                       | N Classes | Input size    | Weights                                                                   |
| :---------------- | :-------------------------------------------- | :-----------------------------------------------------------: | :-------: | :-----------: | :------------------------------------------------------------------------ |
| yolo              | YOLO v2<sup>1</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 608x608       | [weights](https://cloud.hipert.unimore.it/s/nf4PJ3k8bxBETwL/download)                                                                   |
| yolo_224          | YOLO v2<sup>1</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| yolo_berkeley     | YOLO v2<sup>1</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)   | 10        | 416x736       | weights                                                                   |
| yolo_relu         | YOLO v2 (with ReLU, not Leaky)<sup>1</sup>    | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | weights                                                                   |
| yolo_tiny         | YOLO v2 tiny<sup>1</sup>                      | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/m3orfJr8pGrN5mQ/download)                                                                   |
| yolo_voc          | YOLO v2<sup>1</sup>                           | [VOC      ](http://host.robots.ox.ac.uk/pascal/VOC/)          | 21        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/DJC5Fi2pEjfNDP9/download)                                                                   |
| yolo3             | YOLO v3<sup>2</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/jPXmHyptpLoNdNR/download)     |
| yolo3_512   | YOLO v3<sup>2</sup>                                 | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/RGecMeGLD4cXEWL/download)     |
| yolo3_berkeley    | YOLO v3<sup>2</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)   | 10        | 320x544       | [weights](https://cloud.hipert.unimore.it/s/o5cHa4AjTKS64oD/download)                                                                   |
| yolo3_coco4       | YOLO v3<sup>2</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 4         | 416x416       | [weights](https://cloud.hipert.unimore.it/s/o27NDzSAartbyc4/download)                                                                   |
| yolo3_flir        | YOLO v3<sup>2</sup>                           | [FREE FLIR](https://www.flir.com/oem/adas/adas-dataset-form/) | 3         | 320x544       | [weights](https://cloud.hipert.unimore.it/s/62DECncmF6bMMiH/download)                                                                   |
| yolo3_tiny        | YOLO v3 tiny<sup>2</sup>                      | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/LMcSHtWaLeps8yN/download)     |
| yolo3_tiny512     | YOLO v3 tiny<sup>2</sup>                      | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/8Zt6bHwHADqP4JC/download)     |
| dla34             | Deep Leayer Aggreagtion (DLA) 34<sup>3</sup>  | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| dla34_cnet        | Centernet (DLA34 backend)<sup>4</sup>         | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/KRZBbCQsKAtQwpZ/download)     |
| mobilenetv2ssd    | Mobilnet v2 SSD Lite<sup>5</sup>              | [VOC      ](http://host.robots.ox.ac.uk/pascal/VOC/)          | 21        | 300x300       | [weights](https://cloud.hipert.unimore.it/s/x4ZfxBKN23zAJQp/download)     |
| mobilenetv2ssd512 | Mobilnet v2 SSD Lite<sup>5</sup>              | [COCO 2017](http://cocodataset.org/)                          | 81        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/pdCw2dYyHMJrcEM/download)     |
| resnet101         | Resnet 101<sup>6</sup>                        | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| resnet101_cnet    | Centernet (Resnet101 backend)<sup>4</sup>     | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/5BTjHMWBcJk8g3i/download)     |
| csresnext50-panet-spp    | Cross Stage Partial Network <sup>7</sup>     | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/Kcs4xBozwY4wFx8/download)     |
| yolo4             | Yolov4 <sup>8</sup>                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/d97CFzYqCPCp5Hg/download)     |
| yolo4_berkeley             | Yolov4 <sup>8</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)                          | 10        | 540x320       | [weights](https://cloud.hipert.unimore.it/s/nkWFa5fgb4NTdnB/download)     |
| yolo4tiny             | Yolov4 tiny                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/iRnc4pSqmx78gJs/download)     |

## References

1. Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer
   vision and pattern recognition. 2017.
2. Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
3. Yu, Fisher, et al. "Deep layer aggregation." Proceedings of the IEEE conference on computer vision and pattern
   recognition. 2018.
4. Zhou, Xingyi, Dequan Wang, and Philipp Krähenbühl. "Objects as points." arXiv preprint arXiv:1904.07850 (2019).
5. Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on
   computer vision and pattern recognition. 2018.
6. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer
   vision and pattern recognition. 2016.
7. Wang, Chien-Yao, et al. "CSPNet: A New Backbone that can Enhance Learning Capability of CNN." arXiv preprint arXiv:
   1911.11929 (2019).
8. Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "YOLOv4: Optimal Speed and Accuracy of Object
   Detection." arXiv preprint arXiv:2004.10934 (2020).
