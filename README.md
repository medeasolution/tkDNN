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

To compile and install OpenCV4 with contrib us the script ```install_OpenCV4.sh```. It will download and compile OpenCV
in Download folder.

```
bash scripts/install_OpenCV4.sh
```

When using openCV not compiled with contrib, comment the definition of OPENCV_CUDACONTRIBCONTRIB in
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

## Workflow

First, this workflow can be easily automated with a bash script, and it's planned to do it in the future.

Steps needed to do inference on tkDNN with a custom neural network.

* Build and train a NN model with Darknet. In this point you will have `yolo4.cfg`, `yolo4.names` and `yolo4.weights`
  files.
* Export weights and bias for each layer and save them in a binary file (one for layer): refer to next section.
* Export outputs for each layer and save them in a binary file (one for layer).
* Create a new test and define the network, layer by layer using the weights extracted and the output to check the
  results.
* Do inference.

## How to export weights

Weights are essential for any network to run inference. For each test a folder organized as follows is needed (in the
`build/` folder):

```
    yolo4/
    ├── layers/ (folder containing a binary file for each layer with the corresponding wieghts and bias)
    └── debug/  (folder containing a binary file for each layer with the corresponding outputs)
```

Therefore, once the weights have been exported, the folders layers and debug should be placed in the
corresponding `yolo4/` folder.

### Export weights from darknet

To export weights for NNs that are defined in darknet framework,
use [this](https://git.hipert.unimore.it/fgatti/darknet.git) fork of darknet and follow these steps to obtain a correct
debug and layers folder, ready for tkDNN.

```
git clone https://github.com/medeasolution/darknet-export-layers
cd darknet-export-layers
make
mkdir layers debug
./darknet export <path-to-cfg-file> <path-to-weights-file> layers
```

- `<path-to-cfg-file>`: yolo4.cfg used for training
- `<path-to-weights-file>`: yolo4.weights obtained after training with Darknet.

Note: Use compilation with CPU (leave GPU=0 in Makefile) if you also want debug.

Put this `yolo4/` folder in `tkDNN-python/build/` folder. After that, you can run `test_yolo4` as following:

```bash
cd build/
./test_yolo4
```

The source code of this binary is defined in `tests/darknet/yolo4.cpp`, and as defined there, it will look for these two
files:

- `tests/darknet/cfg/yolo4.cfg`
- `tests/darknet/names/yolo4.names`

This two paths is where you should put the files (cfg and names) that you used during training.

After that, you will have the TensorRT engine with FP32 precision in the same folder. For FP16 (faster) precision,
you should first run `export TKDNN_MODE=FP16` as defined in next section.

### FP16 inference

N.b. By default it is used FP32 inference

To run the an object detection demo with FP16 inference follow these steps:

```
export TKDNN_MODE=FP16  # set the half floating point optimization
./test_yolo4            # run the yolo test (is slow)
```

N.b. Using FP16 inference will lead to some errors in the results (first or second decimal).

### INT8 inference

To run the an object detection demo with INT8 inference three environment variables need to be set:

* ```export TKDNN_MODE=INT8```: set the 8-bit integer optimization
* ```export TKDNN_CALIB_IMG_PATH=/path/to/calibration/image_list.txt``` : image_list.txt has in each line the absolute
  path to a calibration image
* ```export TKDNN_CALIB_LABEL_PATH=/path/to/calibration/label_list.txt```: label_list.txt has in each line the absolute
  path to a calibration label

You should provide image_list.txt and label_list.txt, using training images.

N.B.

* Using INT8 inference will lead to some errors in the results.
* The test will be slower: this is due to the INT8 calibration, which may take some time to complete.
* INT8 calibration requires TensorRT version greater than or equal to 6.0
* Only 100 images are used to create the calibration table by default (set in the code).

### BatchSize bigger than 1

```
export TKDNN_BATCHSIZE=2
# build tensorRT files
```

This will create a TensorRT file with the desidered **max** batch size.
The test will still run with a batch of 1, but the created tensorRT can manage the desidered batch size.

Current Python wrapper doesn't support inference with a batch size bigger than 1, but a few changes can be made in order
to support it.

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

## Run the demo

This is an example using yolov4. Once you have succesfully created your rt file, run the demo:

```
./demo yolo4_fp32.rt ../demo/yolo_test.mp4 y
```

In general the demo program takes 7 parameters:

```
./demo <network-rt-file> <path-to-video> <kind-of-network> <number-of-classes> <n-batches> <show-flag>
```

where

* ```<network-rt-file>``` is the rt file generated by a test
* ```<<path-to-video>``` is the path to a video file or a camera input
* ```<kind-of-network>``` is the type of network. Thee types are currently supported: ```y``` (YOLO family), ```c``` (
  CenterNet family) and ```m``` (MobileNet-SSD family)
* ```<number-of-classes>```is the number of classes the network is trained on
* ```<n-batches>``` number of batches to use in inference (N.B. you should first export TKDNN_BATCHSIZE to the required
  n_batches and create again the rt file for the network).
* ```<show-flag>``` if set to 0 the demo will not show the visualization but save the video into result.mp4 (if
  n-batches ==1)
* ```<conf-thresh>``` confidence threshold for the detector. Only bounding boxes with threshold greater than conf-thresh
  will be displayed.

![demo](https://user-images.githubusercontent.com/11562617/72547657-540e7800-388d-11ea-83c6-49dfea2a0607.gif)

## PYTHON

To run the an object detection demo with python (example with yolov4):

```
python darknetTR.py build/yolo4_fp16.rt --video=demo/yolo_test.mp4
```

## FPS Results

Inference FPS of YOLOv4 with tkDNN, average of 1200 images with the same dimension as the input size, on

* RTX 2080Ti (CUDA 10.2, TensorRT 7.0.0, Cudnn 7.6.5);
* Xavier AGX, Jetpack 4.3 (CUDA 10.0, CUDNN 7.6.3, tensorrt 6.0.1 );
* Tx2, Jetpack 4.2 (CUDA 10.0, CUDNN 7.3.1, tensorrt 5.0.6 );
* Jetson Nano, Jetpack 4.4  (CUDA 10.2, CUDNN 8.0.0, tensorrt 7.1.0 ).

| Platform   | Network    | FP32, B=1 | FP32, B=4 | FP16, B=1 | FP16, B=4 | INT8, B=1 | INT8, B=4 | 
| :------:   | :-----:    |:---------:|:---------:|:---------:|:---------:|:---------:|:---------:| 
| RTX 2080Ti | yolo4  320 |  118,59   |  237,31   |  207,81   |  443,32   |  262,37   |  530,93   | 
| RTX 2080Ti | yolo4  416 |  104,81   |  162,86   |  169,06   |  293,78   |  206,93   |  353,26   | 
| RTX 2080Ti | yolo4  512 |   92,98   |  132,43   |  140,36   |  215,17   |  165,35   |  254,96   | 
| RTX 2080Ti | yolo4  608 |   63,77   |   81,53   |  111,39   |  152,89   |  127,79   |  184,72   | 

## MAP Results

Results for COCO val 2017 (5k images), on RTX 2080Ti, with conf threshold=0.001

|                      | CodaLab       | CodaLab   | CodaLab       | CodaLab     | tkDNN map     | tkDNN map |
| -------------------- | :-----------: | :-------: | :-----------: | :---------: | :-----------: | :-------: |
|                      | **tkDNN**     | **tkDNN** | **darknet**   | **darknet** | **tkDNN**     | **tkDNN** |
|                      | MAP(0.5:0.95) | AP50      | MAP(0.5:0.95) | AP50        | MAP(0.5:0.95) | AP50      |
| Yolov3 (416x416)     | 0.381         | 0.675     | 0.380         | 0.675       | 0.372         | 0.663     |
| yolov4 (416x416)     | 0.468         | 0.705     | 0.471         | 0.710       | 0.459         | 0.695     |
| yolov3tiny (416x416) | 0.096         | 0.202     | 0.096         | 0.201       | 0.093         | 0.198     |
| yolov4tiny (416x416) | 0.202         | 0.400     | 0.201         | 0.400       | 0.197         | 0.395     |
| Cnet-dla34 (512x512) | 0.366         | 0.543     | \-            | \-          | 0.361         | 0.535     |
| mv2SSD (512x512)     | 0.226         | 0.381     | \-            | \-          | 0.223         | 0.378     |

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
