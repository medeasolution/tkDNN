# Workflow when building TensorRT engine

First, this workflow can be easily automated with a bash script, and it's planned to do it in the future.

These steps are needed to do inference on tkDNN with a custom neural network.

* Build and train a NN model with Darknet. In this point you will have `yolo4.cfg`, `yolo4.names` and `yolo4.weights`
  files.
* Export weights and bias for each layer and save them in a binary file (one for layer). In this step you will also
  Export outputs for each layer and save them in a binary file (one for layer). More details in next
  section [How to export weights](#how-to-export-weights).
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

Therefore, once the weights have been exported (more details in next section), the layers and debug folders should be
placed in the
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

After that, you can run the `test_yolo4` binary, and you will have the TensorRT engine with FP32 precision in the same
folder. For FP16 (faster) precision, you should first run `export TKDNN_MODE=FP16` as defined in next section.

### FP16 inference

N.b. By default, it is used FP32 inference

To run the object detection demo with FP16 inference follow these steps:

```
export TKDNN_MODE=FP16  # set the half floating point optimization
./test_yolo4            # run the yolo test (is slow)
```

N.b. Using FP16 inference will lead to some errors in the results (first or second decimal).

### INT8 inference

To run the object detection demo with INT8 inference three environment variables need to be set:

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

### Batch Size bigger than 1

```
export TKDNN_BATCHSIZE=2
# build tensorRT files
```

This will create a TensorRT file with the desired **max** batch size.
The test will still run with a batch of 1, but the created tensorRT can manage the desired batch size.

Current Python wrapper doesn't support inference with a batch size bigger than 1, but a few changes can be made in order
to support it.