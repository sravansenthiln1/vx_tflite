# VX TFLite
TFlite implementations from https://github.com/sravansenthiln1/armnn_tflite/
adapted to run on Verisilicon's Vivante NPU hardware platform.

Compatible with VIM3/3L, on the 4.9 kernel.

### Install pip
```shell
sudo apt-get install python3-pip
```

### Install necessary python packages
```shell
pip3 install numpy pillow
```

### Install the TFLite runtime interpreter
```shell
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### Create Library symlink
```shell
sudo ln /usr/lib/libOpenVX.so /usr/lib/libOpenVX.so.1
```

### Copy the necessary library files
If you are running on VIM3:
```shell
sudo cp libs/VIM3/libtim-vx.so /usr/lib/aarch64-linux-gnu/
```

If you are running on VIM3L:
```shell
sudo cp libs/VIM3L/libtim-vx.so /usr/lib/
```

Try the examples

* [Sine Model](./sine_model/) - Basic Neural network TFLite model

* [Digit recognize Model](./digit_recognize/) - Digit recognization model

* [Mobilenet v1 Model](./mobilenet_v1/) - Mobilenet v1 image classification model
