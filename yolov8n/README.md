# Image detection using Tensorflow Lite
Original implementation: https://github.com/sravansenthiln1/armnn_tflite/tree/main/yolov8n

### Add symlinks for libraries
```shell
sudo ln ../libs/libvx_delegate.so libvx_delegate.so
```

### Run the example
```shell
python3 run_npu_inference.py
```

The detections are displayed on `output.png`
