# OpenVINO_Face_Anti-Spoofing

Pytorch and Openvino model for Face Anti-Spoofing task. Model was trained on ID R&D Antispoofing Challenge dataset.

### OpenVINO setup

## Setup

1. Download and install OpenVINO: https://software.seek.intel.com/openvino-toolkit

2. Setup environment

    * Windows
      ```bash
      "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
      ```
      Expected output:
      ```bash
      Python 3.7.6
      ECHO is off.
      PYTHONPATH=C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\accuracy_checker;C:\Program Files (x86)\IntelSWTools\openvino\python\python3.7;C:\Program Files (x86)\IntelSWTools\openvino\python\python3;C:\Users\dkurtaev\opencv\build\lib\Release
      [setupvars.bat] OpenVINO environment initialized
      ```

    * Linux
      ```bash
      source /opt/intel/openvino/bin/setupvars.sh
      ```
      Expected output:
      ```bash
      [setupvars.sh] OpenVINO environment initialized
      ```

### Single image prediction:
```
python3 predict_image.py -i %image_path%
```
### Camera demo:
```
python3 predict_camera.py
```

### Predicted samples:

 Original video | Spoofed video
----------|--------------
![](data/real_gif.gif) | ![](data/spoof_gif.gif) |
