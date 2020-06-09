# XNNPACK Benchmarks

This repo contains the code to benchmark Tensorflow Lite (TFLite) using the XNNPACK delegate against Intel's OpenVINO inference package. The tests were run on a Google Cloud Ubuntu 16.04 LTS VM with 8 vCPUs. Before running the benchmarks, please first update the system:

sudo apt update
sudo apt -y full-upgrade
sudo reboot

Next make sure you have tensorflow installed & generate the TFLite models by running main.py. Note that the XNNPACK models are optimized by removing all XNNPACK incompatible operators (note that XNNPACK only supports a subset of all TFLite operators). This includes explicit padding layers and the mean operation used in the global average pooling layer.

To perform the OpenVINO benchmarks, run "run_benchmarks_openvino.sh". Similarly for XNNPACK, run "run_benchmarks_xnnpack.sh". No system setup is necessary, the scripts will download & install OpenVINO & XNNPACK.

Results:

Model         | Backend                         | Latency (ms) | Throughput (FPS)
---           | ---                             | ---          | ---
MobileNet V2  | Tflite                          | 98.3         | 10.17
MobileNet V2  | Tflite + XNNPACK                | 11.48        | 87.1
MobileNet V2  | OpenVino, latency optimized     | 11.14        | 89.77
ResNet 50     | Tflite                          | 356.5        | 2.81
ResNet 50     | Tflite + XNNPACK                | 118.3        | 8.45
ResNet 50     | OpenVino, latency optimized     | 62.55        | 15.99

And 8 threads:

Model         | Backend                         | Latency (ms) | Throughput (FPS)
---           | ---                             | ---          | ---
MobileNet V2  | Tflite                          | 51.4         | 19.45
MobileNet V2  | Tflite + XNNPACK                | 3.16         | 316.46
MobileNet V2  | OpenVino, latency optimized     | 3.56         | 280.63
MobileNet V2  | OpenVino, throughput optimized  | 10.40        | 382.44
ResNet 50     | Tflite                          | 107.3        | 9.32
ResNet 50     | Tflite + XNNPACK                | 32.8         | 30.49
ResNet 50     | OpenVino, latency optimized     | 16.70        | 59.88
ResNet 50     | OpenVino, throughput optimized  | 58.88        | 67.73