#!/usr/bin/env bash

sudo apt install -y cmake build-essential python3-dev python3-pip

# Install bazel
sudo apt install curl gnupg
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

sudo apt update && sudo apt install -y bazel=3.0.0

# Clone tensorflow repo
cd ~
git clone https://github.com/tensorflow/tensorflow.git
cd ~/tensorflow
git checkout master
./configure

# Build benchmark tool
bazel build -c opt --config=noaws --config=nohdfs --config=nonccl --verbose_failures tensorflow/lite/tools/benchmark:benchmark_model

# MobileNet V2, 1 thread
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=../xnnpack_benchmarks/models/mobilenet_v2_xnnpack.tflite \
  --num_threads=1 --enable_op_profiling=true --max_profiling_buffer_entries=16384 --warmup_runs=20 --num_runs=1000 --use_xnnpack=false

./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=../xnnpack_benchmarks/models/mobilenet_v2_xnnpack.tflite \
  --num_threads=1 --enable_op_profiling=true --max_profiling_buffer_entries=16384 --warmup_runs=20 --num_runs=1000 --use_xnnpack=true

# ResNet 50, 1 thread
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=../xnnpack_benchmarks/models/resnet_50_xnnpack.tflite \
  --num_threads=1 --enable_op_profiling=true --max_profiling_buffer_entries=16384 --warmup_runs=20 --num_runs=1000 --use_xnnpack=false

./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=../xnnpack_benchmarks/models/resnet_50_xnnpack.tflite \
  --num_threads=1 --enable_op_profiling=true --max_profiling_buffer_entries=16384 --warmup_runs=20 --num_runs=1000 --use_xnnpack=true

# MobileNet V2, 8 threads
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=../xnnpack_benchmarks/models/mobilenet_v2_xnnpack.tflite \
  --num_threads=8 --enable_op_profiling=true --max_profiling_buffer_entries=16384 --warmup_runs=20 --num_runs=1000 --use_xnnpack=false

./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=../xnnpack_benchmarks/models/mobilenet_v2_xnnpack.tflite \
  --num_threads=8 --enable_op_profiling=true --max_profiling_buffer_entries=16384 --warmup_runs=20 --num_runs=1000 --use_xnnpack=true

# ResNet 50, 8 threads
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=../xnnpack_benchmarks/models/resnet_50_xnnpack.tflite \
  --num_threads=8 --enable_op_profiling=true --max_profiling_buffer_entries=16384 --warmup_runs=20 --num_runs=1000 --use_xnnpack=false

./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=../xnnpack_benchmarks/models/resnet_50_xnnpack.tflite \
  --num_threads=8 --enable_op_profiling=true --max_profiling_buffer_entries=16384 --warmup_runs=20 --num_runs=1000 --use_xnnpack=true
