#!/usr/bin/env bash

# Install OpenVINO
# See https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_apt.html#install_the_runtime_or_developer_packages_using_the_apt_package_manager
wget https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020?elq_cid=6414030
sudo apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2020?elq_cid=6414030
echo "deb https://apt.repos.intel.com/openvino/2020 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2020.list
sudo apt update

sudo apt install -y cmake build-essential intel-openvino-dev-ubuntu16-2020.3.194 intel-openvino-runtime-ubuntu16-2020.3.194 libtbb2 python3-pip
source /opt/intel/openvino/bin/setupvars.sh

# Install required Python dependencies
pip3 install pyyaml numpy networkx==2.3 protobuf==3.6.1 defusedxml requests

# Download the source Caffe models
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
python3 downloader.py --name mobilenet-v2 -o ~/openvino/
python3 downloader.py --name resnet-50 -o ~/openvino/

# Convert the Caffe models to OpenVINO
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
./install_prerequisites_caffe.sh
cd /opt/intel/openvino/deployment_tools/model_optimizer
python3 mo.py --input_model /home/pieterluitjens/openvino/public/mobilenet-v2/mobilenet-v2.caffemodel --data_type FP16 --output_dir /home/pieterluitjens/openvino/converted/fp16
python3 mo.py --input_model /home/pieterluitjens/openvino/public/mobilenet-v2/mobilenet-v2.caffemodel --data_type FP32 --output_dir /home/pieterluitjens/openvino/converted/fp32
python3 mo.py --input_model /home/pieterluitjens/openvino/public/resnet-50/resnet-50.caffemodel --data_type FP32 --output_dir /home/pieterluitjens/openvino/converted/fp32

# Build the C++ benchmark tool
cd /opt/intel/openvino/deployment_tools/inference_engine/samples/cpp
./build_samples.sh
cd /home/pieterluitjens/inference_engine_cpp_samples_build/intel64/Release

# Benchmark
# See https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html
echo "#########################################################################################################"
echo "Benchmarking 1 thread"

./benchmark_app -m /home/pieterluitjens/openvino/converted/fp32/mobilenet-v2.xml -api sync --progress true -b 1 -niter 1000 --nthreads 1

./benchmark_app -m /home/pieterluitjens/openvino/converted/fp32/resnet-50.xml -api sync --progress true -b 1 -niter 1000 --nthreads 1

echo "#########################################################################################################"
echo "Benchmarking max threads"

./benchmark_app -m /home/pieterluitjens/openvino/converted/fp32/mobilenet-v2.xml -api sync --progress true -b 1 -niter 1000
./benchmark_app -m /home/pieterluitjens/openvino/converted/fp32/mobilenet-v2.xml -api async --progress true -b 1 -niter 1000

./benchmark_app -m /home/pieterluitjens/openvino/converted/fp32/resnet-50.xml -api sync --progress true -b 1 -niter 1000
./benchmark_app -m /home/pieterluitjens/openvino/converted/fp32/resnet-50.xml -api async --progress true -b 1 -niter 1000
