import tensorflow as tf
from utils import convert_keras_to_tflite
from models.mobilenet_v2_xnnpack import MobileNetV2XNNPACK
from models.resnet_50_xnnpack import ResNet50XNNPACK

input_dummy = tf.zeros(shape=(1, 224, 224, 3))

model = tf.keras.applications.MobileNetV2()
model(input_dummy)
model_tflite = convert_keras_to_tflite(model, output_path='./models/mobilenet_v2.tflite')

model = MobileNetV2XNNPACK(weights=None)
model(input_dummy)
model_tflite = convert_keras_to_tflite(model, output_path='./models/mobilenet_v2_xnnpack.tflite')


model = tf.keras.applications.ResNet50()
model(input_dummy)
model_tflite = convert_keras_to_tflite(model, output_path='./models/resnet_50.tflite')

model = ResNet50XNNPACK(weights=None)
model(input_dummy)
model_tflite = convert_keras_to_tflite(model, output_path='./models/resnet_50_xnnpack.tflite')
