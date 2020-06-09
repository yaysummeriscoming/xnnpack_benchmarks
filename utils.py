import os
import tensorflow as tf
import numpy as np
import errno

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def convert_keras_to_tflite(model,
                            output_path=None,
                            quantize=False,
                            quantize_fp16=False,
                            quantize_weights=False,  # Weight quantization only
                            calibration_examples=1000,
                            iterator=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_weights:
        assert quantize is False, 'Only full quantization or weight quantization can be selected'
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # # Allow tf ops too: https://www.tensorflow.org/lite/guide/ops_select
    # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    #                         tf.lite.OpsSet.SELECT_TF_OPS]

    if quantize:
        assert quantize_weights is False, 'Only full quantization or weight quantization can be selected'
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # # todo: for full quantisation
        # https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations
        converter.quantized_input_stats = (0, 255)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # NOTE: tf 2.0 doesn't support quantized inputs/outputs ATM..
        # https://github.com/tensorflow/tensorflow/issues/34416
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        assert iterator is not None, 'For post-training integer quantization, a representative dataset is needed ' \
                                     'to calibrate the min/max values of every tensor'

        def representative_data_gen(num_examples=calibration_examples):
            for i in range(num_examples):
                data = next(iterator)[0]
                yield [data]

        converter.representative_dataset = representative_data_gen

    if quantize_fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    if output_path is not None:
        print('Writing tensorflow lite model to: %s' % output_path)
        mkdir_p(os.path.dirname(output_path))
        open(output_path, "wb").write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    return interpreter

class LiteWrapper():
    def __init__(self, model):
        self.model = model
        self.input_details = model.get_input_details()
        self.output_details = model.get_output_details()

    def __call__(self, x):
        if self.input_details[0]['dtype'] == np.uint8 and x.dtype != np.uint8:
            print('Casting input to uint8')
            x = tf.cast(x, tf.uint8)
        self.model.set_tensor(self.input_details[0]['index'], x)
        self.model.invoke()
        results = self.model.get_tensor(self.output_details[0]['index'])
        return results
