import tensorflow as tf
from tensorflow.keras import layers, models

class XtructureModel(models.Model):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.build_model()
