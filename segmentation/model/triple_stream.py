import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
    Activation, concatenate, Conv2D, Input, Multiply, MaxPooling2D,
    DepthwiseConv2D, Add
)
from tensorflow.keras.applications import InceptionResNetV2
from model.fusion import TripleStreamFusion, StreamFeatureProcessor
from config import Config


class SqueezeExcitation(tf.keras.layers.Layer):
    """Squeeze and Excitation block for channel attention"""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.gap = GlobalAveragePooling2D()
        self.fc1 = Dense(channels // reduction_ratio)
        self.fc2 = Dense(channels)

    def call(self, x):
        b, h, w, c = x.shape
        y = self.gap(x)
        y = self.fc1(y)
        y = tf.nn.relu(y)
        y = self.fc2(y)
        y = tf.nn.sigmoid(y)
        y = tf.reshape(y, [-1, 1, 1, c])
        return x * y

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config


class TripleStreamModel:
    def __init__(self, input_shape=(299, 299, 3), num_classes=7):
        """
        Initialize the triple stream model.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input images
        num_classes : int
            Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = Config()

    def _create_depthwise_block(self, x, filters, stride=1, name_prefix=''):
        """Depthwise Separable Convolution block with Squeeze-Excitation"""
        # Depthwise Convolution
        x = DepthwiseConv2D(3, strides=stride, padding='same',
                            name=f'{name_prefix}_depthwise')(x)
        x = BatchNormalization(name=f'{name_prefix}_dw_bn')(x)
        x = Activation('relu', name=f'{name_prefix}_dw_act')(x)

        # Pointwise Convolution
        x = Conv2D(filters, 1, padding='same', name=f'{name_prefix}_pointwise')(x)
        x = BatchNormalization(name=f'{name_prefix}_pw_bn')(x)
        x = Activation('relu', name=f'{name_prefix}_pw_act')(x)

        # Squeeze-Excitation
        se = SqueezeExcitation(filters)
        x = se(x)

        return x

    def _create_boundary_stream(self, inputs):
        """Create the boundary information stream"""
        x = self._create_depthwise_block(inputs, 32, stride=2, name_prefix='boundary_block1')
        # 150x150x32

        x = self._create_depthwise_block(x, 64, stride=2, name_prefix='boundary_block2')
        # 75x75x64

        x = self._create_depthwise_block(x, 128, stride=2, name_prefix='boundary_block3')
        # 38x38x128

        x = self._create_depthwise_block(x, 256, stride=2, name_prefix='boundary_block4')
        # 19x19x256

        # Additional reduction to match InceptionResNetV2 output size
        x = MaxPooling2D(2, strides=2, padding='valid')(x)
        # 9x9x256

        # Final convolution to get desired channel dimension
        x = Conv2D(self.config.BOUNDARY_FEATURE_DIM, 2, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # 8x8x384

        return x

    def _create_color_stream(self, inputs):
        """Create the color diversity stream"""
        x = self._create_depthwise_block(inputs, 32, stride=2, name_prefix='color_block1')
        # 150x150x32

        x = self._create_depthwise_block(x, 64, stride=2, name_prefix='color_block2')
        # 75x75x64

        x = self._create_depthwise_block(x, 128, stride=2, name_prefix='color_block3')
        # 38x38x128

        x = self._create_depthwise_block(x, 192, stride=2, name_prefix='color_block4')
        # 19x19x192

        # Additional reduction to match InceptionResNetV2 output size
        x = MaxPooling2D(2, strides=2, padding='valid')(x)
        # 9x9x192

        # Final convolution to get desired channel dimension
        x = Conv2D(self.config.COLOR_FEATURE_DIM, 2, strides=1, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # 8x8x256

        return x

    def build(self):
        """Build the complete triple stream model"""
        # Input layers
        original_input = Input(shape=self.input_shape, name='original_input')
        boundary_input = Input(shape=self.input_shape, name='boundary_input')
        color_input = Input(shape=self.input_shape, name='color_input')

        # Main stream (InceptionResNetV2)
        irv2_base = InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=original_input
        )

        # Freeze early layers
        for layer in irv2_base.layers[:700]:
            layer.trainable = False

        main_features = irv2_base.get_layer('conv_7b_ac').output  # 8x8x1536

        # Boundary information stream
        boundary_features = self._create_boundary_stream(boundary_input)

        # Color diversity stream
        color_features = self._create_color_stream(color_input)

        # Feature fusion
        fusion_module = TripleStreamFusion(
            main_channels=self.config.MAIN_FEATURE_DIM,
            boundary_channels=self.config.BOUNDARY_FEATURE_DIM,
            color_channels=self.config.COLOR_FEATURE_DIM
        )
        fused_features = fusion_module([main_features, boundary_features, color_features])

        # Classification head
        x = GlobalAveragePooling2D(name='gap')(fused_features)
        x = BatchNormalization(name='bn_final')(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        output = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create model
        model = Model(
            inputs=[original_input, boundary_input, color_input],
            outputs=output,
            name='triple_stream_model'
        )

        return model

    def compile_model(self, model, learning_rate=0.0001):
        """Compile the model with appropriate optimizer and loss"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.config.LABEL_SMOOTHING
        )

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

        return model