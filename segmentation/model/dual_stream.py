import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
    Activation, concatenate, Conv2D, Input, Multiply, MaxPooling2D,
    DepthwiseConv2D, Add
)
from tensorflow.keras.applications import InceptionResNetV2


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
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


class DualStreamModel:
    def __init__(self, input_shape=(299, 299, 3), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _create_depthwise_block(self, x, filters, stride=1, name_prefix=''):
        """Depthwise Separable Convolution with Squeeze-Excitation"""
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

    def _create_fusion_module(self, orig_features, seg_features):
        """Enhanced feature fusion module"""
        # Channel attention for segmentation features
        seg_gap = GlobalAveragePooling2D()(seg_features)
        seg_attention = Dense(seg_features.shape[-1] // 16, activation='relu')(seg_gap)
        seg_attention = Dense(seg_features.shape[-1], activation='sigmoid')(seg_attention)
        seg_attention = tf.reshape(seg_attention, [-1, 1, 1, seg_features.shape[-1]])

        # Apply attention to segmentation features
        attended_seg = Multiply()([seg_features, seg_attention])

        # Concatenate features
        fused = concatenate([orig_features, attended_seg], axis=-1)

        # Feature selection with 1x1 convolution
        fused = Conv2D(orig_features.shape[-1], 1, padding='same')(fused)
        fused = BatchNormalization()(fused)
        fused = Activation('relu')(fused)

        return fused

    def build(self):
        # Input layers
        original_input = Input(shape=self.input_shape, name='original_input')
        seg_input = Input(shape=self.input_shape, name='segmentation_input')

        # Original image stream (InceptionResNetV2)
        irv2_base = InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=original_input
        )
        for layer in irv2_base.layers[:700]:
            layer.trainable = False
        orig_features = irv2_base.get_layer('conv_7b_ac').output  # 8x8x1536

        # Efficient segmentation stream
        x = self._create_depthwise_block(seg_input, 32, stride=2, name_prefix='seg_block1')
        # 150x150

        x = self._create_depthwise_block(x, 64, stride=2, name_prefix='seg_block2')
        # 75x75

        x = self._create_depthwise_block(x, 128, stride=2, name_prefix='seg_block3')
        # 38x38

        x = self._create_depthwise_block(x, 256, stride=2, name_prefix='seg_block4')
        # 19x19

        # Additional reduction to match InceptionResNetV2 output size
        x = MaxPooling2D(2, strides=2, padding='valid')(x)
        # 9x9

        # Final convolution to match size and channels
        x = Conv2D(1536, 2, strides=1, padding='valid')(x)  # パディングをvalidに変更し、カーネルサイズを2に
        x = BatchNormalization()(x)
        seg_features = Activation('relu')(x)
        # Now 8x8x1536

        # Feature fusion
        fused_features = self._create_fusion_module(orig_features, seg_features)

        # Classification head
        x = GlobalAveragePooling2D(name='gap')(fused_features)
        x = BatchNormalization(name='bn_final')(x)
        x = Dropout(0.3)(x)
        output = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=[original_input, seg_input], outputs=output)
        return model

    def compile_model(self, model, learning_rate=0.0001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model