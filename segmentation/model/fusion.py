import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization,
    Activation, concatenate
)


class FeatureFusionModule(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Channel attention branch
        self.channel_pool = GlobalAveragePooling2D()
        self.channel_fc1 = Dense(self.channels // self.reduction_ratio)
        self.channel_relu = Activation('relu')
        self.channel_fc2 = Dense(self.channels)
        self.channel_sigmoid = Activation('sigmoid')

        # Cross-modality attention
        self.cross_conv1 = Conv2D(self.channels // 4, 1)
        self.cross_bn1 = BatchNormalization()
        self.cross_relu1 = Activation('relu')
        self.cross_conv2 = Conv2D(self.channels, 1)
        self.cross_sigmoid = Activation('sigmoid')

        # Fusion weights
        self.fusion_conv = Conv2D(2, 1)
        self.fusion_softmax = Activation('softmax')

        super().build(input_shape)

    def call(self, inputs):
        orig_features, seg_features = inputs

        # Channel attention for original features
        channel_att = self.channel_pool(orig_features)
        channel_att = self.channel_fc1(channel_att)
        channel_att = self.channel_relu(channel_att)
        channel_att = self.channel_fc2(channel_att)
        channel_att = self.channel_sigmoid(channel_att)
        channel_att = tf.expand_dims(tf.expand_dims(channel_att, 1), 1)

        orig_refined = orig_features * channel_att

        # Cross-modality attention
        cross_att = self.cross_conv1(seg_features)
        cross_att = self.cross_bn1(cross_att)
        cross_att = self.cross_relu1(cross_att)
        cross_att = self.cross_conv2(cross_att)
        cross_att = self.cross_sigmoid(cross_att)

        seg_refined = seg_features * cross_att

        # Adaptive fusion with separate weights generation
        concat_features = concatenate([orig_refined, seg_refined], axis=-1)
        fusion_weights = self.fusion_conv(concat_features)
        fusion_weights = self.fusion_softmax(fusion_weights)

        # Split weights and apply
        w1 = fusion_weights[..., 0:1]
        w2 = fusion_weights[..., 1:2]

        fused_features = orig_refined * w1 + seg_refined * w2

        return fused_features

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config