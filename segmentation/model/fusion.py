import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization,
    Activation, concatenate, Multiply, Layer, GlobalMaxPooling2D,
    Reshape, Add
)


class ChannelGatingBlock(Layer):
    """Channel attention block with gating mechanism"""

    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # Layers
        self.gap = GlobalAveragePooling2D()
        self.gmp = GlobalMaxPooling2D()
        self.fc1 = Dense(channels // reduction_ratio)
        self.fc2 = Dense(channels)
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')

    def call(self, x):
        # Average pooling branch
        avg_pool = self.gap(x)
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.relu(avg_pool)
        avg_pool = self.fc2(avg_pool)

        # Max pooling branch
        max_pool = self.gmp(x)
        max_pool = self.fc1(max_pool)
        max_pool = self.relu(max_pool)
        max_pool = self.fc2(max_pool)

        # Combine attention
        attention = self.sigmoid(avg_pool + max_pool)
        attention = tf.reshape(attention, [-1, 1, 1, self.channels])

        return x * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config


class SpatialGatingBlock(Layer):
    """Spatial attention block with gating mechanism"""

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

        # Layers
        self.conv = Conv2D(1, kernel_size, padding='same')
        self.bn = BatchNormalization()
        self.sigmoid = Activation('sigmoid')

    def call(self, x):
        # Calculate channel-wise statistics
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

        # Concatenate and process
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_gate = self.conv(concat)
        spatial_gate = self.bn(spatial_gate)
        spatial_gate = self.sigmoid(spatial_gate)

        return x * spatial_gate

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config


class StreamFeatureProcessor(Layer):
    """Process features from a single stream with channel and spatial attention"""

    def __init__(self, channels, reduction_ratio=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        # Attention blocks
        self.channel_gate = ChannelGatingBlock(channels, reduction_ratio)
        self.spatial_gate = SpatialGatingBlock(kernel_size)

        # Feature refinement
        self.conv1x1 = Conv2D(channels, 1, padding='same')
        self.bn = BatchNormalization()
        self.relu = Activation('relu')

    def call(self, x):
        # Apply channel attention
        x = self.channel_gate(x)

        # Apply spatial attention
        x = self.spatial_gate(x)

        # Refine features
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size
        })
        return config


class AdaptiveFeatureFusion(Layer):
    """Adaptive fusion of features from multiple streams"""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

        # Fusion weights generation
        self.weight_conv = Conv2D(3, 1, padding='same')  # 3 weights for 3 streams
        self.weight_bn = BatchNormalization()
        self.weight_softmax = Activation('softmax')

        # Feature refinement
        self.fusion_conv = Conv2D(channels, 1, padding='same')
        self.fusion_bn = BatchNormalization()
        self.fusion_relu = Activation('relu')

    def call(self, inputs):
        main_features, boundary_features, color_features = inputs

        # Concatenate features for weight generation
        concat_features = concatenate([main_features, boundary_features, color_features], axis=-1)

        # Generate fusion weights
        weights = self.weight_conv(concat_features)
        weights = self.weight_bn(weights)
        weights = self.weight_softmax(weights)

        # Split weights for each stream
        w1, w2, w3 = tf.split(weights, 3, axis=-1)

        # Weighted fusion
        fused = (main_features * w1 +
                 boundary_features * w2 +
                 color_features * w3)

        # Final refinement
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused)
        fused = self.fusion_relu(fused)

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels
        })
        return config


class TripleStreamFusion(Layer):
    """Complete fusion module for triple stream architecture"""

    def __init__(self, main_channels=1536, boundary_channels=384, color_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.main_channels = main_channels
        self.boundary_channels = boundary_channels
        self.color_channels = color_channels

        # Stream processors
        self.main_processor = StreamFeatureProcessor(main_channels)
        self.boundary_processor = StreamFeatureProcessor(boundary_channels)
        self.color_processor = StreamFeatureProcessor(color_channels)

        # Dimension matching for boundary and color streams
        self.boundary_matcher = Conv2D(main_channels, 1, padding='same')
        self.color_matcher = Conv2D(main_channels, 1, padding='same')

        # Adaptive fusion
        self.fusion = AdaptiveFeatureFusion(main_channels)

    def call(self, inputs):
        main_features, boundary_features, color_features = inputs

        # Process each stream
        main_proc = self.main_processor(main_features)
        boundary_proc = self.boundary_processor(boundary_features)
        color_proc = self.color_processor(color_features)

        # Match dimensions
        boundary_matched = self.boundary_matcher(boundary_proc)
        color_matched = self.color_matcher(color_proc)

        # Fuse features
        fused = self.fusion([main_proc, boundary_matched, color_matched])

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            "main_channels": self.main_channels,
            "boundary_channels": self.boundary_channels,
            "color_channels": self.color_channels
        })
        return config