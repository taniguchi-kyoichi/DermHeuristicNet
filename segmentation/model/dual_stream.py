import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
    Activation, concatenate, Conv2D, Input, Multiply, MaxPooling2D
)
from tensorflow.keras.applications import InceptionResNetV2
from .fusion import FeatureFusionModule


class DualStreamModel:
    def __init__(self, input_shape=(299, 299, 3), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _create_border_attention(self, x, name_prefix):
        """Create border-aware attention module"""
        edge_filters = Conv2D(32, (3, 3), padding='same', name=f'{name_prefix}_edge1')(x)
        edge_filters = BatchNormalization()(edge_filters)
        edge_filters = Activation('relu')(edge_filters)

        edge_filters = Conv2D(32, (3, 3), padding='same', name=f'{name_prefix}_edge2')(edge_filters)
        edge_filters = BatchNormalization()(edge_filters)
        edge_filters = Activation('relu')(edge_filters)

        attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid',
                           name=f'{name_prefix}_attention')(edge_filters)

        return Multiply(name=f'{name_prefix}_attended')([x, attention])

    def build(self):
        # Input layers
        original_input = Input(shape=self.input_shape, name='original_input')
        seg_input = Input(shape=self.input_shape, name='segmentation_input')

        # Original image stream with InceptionResNetV2
        irv2_base = InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=original_input
        )

        # Freeze early layers
        for layer in irv2_base.layers[:700]:
            layer.trainable = False

        # Extract features from IRV2 (8x8x1536)
        orig_features = irv2_base.get_layer('conv_7b_ac').output

        # Segmentation stream with careful size control
        # Initial size: 299x299
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='seg_conv1')(seg_input)
        x = BatchNormalization(name='seg_bn1')(x)
        x = Activation('relu', name='seg_act1')(x)
        # Size: 150x150

        x = MaxPooling2D(3, strides=2, padding='same', name='seg_pool1')(x)
        # Size: 75x75

        x = Conv2D(128, (3, 3), padding='same', name='seg_conv2')(x)
        x = BatchNormalization(name='seg_bn2')(x)
        x = Activation('relu', name='seg_act2')(x)
        x = MaxPooling2D(3, strides=2, padding='same', name='seg_pool2')(x)
        # Size: 38x38

        x = Conv2D(256, (3, 3), padding='same', name='seg_conv3')(x)
        x = BatchNormalization(name='seg_bn3')(x)
        x = Activation('relu', name='seg_act3')(x)
        x = MaxPooling2D(3, strides=2, padding='same', name='seg_pool3')(x)
        # Size: 19x19

        x = Conv2D(512, (3, 3), padding='same', name='seg_conv4')(x)
        x = BatchNormalization(name='seg_bn4')(x)
        x = Activation('relu', name='seg_act4')(x)
        x = MaxPooling2D(2, strides=2, padding='same', name='seg_pool4')(x)
        # Size: 10x10

        # Final convolution to match size and channels
        seg_features = Conv2D(1536, (3, 3), strides=1, padding='valid', name='seg_conv5')(x)
        seg_features = BatchNormalization(name='seg_bn5')(seg_features)
        seg_features = Activation('relu', name='seg_act5')(seg_features)
        # Size: 8x8x1536

        # Apply border attention to both streams
        attended_orig = self._create_border_attention(orig_features, 'orig')
        attended_seg = self._create_border_attention(seg_features, 'seg')

        # Feature fusion
        fusion_module = FeatureFusionModule(channels=1536)
        fused_features = fusion_module([attended_orig, attended_seg])

        # Classification head
        x = GlobalAveragePooling2D(name='gap')(fused_features)
        x = BatchNormalization(name='bn_final')(x)

        # Multi-scale feature aggregation
        x1 = Dense(512, activation='relu', name='dense_512')(x)
        x2 = Dense(256, activation='relu', name='dense_256')(x)
        x3 = Dense(128, activation='relu', name='dense_128')(x)

        multi_scale = concatenate([x1, x2, x3], name='multi_scale_concat')

        x = BatchNormalization(name='bn_multi_scale')(multi_scale)
        x = Dropout(0.3, name='dropout')(x)

        # Output layer
        output = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create model
        model = Model(inputs=[original_input, seg_input], outputs=output)
        return model

    def compile_model(self, model, learning_rate=0.0001):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0
        )

        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        )

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        return model