# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, MaxPooling2D,
                                     Activation, concatenate)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from model.attention import SoftAttention


def create_data_generators(train_path, test_path, image_size, batch_size):
    """Create and configure data generators for training and testing"""
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
    )

    print("\nTrain Batches: ")
    train_batches = datagen.flow_from_directory(
        directory=train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True
    )

    print("\nTest Batches: ")
    test_batches = datagen.flow_from_directory(
        directory=test_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )

    return train_batches, test_batches


def create_model():
    """Create and configure the model architecture"""
    # Load pre-trained InceptionResNetV2
    irv2 = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax",
    )

    # Get the output of the layer before the last 28 layers
    conv = irv2.layers[-28].output

    # Add attention mechanism
    attention_layer, map2 = SoftAttention(
        aggregate=True,
        m=16,
        concat_with_x=False,
        ch=int(conv.shape[-1]),
        name='soft_attention'
    )(conv)

    # Apply pooling
    attention_layer = MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer)
    conv = MaxPooling2D(pool_size=(2, 2), padding="same")(conv)

    # Combine features
    conv = concatenate([conv, attention_layer])
    conv = Activation('relu')(conv)
    conv = Dropout(0.5)(conv)

    # Output layer
    output = Flatten()(conv)
    output = Dense(7, activation='softmax')(output)

    # Create model
    model = Model(inputs=irv2.input, outputs=output)

    return model


def compile_model(model):
    """Compile the model with optimizer and loss function"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=0.1)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks():
    """Configure training callbacks"""
    checkpoint = ModelCheckpoint(
        filepath='IRV2+SA.hdf5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
        min_delta=0.001
    )

    return [checkpoint, early_stop]


def get_class_weights():
    """Define class weights for handling class imbalance"""
    return {
        0: 1.0,  # akiec
        1: 1.0,  # bcc
        2: 1.0,  # bkl
        3: 1.0,  # df
        4: 5.0,  # mel
        5: 1.0,  # nv
        6: 1.0,  # vasc
    }


def train_model(train_df, test_df):
    """Train the model using preprocessed data"""
    # Configuration
    train_path = 'HAM10000/train_dir'
    test_path = 'HAM10000/test_dir'
    batch_size = 16
    image_size = 299

    # Create data generators
    train_batches, test_batches = create_data_generators(
        train_path, test_path, image_size, batch_size
    )

    # Create and compile model
    model = create_model()
    model = compile_model(model)

    # Get callbacks and class weights
    callbacks = get_callbacks()
    class_weights = get_class_weights()

    # Train model
    history = model.fit(
        train_batches,
        steps_per_epoch=(len(train_df) / 10),
        epochs=150,
        verbose=1,
        validation_data=test_batches,
        validation_steps=len(test_df) / batch_size,
        callbacks=callbacks,
        class_weight=class_weights
    )

    return model, history