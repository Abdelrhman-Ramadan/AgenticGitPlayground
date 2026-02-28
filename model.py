import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions

import numpy as np

def train_resnet_classifier(
    train_generator,
    val_generator,
    num_classes,
    input_shape=(224, 224, 3),
    transfer_learning=False,
    fine_tuning=False,
    # mode="transfer",          # "transfer" or "finetune"
    learning_rate=1e-4,
    epochs=10
):
    """
    Training pipeline for image classification using ResNet50.

    Args:
        train_generator: training data generator
        val_generator: validation/test data generator
        num_classes: number of output classes
        input_shape: image input shape
        mode: "transfer" or "finetune"
        learning_rate: optimizer learning rate
        epochs: number of training epochs
    """

    # ==========================
    # 1️⃣ Load Pretrained ResNet
    # ==========================
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        trainable=False
    )

    # ==========================================
    # 2️⃣ Transfer Learning vs Fine-tuning Logic
    # ==========================================

    # if transfer_learning :
    #     # Freeze entire backbone
    #     # To-Do
        
    # elif fine_tuning :
    #     # TODO:
    #     # Here you can unfreeze some top layers
    #     # Example:
    #     # for layer in base_model.layers[-30:]:
    #     #     layer.trainable = True
    #     # base_model.trainable = True

    #     # base_model.trainable = True

    # else:
    #     # raise ValueError("mode must be either 'transfer' or 'finetune'")

    # ==========================
    # 3️⃣ Build Classification Head
    # ==========================
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    # ==========================
    # 4️⃣ Compile Model
    # ==========================
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # ==========================
    # 5️⃣ Train
    # ==========================
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    return model, history
