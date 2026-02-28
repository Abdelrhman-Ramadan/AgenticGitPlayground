import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


def train_resnet_classifier(
    train_generator,
    val_generator,
    num_classes,
    input_shape=(224, 224, 3),
    fine_tune=False,
    learning_rate=1e-4,
    epochs=10
):

    # ==========================
    # 1️⃣ Load ResNet50
    # ==========================
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # ==================================
    # 2️⃣ Boolean Switching Logic
    # ==================================
    if fine_tune:
        print("Fine-Tuning Mode")

        base_model.trainable = True
        # train only top part of ResNet
        fine_tune_at = int(len(base_model.layers) * 0.75)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    else:
        print("Transfer Learning Mode")
        # freeze entire backbone
        base_model.trainable = False

    # ==========================
    # 3️⃣ Classification Head
    # ==========================
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # ==========================
    # 4️⃣ Compile
    # ==========================
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ==========================
    # 5️⃣ Train
    # ==========================
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    return model, history