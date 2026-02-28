# mkdir -p chess_data
# kaggle datasets download -d niteshfre/chessman-image-dataset -p chess_data
# unzip the data

from pathlib import Path
import tensorflow as tf

def load_train_test_keras(base_dir=None,
                         image_size=(224, 224),
                         batch_size=32,
                         test_split=0.2,
                         seed=42):
    # base_dir should be the folder that contains Bishop/King/... subfolders
    if base_dir is None:
        here = Path(__file__).resolve().parent
        base_dir = here / "chess_data" / "Chessman-image-dataset" / "Chess"
    base_dir = str(base_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=test_split,
        subset="training",
    )  # subset+validation_split create the split

    test_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=test_split,
        subset="validation",
    )  # same split settings + same seed => complementary split

    return train_ds, test_ds, train_ds.class_names

# Example usage + print something
train_ds, test_ds, class_names = load_train_test_keras()
print("Classes:", class_names)

x_batch, y_batch = next(iter(train_ds))
print("Train batch images:", x_batch.shape, "labels:", y_batch.shape)

x2, y2 = next(iter(test_ds))
print("Test batch images:", x2.shape, "labels:", y2.shape)
