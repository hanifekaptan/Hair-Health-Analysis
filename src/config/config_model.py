import datetime
from tensorflow import keras
import os
from datetime import datetime

INPUT_SHAPE = (224, 224, 3)

BATCH_SIZE = 16

EPOCHS = 50

def COMPILER_PARAMS_CLASS(learning_rate):
    params = {
        "optimizer": keras.optimizers.Adam(learning_rate=learning_rate),
        "loss": keras.losses.CategoricalCrossentropy(),
        "metrics": ["accuracy", "top_k_categorical_accuracy", "precision", "recall"]
    }
    return params


def CALLBACKS_PARAMS_CLASS(best_model_dir):
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir, exist_ok=True)

    log_dir = os.path.join(best_model_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(best_model_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            mode="min",
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1,
            mode="min"
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_training_log.csv"),
            separator=",",
            append=True
        )
    ]
    return callbacks

def COMPILER_PARAMS_SEG(learning_rate):
    params = {
        "optimizer": keras.optimizers.Adam(learning_rate=learning_rate),
        "loss": keras.losses.CategoricalCrossentropy(),
        "metrics": [
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.MeanIoU(num_classes=2),
            "accuracy"
        ]
    }
    return params

def CALLBACKS_PARAMS_SEG(best_model_dir: str):
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir, exist_ok=True)

    log_dir = os.path.join(best_model_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(best_model_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_training_log.csv"),
            separator=",",
            append=True
        )
    ]
    return callbacks

