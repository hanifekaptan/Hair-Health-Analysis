from tensorflow import keras
import os

INPUT_SHAPE = (224, 224, 3)

BATCH_SIZE = 16

EPOCHS = 50

def _get_compiler_params(learning_rate):
    params = {
        "optimizer": keras.optimizers.Adam(learning_rate=learning_rate),
        "loss": keras.losses.CategoricalCrossentropy(),
        "metrics": ["accuracy", "top_k_categorical_accuracy", "precision", "recall"]
    }
    return params

COMPILER_PARAMS = lambda learning_rate: _get_compiler_params(learning_rate)

def _get_callbacks_params(model_dir):
    best_model_path = os.path.join(model_dir, "best_model.keras")
    params = [
        keras.callbacks.ModelCheckpoint(
            filepath = best_model_path,
            monitor = "val_loss",
            save_best_only = True,
            mode = "min",
            verbose = 1,
            save_weights_only = False
        ),
        keras.callbacks.EarlyStopping(
            monitor = "val_loss",
            min_delta = 0,
            patience = 5,
            verbose = 1,
            mode = "auto",
            baseline = None,
            restore_best_weights = True
        ),
        keras.callbacks.CSVLogger(
            os.path.join(model_dir, 'logs', 'training_log.csv'),
            append=True)
        
    ]
    return params

CALLBACKS_PARAMS = lambda model_dir: _get_callbacks_params(model_dir)