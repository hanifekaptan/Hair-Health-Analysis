from src.config.config_model import INPUT_SHAPE
from tensorflow import keras

class HairstyleClassificationModel:

    def mobilenet_model(
            self,
            num_classes: int,
            input_shape: tuple[int, int, int] = INPUT_SHAPE
    ) -> keras.Model:
        """
        MobileNetV2 mimarisine dayalı bir saç tipi sınıflandırma modeli oluşturur.

        Args:
            num_classes (int): Sınıf sayısı (saç tipi sayısı).
            input_shape (tuple[int, int, int]): Giriş görüntüsünün boyutu (yükseklik, genişlik, kanal).
                                                Varsayılan olarak INPUT_SHAPE kullanılır.

        Returns:
            keras.Model: Oluşturulan Keras modeli.
        """
        inputs = keras.Input(shape=input_shape)
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        tensor = base_model.output
        tensor = keras.layers.GlobalAveragePooling2D()(tensor)
        tensor = keras.layers.Dense(32, activation="relu")(tensor)
        tensor = keras.layers.Dense(16, activation="relu")(tensor)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(tensor)
        model = keras.Model(inputs=inputs, outputs=outputs)

        return model