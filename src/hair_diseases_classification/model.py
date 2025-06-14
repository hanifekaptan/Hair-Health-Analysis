from tensorflow import keras

class HairDiseasesClassificationModel:

    def __init__(
            self
    ):
        pass


    def mobilenet_model(
            self,
            num_classes: int,
            input_shape: tuple[int, int, int] = (224, 224, 3)
    ) -> keras.Model:
        """
        MobileNetV2 mimarisine dayalı bir saç hastalıkları sınıflandırma modeli oluşturur.

        Args:
            num_classes (int): Sınıf sayısı (saç hastalığı sayısı).
            input_shape (Tuple[int, int, int]): Giriş görüntüsünün boyutu (yükseklik, genişlik, kanal).
                                                Varsayılan olarak (224, 224, 3) olarak ayarlanmıştır.

        Returns:
            model (keras.Model): Oluşturulan Keras modeli.
        """

        inputs = keras.Input(shape=input_shape)
        base_model = keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    weights='imagenet',
                                                    input_tensor=inputs)
        base_model.trainable = False
        tensor = base_model(inputs)
        tensor= keras.layers.GlobalAveragePooling2D()(tensor)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(tensor)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model