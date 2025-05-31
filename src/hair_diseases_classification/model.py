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
        # Use a functional approach to connect the input layer to the base model and subsequent layers
        x = base_model(inputs) # Pass the defined inputs to the base model
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

        # Create the Keras Model using the specified inputs and outputs
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model