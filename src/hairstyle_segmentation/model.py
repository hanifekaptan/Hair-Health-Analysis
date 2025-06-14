from tensorflow import keras


class HairstyleSegmentation:

    def __init__(
            self
            ):
        self.model = None


    def _mobilenet_separable_conv(
            self,
            input_tensor: keras.layers.Layer,
            channel: int
    ) -> keras.layers.Layer:
        """
        MobileNet mimarisine özgü ayrılabilir evrişim katmanını oluşturur.

        Args:
            input_tensor: Giriş tensörü (keras.layers.Layer).
            channel: Çıkış kanalı sayısı (int).

        Returns:
            ReLU aktivasyonlu ayrılabilir evrişim katmanı (keras.layers.Layer).
        """
        tensor = keras.layers.DepthwiseConv2D(kernel_size=3, padding="same")(input_tensor)
        tensor = keras.layers.Conv2D(filters=channel, kernel_size=1, padding="same")(tensor)
        relu = keras.layers.ReLU()(tensor)
        return relu


    def _mobilenet_upsampling(
            self,
            skip_connection_tensor: keras.layers.Layer,
            upsampled_path_tensor: keras.layers.Layer,
            target_filters: int
    ) -> keras.layers.Layer:
        """
        MobileNet'in yukarı örnekleme ve atlama bağlantılarını birleştirme mantığı.
        """
        target_height = skip_connection_tensor.shape[1]
        target_width = skip_connection_tensor.shape[2]
        up_sampling_height_factor = target_height // upsampled_path_tensor.shape[1]
        up_sampling_width_factor = target_width // upsampled_path_tensor.shape[2]

        if up_sampling_height_factor > 1 or up_sampling_width_factor > 1:
            upsampled_path_tensor_spatially_adjusted = keras.layers.UpSampling2D(
                size=(up_sampling_height_factor, up_sampling_width_factor),
                interpolation="bilinear"
            )(upsampled_path_tensor)
        else:
            upsampled_path_tensor_spatially_adjusted = upsampled_path_tensor

        target_channels = skip_connection_tensor.shape[-1] 
        upsampled_path_tensor_fully_adjusted = keras.layers.Conv2D(
            filters=target_channels,
            kernel_size=(1, 1),
            padding='same',
            name=f"upsampling_conv_adjust_channels_{target_channels}_{target_height}x{target_width}" # Benzersiz bir isim
        )(upsampled_path_tensor_spatially_adjusted)
        tensor = keras.layers.Add()([skip_connection_tensor, upsampled_path_tensor_fully_adjusted])
        return tensor


    def mobilenet_model(
            self,
            input_shape: tuple[int, int, int]
    ) -> keras.Model:
        """
        MobileNetV2 temelli saç stili segmentasyon modelini oluşturur.

        Doğruluk: 0.78

        Args:
            input_shape: Giriş görüntüsünün şekli (yükseklik, genişlik, kanal).
                         Bu fonksiyon için (Y, G, 3) şeklinde beklenir.

        Returns:
            Oluşturulan Keras modeli (keras.Model).
        """
        input_tensor = keras.Input(shape=input_shape)
        base_model = keras.applications.MobileNetV2(
            input_tensor=input_tensor,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False

        layer_names = [
            "block_1_expand_relu",   # ~112x112
            "block_3_expand_relu",   # ~56x56
            "block_6_expand_relu",   # ~28x28
            "block_13_expand_relu"   # ~14x14
        ]
        mobile_net_sublayers = {name: base_model.get_layer(name).output for name in layer_names}
        tensor = base_model.get_layer("out_relu").output # MobileNetV2'nin çıktısı
        tensor = self._mobilenet_upsampling(mobile_net_sublayers["block_13_expand_relu"], tensor, 1024)
        tensor = self._mobilenet_separable_conv(tensor, 64)
        tensor = self._mobilenet_upsampling(mobile_net_sublayers["block_6_expand_relu"], tensor, 64)
        tensor = self._mobilenet_separable_conv(tensor, 64)
        tensor = self._mobilenet_upsampling(mobile_net_sublayers["block_3_expand_relu"], tensor, 64)
        tensor = self._mobilenet_separable_conv(tensor, 64)
        tensor = self._mobilenet_upsampling(mobile_net_sublayers["block_1_expand_relu"], tensor, 64)
        tensor = self._mobilenet_separable_conv(tensor, 64)
        tensor = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(tensor)
        tensor = self._mobilenet_separable_conv(tensor, 64)
        tensor = keras.layers.Conv2D(filters=2, kernel_size=1, padding="same")(tensor) 
        tensor = keras.layers.Softmax()(tensor) 
        model = keras.Model(inputs=input_tensor, outputs=tensor)
        return model


    def _semantic_segmentation_feature_extraction(
            self,
            tensor: keras.layers.Layer,
            filters: int,
            previous_block_activation: keras.layers.Layer
    ) -> tuple[keras.layers.Layer, keras.layers.Layer]:
        """
        Semantik segmentasyon için özellik çıkarma bloğunu oluşturur.

        Args:
            tensor: Giriş tensörü (keras.layers.Layer).
            filters: Evrişim filtrelerinin sayısı (int).
            previous_block_activation: Önceki bloktan aktivasyon (keras.layers.Layer).

        Returns:
            Özellik çıkarılmış tensör ve güncellenmiş önceki blok aktivasyonu (tuple[keras.layers.Layer, keras.layers.Layer]).
        """
        tensor = keras.layers.Activation("relu")(tensor)
        tensor = keras.layers.SeparableConv2D(filters, 3, padding="same")(tensor)
        tensor = keras.layers.BatchNormalization()(tensor)
        tensor = keras.layers.Activation("relu")(tensor)
        tensor = keras.layers.SeparableConv2D(filters, 3, padding="same")(tensor)
        tensor = keras.layers.BatchNormalization()(tensor)
        tensor = keras.layers.MaxPooling2D(3, strides=2, padding="same")(tensor)
        residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        tensor = keras.layers.add([tensor, residual])
        previous_block_activation = tensor
        return tensor, previous_block_activation


    def _semantic_segmentation_upsampling(
            self,
            tensor: keras.layers.Layer,
            filters: int,
            previous_block_activation: keras.layers.Layer
    ) -> tuple[keras.layers.Layer, keras.layers.Layer]:
        """
        Semantik segmentasyon için yukarı örnekleme bloğunu oluşturur.

        Args:
            tensor: Giriş tensörü (keras.layers.Layer).
            filters: Evrişim filtrelerinin sayısı (int).
            previous_block_activation: Önceki bloktan aktivasyon (keras.layers.Layer).

        Returns:
            Yukarı örneklenmiş tensör ve güncellenmiş önceki blok aktivasyonu (tuple[keras.layers.Layer, keras.layers.Layer]).
        """
        tensor = keras.layers.Activation("relu")(tensor)
        tensor = keras.layers.Conv2DTranspose(filters, 3, padding="same")(tensor)
        tensor = keras.layers.BatchNormalization()(tensor)
        tensor = keras.layers.Activation("relu")(tensor)
        tensor = keras.layers.Conv2DTranspose(filters, 3, padding="same")(tensor)
        tensor = keras.layers.BatchNormalization()(tensor)
        tensor = keras.layers.UpSampling2D(2)(tensor)
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        tensor = keras.layers.add([tensor, residual])
        previous_block_activation = tensor
        return tensor, previous_block_activation


    def semantic_segmentation_model(
            self,
            input_shape: tuple[int, int, int]
    ) -> keras.Model:
        """
        Semantik segmentasyon modelini oluşturur.

        Doğruluk: 0.94

        Args:
            input_shape: Giriş görüntüsünün şekli (yükseklik, genişlik, kanal) (tuple[int, int, int]).

        Returns:
            Oluşturulan Keras modeli (keras.Model).
        """
        adjusted_input_shape = (input_shape[0], input_shape[1], 1)
        
        input_tensor = keras.Input(shape=adjusted_input_shape) # Burayı güncelledik
        tensor = keras.layers.Conv2D(32, 3, strides=2, padding="same")(input_tensor)
        tensor = keras.layers.BatchNormalization()(tensor)
        tensor = keras.layers.Activation("relu")(tensor)
        previous_block_activation = tensor
        tensor, previous_block_activation = self._semantic_segmentation_feature_extraction(tensor, 64, previous_block_activation)
        tensor, previous_block_activation = self._semantic_segmentation_feature_extraction(tensor, 128, previous_block_activation)
        tensor, previous_block_activation = self._semantic_segmentation_feature_extraction(tensor, 256, previous_block_activation)
        tensor, previous_block_activation = self._semantic_segmentation_upsampling(tensor, 256, previous_block_activation)
        tensor, previous_block_activation = self._semantic_segmentation_upsampling(tensor, 128, previous_block_activation)
        tensor, previous_block_activation = self._semantic_segmentation_upsampling(tensor, 64, previous_block_activation)
        tensor, previous_block_activation = self._semantic_segmentation_upsampling(tensor, 32, previous_block_activation)
        outputs = keras.layers.Conv2D(3, 3, activation="softmax", padding="same")(tensor)
        model = keras.Model(inputs=input_tensor, outputs=outputs)
        return model