from src.utils.data_preprocessing import DataPreprocessing
from src.config.config_saved_model_dir import HAIRSTYLE_SEG_MODEL_DIR
from src.config.config_data_loading import IMG_SIZE
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import os
from PIL import Image


class HairstyleSegmentationInference:

    def __init__(
            self
            ):
        self.model_path = os.path.join(HAIRSTYLE_SEG_MODEL_DIR, "best_model.keras")
        self.img_size = IMG_SIZE
        self.data_preprocessing = DataPreprocessing()
        self.segmentation_model: keras.Model = self._load_hair_segmentation_model()
        self.num_classes = 2


    def _load_hair_segmentation_model(
            self
            ) -> keras.Model:
        """
        Kaydedilmiş saç segmentasyon modelini yükler.

        Returns:
            keras.Model: Yüklenen Keras modeli.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Hair segmentation model not found at: {self.model_path}. Please train the model first.")
        
        hair_segmentation_model = keras.models.load_model(self.model_path, compile=False)
        return hair_segmentation_model


    def input_from_user(
            self,
            image_input
            ) -> np.ndarray:
        """
        Giriş görüntüsünü ön işler. Bir dosya yolu, bir PIL Image nesnesi veya bir NumPy dizisi kabul edebilir.
        Ön işlenmiş bir TensorFlow Tensor (batch boyutu dahil) döndürür.
        """
        if isinstance(image_input, str):
            img = tf.io.read_file(image_input)
            img = tf.image.decode_image(img, channels=3)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
            img = tf.convert_to_tensor(img, dtype=tf.float32)
        elif isinstance(image_input, np.ndarray):
            img = tf.convert_to_tensor(image_input, dtype=tf.float32)
        elif tf.is_tensor(image_input):
            img = tf.cast(image_input, dtype=tf.float32)
        else:
            raise ValueError(f"Desteklenmeyen görüntü giriş türü: {type(image_input)}. Beklenen: dosya yolu (str), PIL Image, NumPy dizisi veya TensorFlow Tensor.")

        if img.shape[-1] == 1:
            img = tf.image.grayscale_to_rgb(img)
        elif img.shape[-1] != 3:
            img = img[..., :3]
        if len(img.shape) == 3:
            img = tf.expand_dims(img, axis=0)
        elif len(img.shape) != 4:
            raise ValueError(f"Görüntü girişi için 3 veya 4 boyut bekleniyordu, {len(img.shape)} alındı.")
        
        img = tf.image.resize(img, self.img_size, method=tf.image.ResizeMethod.BILINEAR) # Görüntüler için BILINEAR kullanın
        img = img / 255.0

        return img


    def segment_hair(
            self,
            image_array: np.ndarray
            ) -> np.ndarray:
        """
        Görüntüdeki saçı segment eder.

        Args:
            image_array (np.ndarray): Giriş görüntü dizisi.

        Returns:
            np.ndarray: Saç maskesi dizisi.
        """
        image_processed = self.data_preprocessing.mobilenet_preprocess(image_array)
        hair_mask = self.segmentation_model.predict(image_processed)
        return hair_mask


    def display_result(
            self,
            image: np.ndarray,
            hair_mask: np.ndarray
            ) -> None:
        """
        Orijinal görüntüyü ve segment edilmiş saç maskesini yan yana gösterir.

        Args:
            image (np.ndarray): Orijinal görüntü dizisi.
            hair_mask (np.ndarray): Saç maskesi dizisi.
        """
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(image).astype(np.uint8))
        plt.axis('off')
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(np.argmax(hair_mask, axis=-1)), cmap='gray')
        plt.axis('off')
        plt.title("Segmented Hair")
        plt.tight_layout()
        plt.show()


    def inference(
            self,
            image_data
            ) -> np.ndarray:
        """
        Saç segmentasyon çıkarımı yapar.

        Args:
            image_data: Görüntü verisi.

        Returns:
            np.ndarray: Saç maskesi dizisi.
        """
        preprocessed_img = self.input_from_user(image_data)
        predictions = self.segment_hair(preprocessed_img)
        if self.num_classes > 1:
            segmented_mask = tf.argmax(predictions, axis=-1)
        else:
            segmented_mask = (predictions > 0.5)
            if segmented_mask.shape.rank == 4:
                segmented_mask = tf.squeeze(segmented_mask, axis=-1)
        
        segmented_mask = tf.squeeze(segmented_mask, axis=0)
        segmented_mask = tf.cast(segmented_mask, tf.uint8) * 255
        return segmented_mask.numpy()