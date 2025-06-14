from src.utils.data_preprocessing import DataPreprocessing
from src.config.config_saved_model_dir import (
    HAIRSTYLE_SEG_MODEL_DIR,
    HAIRSTYLE_CLASS_MODEL_DIR
)
from src.config.config_data_loading import IMG_SIZE
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import tensorflow as tf


class HairTypeInference:

    class_names = ["Braids", "Curly", "Dreadlocks", "Kinky", "Short Men", "Straight", "Wavy"]
    
    def __init__(
            self
    ):
        self.img_size = IMG_SIZE
        self.seg_model_path = os.path.join(HAIRSTYLE_SEG_MODEL_DIR, "best_model.keras")
        self.class_model_path = os.path.join(HAIRSTYLE_CLASS_MODEL_DIR, "best_model.keras")
        self.data_preprocessing = DataPreprocessing()
        self.segmantiton_model = self._load_hair_segmentation_model()
        self.classification_model = self._load_hair_classification_model()


    def _load_hair_segmentation_model(
            self
    ) -> keras.Model:
        """
        Kaydedilmiş saç segmentasyon modelini yükler.

        Returns:
            keras.Model: Yüklenen Keras segmentasyon modeli.
        """
        hair_segmentation_model = keras.models.load_model(self.seg_model_path)
        return hair_segmentation_model


    def _load_hair_classification_model(
            self
    ) -> keras.Model:
        """
        Kaydedilmiş saç sınıflandırma modelini yükler.

        Returns:
            keras.Model: Yüklenen Keras sınıflandırma modeli.
        """
        hair_classification_model = keras.models.load_model(self.class_model_path)
        return hair_classification_model


    def input_from_user(
            self,
            image_path: str
    ) -> np.ndarray:
        """
        Kullanıcının sağladığı görüntü yolundan bir görüntü yükler ve önişleme için hazırlar.

        Args:
            image_path (str): Görüntü dosyasının yolu.

        Returns:
            np.ndarray: Görüntü dizisi (NumPy array).
        """
        img = keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


    def segment_hair(
            self,
            image_array: np.ndarray
    ) -> np.ndarray:
        """
        Görüntüdeki saçı segment eder.

        Args:
            image_array (np.ndarray): Giriş görüntü dizisi (256x256, Mobilenet ön işleme tabi tutulmuş).

        Returns:
            np.ndarray: Segmentasyon maskesi dizisi.
        """
        image_array_batched = np.expand_dims(image_array, axis=0)
        segmentation_mask = self.segmantiton_model.predict(image_array_batched)
        
        if segmentation_mask.ndim == 4 and segmentation_mask.shape[0] == 1:
            segmentation_mask = segmentation_mask[0]
            
        return segmentation_mask


    def overlay_hair(
            self,
            image: np.ndarray,
            segmentation_mask: np.ndarray
    ) -> np.ndarray:
        """
        Orijinal görüntüyü segmentasyon maskesi ile birleştirir.

        Args:
            image (np.ndarray): Orijinal görüntü dizisi.
            segmentation_mask (np.ndarray): Segmentasyon maskesi dizisi.

        Returns:
            np.ndarray: Maskelenmiş görüntü dizisi.
        """
        return self.data_preprocessing.masked_image(image, segmentation_mask)


    def classify_hair(
            self,
            image_array: np.ndarray
    ) -> np.ndarray:
        """
        Saçı sınıflandırır.

        Args:
            image_array (np.ndarray): Giriş görüntü dizisi (224x224, Mobilenet ön işleme tabi tutulmuş).

        Returns:
            np.ndarray: Saç tipi olasılıkları dizisi.
        """
        image_array_batched = np.expand_dims(image_array, axis=0)
        hair_type_probs = self.classification_model.predict(image_array_batched)
        return hair_type_probs


    def display_result(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            hair_type_probs: np.ndarray
    ) -> None:
        """
        Orijinal görüntüyü, maskeyi ve saç tipi sınıflandırma sonuçlarını yan yana gösterir.

        Args:
            image (np.ndarray): Orijinal görüntü dizisi.
            mask (np.ndarray): Segmentasyon maskesi dizisi.
            hair_type_probs (np.ndarray): Saç tipi olasılıkları dizisi.
        """
        probabilities = hair_type_probs[0]
        predicted_index = np.argmax(probabilities)
        hair_class = self.class_names[predicted_index]
        probability = probabilities[predicted_index]
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.axis('off')
        plt.title(f"{hair_class}: {probability:.2f}")
        plt.tight_layout()
        plt.show()


    def inference(
            self,
            image_array: np.ndarray
    ) -> tuple[str, float]:
        """
        Saç segmentasyonu ve sınıflandırması çıkarımı yapar.

        Args:
            image_array (np.ndarray): Giriş görüntü dizisi (API'den gelen orijinal hali).

        Returns:
            tuple[str, float]: Tahmin edilen saç tipi ve olasılığı.
        """
        segmentation_input_image = tf.image.resize(image_array, (256, 256))
        segmentation_input_image = np.array(segmentation_input_image)
        segmentation_input_image = keras.applications.mobilenet_v3.preprocess_input(segmentation_input_image)

        segmentation_mask = self.segment_hair(segmentation_input_image)

        image_for_overlay = tf.image.resize(image_array, (256, 256))
        image_for_overlay = np.array(image_for_overlay)

        overlayed_image = self.overlay_hair(image_for_overlay, segmentation_mask)

        classification_ready_image = tf.image.resize(overlayed_image, self.img_size)
        classification_ready_image = np.array(classification_ready_image)
        classification_ready_image = keras.applications.mobilenet_v3.preprocess_input(classification_ready_image)

        raw_hair_type_probs = self.classify_hair(classification_ready_image)
        
        probabilities = raw_hair_type_probs[0]
        predicted_index = np.argmax(probabilities)
        hair_class = self.class_names[predicted_index]
        probability = float(probabilities[predicted_index])

        return hair_class, probability