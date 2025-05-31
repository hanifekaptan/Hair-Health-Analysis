from src.config.config_data_loading import *
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2


class DataPreprocessing:

    def __init__(
            self
            ):
        self.image_size = IMG_SIZE


    def mobilenet_preprocess(
            self,
            image: np.ndarray
            ) -> np.ndarray:
        """
        Mobilnet modeli için veri ön işleme yapar

        Args:
            image (np.ndarray): Görüntü

        Returns:
            np.ndarray: Önisleme sonrası veri
        """
        resized_image = tf.image.resize(image, self.image_size)
        image_array = np.array(resized_image)
        image_array = keras.applications.mobilenet_v3.preprocess_input(image_array)
        return image_array
    
    
    def segmentation_preprocess(
            self,
            image: np.ndarray
            ) -> np.ndarray:
        """
        Segmentation modeli için verileri yeniden boyutlandırır, normalize eder ve grayscale yapar

        Args:
            image (np.ndarray): Görüntü

        Returns:
            np.ndarray: Önişleme sonrası veri
        """
        resized_image = tf.image.resize(image, self.image_size)
        image_array = np.array(resized_image)
        image_array = keras.preprocessing.image.normalize(image_array)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        return image_array


    def _segmentation_mask_preprocess(
            self,
            mask: np.ndarray
            ) -> np.ndarray:
        """
        Segmentasyon için maske verilerinin boyutunu kontrol eder ve normalize eder

        Args:
            mask (np.ndarray): Maske verisi

        Returns:
            np.ndarray: Önişleme sonrası veri
        """
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.max() > 1:
            mask = mask / 255.0
        return mask
    

    def masked_image(
            self,
            image_array: np.ndarray,
            mask_array: np.ndarray
            ) -> np.ndarray:
        """
        Maske verilerini kullanarak görüntüyü maskeler

        Args:
            image_array (np.ndarray): Görüntü
            mask_array (np.ndarray): Maske

        Returns:
            np.ndarray: Maskelenmiş görüntü
        """
        mask_array = self._segmentation_mask_preprocess(mask_array)
        masked_image = image_array.astype(np.uint8)
        masked_image[mask_array <= 0.5] = [255, 255, 255]
        return masked_image