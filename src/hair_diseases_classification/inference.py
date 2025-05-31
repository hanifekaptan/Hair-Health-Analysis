from src.utils.data_preprocessing import DataPreprocessing
from src.config.config_saved_model_path import *
from src.config.config_data_loading import IMG_SIZE
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


class HairDiseasesInference:

    def __init__(
            self
            ):
        
        self.img_size = IMG_SIZE
        self.model_path = HAIR_DISEASES_CLASS_MODEL_PATH
        self.data_preprocessing = DataPreprocessing()
        self.model = self._load_hair_classification_model()
    

    def _load_hair_classification_model(
            self
            ) -> keras.Model:
        """
        Kaydedilmiş saç hastalıkları sınıflandırma modelini yükler.

        Returns:
            hair_classification_model (keras.Model): Yüklenen Keras sınıflandırma modeli.
        """
        hair_classification_model = keras.models.load_model(self.model_path)
        return hair_classification_model
    

    def input_from_user(
            self,
            image_path: str
    ) -> np.ndarray:
        """
        Kullanıcının sağladığı görüntü yolundan bir görüntü yükler ve ön işleme için hazırlar.

        Args:
            image_path (str): Görüntü dosyasının yolu.

        Returns:
            img_array (np.ndarray): Görüntü dizisi (NumPy array).
        """
        img = keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


    def classify_disease(
            self,
            image_array: np.ndarray
    ) -> np.ndarray:
        """
        Görüntüdeki saçı sınıflandırır.

        Args:
            image_array (np.ndarray): Giriş görüntü dizisi.

        Returns:
            hair_type_probs (np.ndarray): Saç hastalığı olasılıkları dizisi.
        """
        image = self.data_preprocessing.mobilenet_preprocess(image_array)
        hair_type_probs = self.model.predict(image)
        return hair_type_probs
    

    def display_result(
            self,
            image: np.ndarray,
            hair_type_probs: np.ndarray
    ) -> None:
        """
        Orijinal görüntüyü ve saç hastalığı sınıflandırma sonuçlarını gösterir.

        Args:
            image (np.ndarray): Orijinal görüntü dizisi.
            hair_type_probs (np.ndarray): Saç hastalığı olasılıkları dizisi.
        """
        hair_class, probability = hair_type_probs[0]
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{hair_class}: {probability:.2f}")
        plt.show()


    def inference(
            self,
            image_array
            ) -> np.ndarray:
        """
        Saç hastalığı olasılıklarından en yüksek olasılıklı sınıfı ve olasılığı döndürür.

        Args:
            image_array (np.ndarray): Giriş görüntü dizisi.

        Returns:
            tuple[str, float]: En yüksek olasılıklı sınıf adı ve olasılığı.
        """
        hair_type_probs = self.classify_disease(image_array)
        return hair_type_probs
