from typing import Generator
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    log_loss)
from tensorflow import keras
import seaborn as sns
import numpy as np


class Evaluation:

    def __init__(
            self,
            model_path: str
            ):
        
        self.keras_model = keras.models.load_model(model_path)

        
class ClassificationEvaluation(Evaluation):

    def __init__(
            self,
            test_data_gen: keras.preprocessing.image.DirectoryIterator,
            model_path: str
            ):
        super().__init__(model_path)
        self.test_gen = test_data_gen
        self.class_names = self.test_gen.class_indices.keys()
        self.model = self.keras_model

        
    def _get_y_pred(
            self
            ) -> np.ndarray:
        """
        Test verileri ile modeli tahmin eder ve tahmin sonuçlarını (olasılıklar) döndürür.
        
        Args:

        Returns:
            np.ndarray: Tahmin olasılıkları
        """
        predictions = self.model.predict(self.test_gen)
        return predictions
    
    
    def _get_y_true(
            self
            ) -> np.ndarray:
        """
        Gerçek etiketleri alır ve döndürür.
        
        Args:

        Returns:
            np.ndarray: Gerçek etiketler
        """
        return self.test_gen.classes
        
    
    def _get_classification_report(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
            ) -> str | dict:
        """
        Sınıflandırma raporunu döndürür.
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler

        Returns:
            str | dict: Sınıflandırma raporu
        """
        classification_rep = classification_report(y_true, y_pred, target_names=self.class_names) 
        return classification_rep
    

    def _get_confusion_matrix(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
            ) -> np.ndarray:
        """
        Karışıklık matrisini döndürür.
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler

        Returns:
            np.ndarray: Karışıklık matrisi
        """
        confusion_mat = confusion_matrix(y_true, y_pred)
        return confusion_mat
    

    def _get_accuracy(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
            ) -> np.float64:
        """
        Doğruluk oranını döndürür.
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler

        Returns:
            float: Doğruluk oranı
        """
        accuracy = accuracy_score(y_true, y_pred)
        return np.float64(accuracy)
    

    def _get_loss(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
            ) -> np.float64:
        """
        Kayıp oranını döndürür.
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler

        Returns:
            float: Kayıp oranı
        """
        y_true_one_hot = keras.utils.to_categorical(y_true, num_classes=len(self.class_names))
        loss = log_loss(y_true_one_hot, y_pred)
        return np.float64(loss)


    def print_results(
            self,
            classification_report: str | dict,
            confusion_matrix: np.ndarray,
            accuracy: np.float64,
            loss: np.float64
            ) -> None:
        """
        Sonucu yazdırır.
        
        Args:
            classification_report (str | dict): Sınıflandırma raporu
            confusion_matrix (np.ndarray): Karışıklık matrisi
            accuracy (np.float64): Doğruluk oranı
            loss (np.float64): Kayıp oranı

        Returns:
            None
        """
        print("Sınıflandırma Raporu:")
        print(classification_report)
        print("Karışıklık Matrisi:")
        print(confusion_matrix)
        print("Doğruluk Oranı:", accuracy)
        print("Kayıp Oranı:", loss)


    def visualize_confusion_matrix(
            self,
            confusion_mat: np.ndarray
            ) -> None:
        """
        Karışıklık matrisini görüntüleme fonksiyonu.
        
        Args:
            confusion_mat (np.ndarray): Karışıklık matrisi

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names) # Sınıf isimlerini ekle
        plt.xlabel('Tahmin Edilen Etiketler')
        plt.ylabel('Gerçek Etiketler')
        plt.title('Karmaşıklık Matrisi')
        plt.show()


    def evaluation(
            self
            ) -> None:
        """
        Modeli test verileriyle değerlendirir ve sonuçları yazdırır.
        
        Args:
            None

        Returns:
            None
        """
        y_true = self._get_y_true()
        predictions = self._get_y_pred()
        y_pred_classes = np.argmax(predictions, axis=1)

        classification_rep = self._get_classification_report(y_true, y_pred_classes)
        confusion_mat = self._get_confusion_matrix(y_true, y_pred_classes)
        accuracy = self._get_accuracy(y_true, y_pred_classes)
        loss = self._get_loss(y_true, predictions)

        self.print_results(classification_rep, confusion_mat, accuracy, loss)
        self.visualize_confusion_matrix(confusion_mat)

class SegmentationEvaluation(Evaluation):

    def __init__(
            self,
            test_data_gen: Generator,
            model_path: str,
            num_classes: int = 2
            ):
        
        super().__init__(model_path)
        self.model = self.keras_model
        self.test_gen = test_data_gen
        self.num_classes = num_classes


    def _get_y_pred(
            self
            ) -> list[np.ndarray]:
        """
        Test verileri ile modeli tahmin eder ve tahmin edilen maskeleri döndürür.

        Args:
            None

        Returns:
            list[np.ndarray]: Tahmin edilen maskeler
        """
        predictions = self.model.predict(self.test_gen)
        y_pred = np.argmax(predictions, axis=-1)  
        return list(y_pred)


    def _get_y_true(
            self
            ) -> list[np.ndarray]:
        """
        Gerçek maskeleri alır ve döndürür.
        
        Args:
            None

        Returns:
            list[np.ndarray]: Gerçek maskeler
        """
        y_true = []
        for _, mask in self.test_gen:
            y_true.extend(np.argmax(mask, axis=-1).flatten())
        return y_true


    def _calculate_iou(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            class_id: int
            ) -> float:
        """
        Belirli bir sınıf için IoU (Intersection over Union) hesaplar.

        Args:
            y_true (np.ndarray): Gerçek maskeler
            y_pred (np.ndarray): Tahmin edilen maskeler
            class_id (int): IoU'nun hesaplanacağı sınıfın indeksi

        Returns:
            float: IoU değeri
        """
        y_true_class = (y_true == class_id).astype(np.bool_)
        y_pred_class = (y_pred == class_id).astype(np.bool_)
        
        intersection = np.logical_and(y_true_class, y_pred_class).sum()
        union = np.logical_or(y_true_class, y_pred_class).sum()
        if union == 0:
            return 0.0
        return intersection / union


    def _calculate_mean_iou(
            self,
            y_true: list[np.ndarray],
            y_pred: list[np.ndarray]
            ) -> float:
        """
        Tüm sınıflar için ortalama IoU (Intersection over Union) hesaplar.

        Args:
            y_true (list[np.ndarray]): Gerçek maskelerin listesi
            y_pred (list[np.ndarray]): Tahmin edilen maskelerin listesi

        Returns:
            float: Ortalama IoU değeri
        """
        total_iou = 0.0
        for class_id in range(self.num_classes):
            iou = self._calculate_iou(np.array(y_true), np.array(y_pred), class_id)
            total_iou += iou
        return total_iou / self.num_classes


    def _calculate_dice_coefficient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            class_id: int
            ) -> float:
        """
        Belirli bir sınıf için Dice katsayısını hesaplar.

        Args:
            y_true (np.ndarray): Gerçek maskeler
            y_pred (np.ndarray): Tahmin edilen maskeler
            class_id (int): Dice katsayısının hesaplanacağı sınıfın indeksi

        Returns:
            float: Dice katsayısı değeri
        """
        y_true_class = (y_true == class_id).astype(np.bool_)
        y_pred_class = (y_pred == class_id).astype(np.bool_)
        
        intersection = np.sum(y_true_class & y_pred_class)
        total_pixels = np.sum(y_true_class) + np.sum(y_pred_class)
        if total_pixels == 0:
            return 0.0
        return 2.0 * intersection / total_pixels


    def _calculate_mean_dice_coefficient(
            self,
            y_true: list[np.ndarray],
            y_pred: list[np.ndarray]
            ) -> float:
        """
        Tüm sınıflar için ortalama Dice katsayısını hesaplar.

        Args:
            y_true (list[np.ndarray]): Gerçek maskelerin listesi
            y_pred (list[np.ndarray]): Tahmin edilen maskelerin listesi

        Returns:
            float: Ortalama Dice katsayısı değeri
        """
        total_dice = 0.0
        for class_id in range(self.num_classes):
            dice = self._calculate_dice_coefficient(np.array(y_true), np.array(y_pred), class_id)
            total_dice += dice
        return total_dice / self.num_classes
    
    
    def visualize_predictions(
            self,
            num_images: int = 5
            ) -> None:
        """
        Tahminleri görselleştirir.
        
        Args:
            num_images (int): Görselleştirilecek resim sayısı, varsayılan 5

        Returns:
            None
        """
        plt.figure(figsize=(15, 5 * num_images))
        for i in range(num_images):
            img_batch, mask_batch = next(self.test_gen)
            img = img_batch[0]
            true_mask = np.argmax(mask_batch[0], axis=-1)
            pred_mask = np.argmax(self.model.predict(img_batch)[0], axis=-1)
            plt.subplot(num_images, 3, i * 3 + 1)
            plt.imshow(img)
            plt.title("Görüntü")
            plt.axis('off')
            plt.subplot(num_images, 3, i * 3 + 2)
            plt.imshow(true_mask, cmap='gray')
            plt.title("Gerçek Maske")
            plt.axis('off')
            plt.subplot(num_images, 3, i * 3 + 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title("Tahmin Edilen Maske")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        
    def evaluation(
            self
            ) -> None:
        """
        Modeli test verileriyle değerlendirir ve sonuçları yazdırır.
        
        Args:
            None

        Returns:
            None
        """
        y_true = self._get_y_true()
        predictions = self._get_y_pred()
        y_pred = self._get_y_pred()
        mean_iou = self._calculate_mean_iou(y_true, y_pred)
        mean_dice = self._calculate_mean_dice_coefficient(y_true, y_pred)
        print(f"Ortalama IoU: {mean_iou:.4f}")
        print(f"Ortalama Dice Katsayısı: {mean_dice:.4f}")
        self.visualize_predictions()