import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    log_loss)
from tensorflow import keras
import seaborn as sns
import numpy as np
import os
import tensorflow as tf


class Evaluation:

    def __init__(
            self,
            model_dir: str
            ):
        self.model_path = os.path.join(model_dir, "best_model.keras")        
        self.keras_model = keras.models.load_model(self.model_path)


class ClassificationEvaluation(Evaluation):

    def __init__(
            self,
            test_data_gen: keras.preprocessing.image.DirectoryIterator,
            model_dir: str
            ):
        super().__init__(model_dir)
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


class SegmentationEvaluation:

    def __init__(
            self,
            test_dataset: tf.data.Dataset,
            model_path: str
            ):
        """
        Segmentasyon modeli değerlendirme sınıfı.

        Args:
            test_dataset (tf.data.Dataset): Test veri seti
            model_path (str): Model dosyasının yolu

        Returns:
            None
        """
        self.test_dataset = test_dataset
        self.model_path = model_path
        self.model = self._load_model()
        self.num_classes = 2


    def _load_model(
            self
            ) -> keras.Model:
        """
        Kaydedilmiş modeli yükler.

        Returns:
            keras.Model: Yüklenen model

        Raises:
            FileNotFoundError: Model dosyası bulunamazsa
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        return keras.models.load_model(self.model_path, compile=False)


    def _calculate_iou(
            self,
            y_true_segment: np.ndarray,
            y_pred_segment: np.ndarray,
            class_id: int
            ) -> float:
        """
        Belirli bir sınıf için IoU (Intersection over Union) skorunu hesaplar.

        Args:
            y_true_segment (np.ndarray): Gerçek segmentasyon maskesi
            y_pred_segment (np.ndarray): Tahmin edilen segmentasyon maskesi
            class_id (int): Hesaplanacak sınıf ID'si

        Returns:
            float: Hesaplanan IoU skoru (0-1 arası)
        """
        y_true_flat = y_true_segment.flatten()
        y_pred_flat = y_pred_segment.flatten()
        y_true_class = (y_true_flat == class_id)
        y_pred_class = (y_pred_flat == class_id)
        intersection = np.logical_and(y_true_class, y_pred_class).sum()
        union = np.logical_or(y_true_class, y_pred_class).sum()
        iou = (intersection + 1e-10) / (union + 1e-10) 
        return iou


    def _calculate_mean_iou(
            self,
            y_true_all: np.ndarray,
            y_pred_all: np.ndarray
            ) -> float:
        """
        Tüm sınıflar için ortalama IoU skorunu hesaplar.

        Args:
            y_true_all (np.ndarray): Tüm gerçek segmentasyon maskeleri
            y_pred_all (np.ndarray): Tüm tahmin edilen segmentasyon maskeleri

        Returns:
            float: Tüm sınıfların ortalama IoU skoru (0-1 arası)
        """
        all_iou_scores = []
        for class_id in range(self.num_classes):
            class_iou = self._calculate_iou(y_true_all, y_pred_all, class_id)
            all_iou_scores.append(class_iou)
            print(f"Class {class_id} IoU: {class_iou:.4f}")
        
        mean_iou = np.mean(all_iou_scores)
        return mean_iou


    def evaluation(
            self
            ) -> None:
        """
        Modeli test veri seti üzerinde değerlendirir ve sonuçları yazdırır.

        Returns:
            None

        Note:
            Şu anda fonksiyon sadece tahminleri toplar ve yazdırır.
            IoU hesaplaması yorum satırı olarak bırakılmıştır.
        """
        y_true_collected = []
        y_pred_collected = []
        for images, masks in self.test_dataset:
            predictions = self.model.predict(images, verbose=0) 
            predicted_masks = tf.argmax(predictions, axis=-1)
            y_true_collected.append(tf.argmax(masks, axis=-1).numpy())
            y_pred_collected.append(predicted_masks.numpy())
        if not y_true_collected or not y_pred_collected:
            print("Uyarı: Test veri kümesinden hiç veri toplanamadı.")
            return
        # y_true_final = np.concatenate(y_true_collected, axis=0)
        # y_pred_final = np.concatenate(y_pred_collected, axis=0)
        # mean_iou = self._calculate_mean_iou(y_true_final, y_pred_final)
        # print(f"Model değerlendirme tamamlandı. Ortalama IoU: {mean_iou:.4f}")