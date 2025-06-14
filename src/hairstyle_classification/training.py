from src.config.config_data_path import HAIRSTYLE_CLASS_DATA_PATH
from src.config.config_saved_model_dir import (
    HAIRSTYLE_SEG_MODEL_DIR,
    HAIRSTYLE_CLASS_MODEL_DIR
)
from src.hairstyle_classification.model import HairstyleClassificationModel
from src.utils.data_loading import ClassificationDataLoader
from src.utils.model_training import ClassificationTraining
from src.utils.model_evaluation import ClassificationEvaluation
from typing import Generator
import os


class Training:

    def __init__(
            self
            ):
        
        self.seg_model_path = HAIRSTYLE_SEG_MODEL_DIR
        self.class_model_dir = HAIRSTYLE_CLASS_MODEL_DIR
        self.class_model_path = os.path.join(self.class_model_dir, "best_model.keras")
        self.data_dir = HAIRSTYLE_CLASS_DATA_PATH
        self.learning_rate = 0.0001
        self.epochs = 50
        
        self.data_loader = ClassificationDataLoader(self.data_dir)
        self.train_gen, self.val_gen = self.train_val_generator()
        self.test_gen = self.test_generator()

        num_classes = len(self.data_loader.class_names)
        model_builder = HairstyleClassificationModel()
        keras_model = model_builder.mobilenet_model(num_classes=num_classes)
        self.train_temp = ClassificationTraining(self.data_dir, keras_model)


    def train_val_generator(
            self
            ) -> list[Generator]:
        """
        Eğitim ve doğrulama veri üreteçlerini oluşturur.

        Returns:
            list[Generator]: [eğitim veri üreteci, doğrulama veri üreteci] listesi.
        """
        train_gen, val_gen = self.data_loader.train_val_gen()
        return [train_gen, val_gen]
    

    def test_generator(
            self
            ) -> Generator:
        """
        Test veri üretecini oluşturur.

        Returns:
            Generator: Test veri üreteci.
        """
        test_gen = self.data_loader.test_gen()
        return test_gen
    

    def train(
            self
            ):
        """
        Saç tipi sınıflandırma modelini eğitir.

        Returns:
            keras.Model: Eğitilen Keras modeli.
        """
        model = self.train_temp.train(
            train_gen=self.train_gen,
            val_gen=self.val_gen,
            best_model_dir=self.class_model_dir,
            learning_rate=self.learning_rate,
            epochs=self.epochs
            )
        self.model = model
    

    def evaluate(
            self
            ) -> None:
        """
        Eğitilmiş modeli test verileri üzerinde değerlendirir.
        """
        self.eval = ClassificationEvaluation(self.test_gen, self.class_model_path)
        self.eval.evaluation()
        