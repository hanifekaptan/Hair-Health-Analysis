from src.config.config_data_path import HAIR_DISEASES_CLASS_DATA_PATH
from src.config.config_saved_model_path import HAIR_DISEASES_CLASS_MODEL_PATH
from src.hair_diseases_classification.model import HairDiseasesClassificationModel
from src.utils.data_loading import ClassificationDataLoader
from src.utils.model_training import ClassificationTraining
from src.utils.model_evaluation import ClassificationEvaluation
from typing import Generator
import os


class Training:

    def __init__(
            self
            ):
        
        self.class_model_path = HAIR_DISEASES_CLASS_MODEL_PATH
        self.data_dir = HAIR_DISEASES_CLASS_DATA_PATH
        self.learning_rate = 0.001
        self.epochs = 15
        
        if not os.path.exists(self.class_model_path):
            os.makedirs(self.class_model_path)

        self.data_loader = ClassificationDataLoader(self.data_dir)
        self.train_gen, self.val_gen = self.train_val_generator()
        self.test_gen = self.test_generator()

        num_classes = len(self.data_loader.class_names)
        model_builder = HairDiseasesClassificationModel()
        keras_model = model_builder.mobilenet_model(num_classes=num_classes)
        self.train_temp = ClassificationTraining(self.data_dir, keras_model)


    def train_val_generator(
            self
            ) -> list[Generator]:
        """
        Train ve val gen değişkenlerini oluşturur.
        
        Args:

        Returns:
            list: Train gen ve val gen
        """
        train_gen, val_gen = self.data_loader.train_val_gen()
        return [train_gen, val_gen]
    

    def test_generator(
            self
            ) -> Generator:
        """
        Test gen değişkenini oluşturur.
        
        Args:

        Returns:
            Generator: Test gen
        """
        test_gen = self.data_loader.test_gen()
        return test_gen
    

    def train(
            self
            ):
        """
        Modeli eğitir.
        
        Args:

        Returns:
            keras.Model: Eğitilen model
        """
        model = self.train_temp.train(
            train_gen=self.train_gen,
            val_gen=self.val_gen,
            best_model_dir=self.class_model_path,
            learning_rate=self.learning_rate,
            epochs=self.epochs
            )
        self.model = model
    

    def evaluate(
            self
            ) -> None:
        """
        Modeli test verileriyle değerlendirir.
        
        Args:
            
        Returns:
            None
        """
        self.eval = ClassificationEvaluation(self.test_gen, self.class_model_path)
        self.eval.evaluation()