from src.utils.data_loading import (
    SegmentationDataLoader,
    ClassificationDataLoader)
from src.config.config_model import *
import matplotlib.pyplot as plt


class Training:

    def __init__(
            self,
            data_dir: str
            ):
        
        self.data_dir = data_dir


class ClassificationTraining(Training):

    def __init__(
            self,
            data_dir: str,
            keras_model: keras.Model
            ):
        
        super().__init__(data_dir)
        self.data_loader = ClassificationDataLoader(data_dir)
        self.preprocessing_function = self.data_loader.data_preprocessing.mobilenet_preprocess
        self.model = keras_model
          

    def train(
            self,
            train_gen: keras.preprocessing.image.DirectoryIterator,
            val_gen: keras.preprocessing.image.DirectoryIterator,
            best_model_dir: str,
            learning_rate: float,
            epochs: int
            ) -> keras.Model:
        """
        Modeli eğitir.

        Args:
            train_gen (keras.preprocessing.image.DirectoryIterator): Eğitim verileri
            val_gen (keras.preprocessing.image.DirectoryIterator): Test verileri
            best_model_dir (str): En iyi modelin kaydedileceği dizin
            learning_rate (float): Öğrenme oranı
            epochs (int): Eğitim sayısı

        Returns:
            keras.Model: Eğitilen model
        """
        callbacks = CALLBACKS_PARAMS(best_model_dir)
        compiler_params = COMPILER_PARAMS(learning_rate)
        self.model.compile(**compiler_params)
        self.model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // train_gen.batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_gen.samples // val_gen.batch_size,
            callbacks = callbacks
        )
        return self.model


    def visualize_training(
            self
            ) -> None:
        """
        Modelin eğitim sonucunu görsel olarak gösterir.

        Args:
        
        Returns:
            None
        """
        plt.plot(self.model.history.history['accuracy'])
        plt.plot(self.model.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


class SegmentationTraining(Training):

    def __init__(
            self,
            data_dir: str,
            keras_model: keras.Model
            ):
        
        super().__init__(data_dir)
        self.data_loader = SegmentationDataLoader(data_dir)
        self.model = keras_model

        
    def train(
            self,
            train_gen: keras.preprocessing.image.DirectoryIterator,
            val_gen: keras.preprocessing.image.DirectoryIterator,
            best_model_dir,
            learning_rate,
            epochs
            ) -> keras.Model:
        """
        Modeli eğitir.

        Args:
            train_gen (keras.preprocessing.image.DirectoryIterator): Eğitim verileri
            val_gen (keras.preprocessing.image.DirectoryIterator): Test verileri
            best_model_dir (str): En iyi modelin kaydedileceği dizin
            learning_rate (float): Öğrenme oranı
            epochs (int): Eğitim sayısı

        Returns:
            keras.Model: Eğitilen model
        """
        callbacks = CALLBACKS_PARAMS(best_model_dir)
        compiler_params = COMPILER_PARAMS(learning_rate)
        self.model.compile(**compiler_params)
        self.model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // train_gen.batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_gen.samples // val_gen.batch_size,
            callbacks = callbacks
        )
        return self.model


    def visualize_training(
            self
            ) -> None:
        """
        Modelin eğitim sonucunu görsel olarak gösterir.

        Args:
        
        Returns:
            None
        """
        plt.plot(self.model.history.history['accuracy'])
        plt.plot(self.model.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
