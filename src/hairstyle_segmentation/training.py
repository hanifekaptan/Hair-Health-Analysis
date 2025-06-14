from src.config.config_model import INPUT_SHAPE
from src.hairstyle_segmentation.model import HairstyleSegmentation
from src.utils.data_loading import SegmentationDataLoader
from src.utils.model_training import SegmentationTraining
from src.utils.model_evaluation import SegmentationEvaluation
from src.config.config_saved_model_dir import HAIRSTYLE_SEG_MODEL_DIR
from src.config.config_data_path import HAIRSTYLE_SEG_DATA_PATH
import tensorflow as tf
import os


class Training:

    def __init__(
            self
            ):
        
        self.seg_model_dir = HAIRSTYLE_SEG_MODEL_DIR
        self.seg_model_path = os.path.join(self.seg_model_dir, "best_model.keras")
        self.data_dir = HAIRSTYLE_SEG_DATA_PATH
        self.learning_rate = 0.001
        self.epochs = 15
        
        if not os.path.exists(self.seg_model_dir):
            os.makedirs(self.seg_model_dir)

        self.data_loader = SegmentationDataLoader(self.data_dir)
        self.train_dataset, self.val_dataset = self.data_loader.train_val_gen()
        self.test_dataset = self.data_loader.test_gen()
        
        self.num_train_samples = self.data_loader.num_train_samples
        self.num_val_samples = self.data_loader.num_val_samples
        self.num_test_samples = self.data_loader.num_test_samples

        self.hair_segmentation = HairstyleSegmentation()
        keras_model = self.hair_segmentation.mobilenet_model(input_shape=(INPUT_SHAPE))

        self.train_temp = SegmentationTraining(self.data_dir, keras_model)


    def train_val_generator(
            self
            ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        tf.data.Dataset nesnelerini doğrudan döndürür.
        """
        return self.train_dataset, self.val_dataset
    

    def test_generator(
            self
            ) -> tf.data.Dataset:
        """
        tf.data.Dataset nesnesini doğrudan döndürür.
        """
        return self.test_dataset
    

    def train(
            self
            ):
        """
        Saç stili segmentasyon modelini eğitir.

        Returns:
            keras.Model: Eğitilen Keras modeli.
        """
        model = self.train_temp.train(
            train_gen=self.train_dataset,
            val_gen=self.val_dataset,
            best_model_dir=self.seg_model_dir,
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
        self.eval = SegmentationEvaluation(self.test_dataset, self.seg_model_path)
        self.eval.evaluation()
        