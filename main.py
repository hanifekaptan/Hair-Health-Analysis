from src.utils.model_evaluation import (
    ClassificationEvaluation,
    SegmentationEvaluation
)
from src.utils.data_loading import (
    ClassificationDataLoader,
    SegmentationDataLoader
)
from src.config.config_saved_model_dir import *
from src.config.config_data_path import *
import uvicorn
import os

from src.hair_diseases_classification.api import HairDiseasesAPI
from src.hair_diseases_classification.training import Training as HairDiseasesClassTraining

from src.hairstyle_classification.api import HairstyleClassificationAPI
from src.hairstyle_classification.training import Training as HairstyleClassTraining

from src.hairstyle_segmentation.api import HairstyleSegmentationAPI
from src.hairstyle_segmentation.training import Training as HairstyleSegTraining

class HairDiseasesClassificationApp:

    def __init__(self):
        self.data_path = HAIR_DISEASES_CLASS_DATA_PATH
        self.model_dir = HAIR_DISEASES_CLASS_MODEL_DIR
        self.data_loader = ClassificationDataLoader(self.data_path)
        self.training = HairDiseasesClassTraining()
        self.test_gen = self.data_loader.test_gen()
        

    def train(self):
        print("Starting model training...")
        self.training.train()
        print("Model training completed.")


    def evaluate(self):
        self.eval = ClassificationEvaluation(self.test_gen, self.model_dir)
        print("Starting model evaluation...")
        self.eval.evaluation()
        print("Model evaluation completed.")


    def api(self):
        self.api_instance = HairDiseasesAPI()
        app_to_run = self.api_instance.app
        print("Starting model api...")
        uvicorn.run(app_to_run, host="127.0.0.1", port=8000)
        

class HairstyleClassificationApp:
    
    def __init__(self):
        self.data_dir = HAIRSTYLE_CLASS_DATA_PATH
        self.model_path = HAIRSTYLE_CLASS_MODEL_DIR
        self.data_loader = ClassificationDataLoader(self.data_dir)
        self.training = HairstyleClassTraining()
        self.test_gen = self.data_loader.test_gen()
        

    def train(self):
        print("Starting model training...")
        self.training.train()
        print("Model training completed.")


    def evaluate(self):
        self.eval = ClassificationEvaluation(self.test_gen, self.model_path)
        print("Starting model evaluation...")
        self.eval.evaluation()
        print("Model evaluation completed.")


    def api(self):
        self.api_instance = HairstyleClassificationAPI()
        app_to_run = self.api_instance.app
        print("Starting model api...")
        uvicorn.run(app_to_run, host="127.0.0.1", port=8000)
        

class HairStyleSegmentationApp:
    
    def __init__(self):
        self.data_dir = HAIRSTYLE_SEG_DATA_PATH
        self.model_path = os.path.join(HAIRSTYLE_SEG_MODEL_DIR, "best_model.keras")
        self.data_loader = SegmentationDataLoader(self.data_dir)
        self.training = HairstyleSegTraining()
        self.test_gen = self.training.test_dataset 
        


    def train(self):
        print("Starting model training...")
        self.training.train()
        print("Model training completed.")


    def evaluate(self):
        self.eval = SegmentationEvaluation(self.test_gen, self.model_path)
        print("Starting model evaluation...")
        self.eval.evaluation()
        print("Model evaluation completed.")


    def api(self):
        self.api_instance = HairstyleSegmentationAPI()
        app_to_run = self.api_instance.app
        print("Starting model api...")
        uvicorn.run(app_to_run, host="127.0.0.1", port=8000)
 


if __name__ == "__main__":
    print("Starting model training and evaluation...")
    
    hair_diseases_classification = HairDiseasesClassificationApp()
    hairstyle_classification = HairstyleClassificationApp()
    hairstyle_segmentation = HairStyleSegmentationApp()

    # print("Modelleri eğitmek istiyorsanız, 'train' fonksiyonunu kullanabilirsiniz.")
    # hairstyle_segmentation.train()
    # hairstyle_classification.train()
    # hair_diseases_classification.train()
    # print("Model eğitimi tamamlandı. Kaydedilen model isimlerini 'best_model.keras' olarak değiştiriniz.")

    # print("Eğitilen modelleri test etmek istiyorsanız, 'evaluate' fonksiyonunu kullanabilirsiniz.")
    # hairstyle_segmentation.evaluate()
    # hairstyle_classification.evaluate()
    # hair_diseases_classification.evaluate()

    # print("Eğitilen modelleri api olarak kullanmak istiyorsanız, 'api' fonksiyonunu kullanabilirsiniz.")
    # hairstyle_segmentation.api()
    # hairstyle_classification.api()
    # hair_diseases_classification.api()
    