from src.utils.data_preprocessing import DataPreprocessing
from src.config.config_data_loading import *
from tensorflow import keras
import os
from typing import Generator


class DataLoader:

    def __init__(
            self,
            data_dir: str,
            img_size: tuple = IMG_SIZE,
            batch_size: int = BATCH_SIZE
            ):
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_preprocessing = DataPreprocessing()


    def _dir_exists(
            self,
            path: str
            ) -> bool:
        """
        Veri klasörlerinin ilgili dizinde var olup olmadığını kontrol eder.

        Args:
            path (str): Dizinin yolu

        Returns:
            bool: Dizin varsa True, yoksa False
        """
        return os.path.exists(path) and os.path.isdir(path)


    def _get_subdata_dir(
            self,
            parent_dir: str,
            subdir_name: str
            ) -> str:
        """
        Bir dizine ait alt dizin (train, test veya val) varsa onun yolunu döndürür.
        
        Args:
            parent_dir (str): Üst dizin (verilerin bulunduğu dizin)
            subdir_name (str): Alt dizin ismi (train, test veya val)

        Returns:
            str: Varsa alt dizin yolu
        """
        target_dir = os.path.join(parent_dir, subdir_name)
        if self._dir_exists(target_dir):
            return target_dir
        else:
            raise Exception(f"{target_dir} dizini bulunamadı.")


class ClassificationDataLoader(DataLoader):

    def __init__(
            self,
            data_dir: str
            ):
        
        super().__init__(data_dir = data_dir)
        self.class_mode = "categorical"
        self.train_data_dir = self._get_subdata_dir(self.data_dir, "train")
        self.test_data_dir = self._get_subdata_dir(self.data_dir, "test")
        self.val_data_dir = self._get_subdata_dir(self.data_dir, "val")
        self.class_names = self._get_class_names(self.train_data_dir)
        self.preprocessing_function = self.data_preprocessing.mobilenet_preprocess

    
    def _get_class_names(
            self,
            data_dir: str
            ) -> list:
        """
        Veri dizinindeki sınıf isimlerini (alt dizinleri) döndürür.

        Args:
            data_dir (str): Veri dizini

        Returns:
            list: Sınıf isimleri
        """
        return sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
 

    def train_val_gen(
            self
            ) -> list:
        """
        Veri klasörlerinden train gen ve val gen değerlerini döndürür.
        Args:

        Returns:
            list: train gen ve val gen
        """
        if not self.val_data_dir:
            train_gen, val_gen = self._train_val_gen_from_train()
        else:
            train_gen = self._train_gen()
            val_gen = self._val_gen()
        return [train_gen, val_gen]


    def _get_flow_from_data_dir_generator(
            self,
            data_dir: str,
            datagen: keras.preprocessing.image.ImageDataGenerator,
            subset: str | None = None,
            shuffle: bool = False
            ) -> keras.preprocessing.image.DirectoryIterator:
        """
        Veri generatörü oluşturur.
        
        Args:
            data_dir (str): Veri dizini
            datagen (keras.preprocessing.image.ImageDataGenerator): Veri generatörü
            subset (str | None, optional): Veri seti (train, validation, test), varsayılan None

        Returns:
            keras.preprocessing.image.DirectoryIterator: Veri generatörü
        """
        gen = datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            classes=self.class_names,
            subset=subset,
            shuffle=shuffle
        )
        return gen
    

    def _train_val_gen_from_train(
            self
            ) -> list:
        """
        Train ve val genetatörü değişkenlerini train data dizinine göre oluşturur.
        
        Args:

        Returns:
            list: Train gen ve val gen
        """
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function,
            validation_split=0.1
        )
        train_gen = self._get_flow_from_data_dir_generator(self.train_data_dir, train_datagen, subset="training", shuffle = True)
        val_gen = self._get_flow_from_data_dir_generator(self.train_data_dir, train_datagen, subset="validation")
        return [train_gen, val_gen]
        
    
    def _train_gen(
            self
            ) -> keras.preprocessing.image.DirectoryIterator:
        """
        Train genetatörü değişkenini train data dizinine göre oluşturur.
        
        Args:

        Returns:
            keras.preprocessing.image.DirectoryIterator: Train gen
        """
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function)
        return self._get_flow_from_data_dir_generator(self.train_data_dir, train_datagen, shuffle = True)
    

    def _val_gen(
            self
            ) -> keras.preprocessing.image.DirectoryIterator:
        """
        Val generatörü değişkenini val data dizinine göre oluşturur.
        
        Args:

        Returns:
            keras.preprocessing.image.DirectoryIterator: Val gen
        """
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function)
        return self._get_flow_from_data_dir_generator(self.val_data_dir, val_datagen)
    

    def test_gen(
            self
            ) -> keras.preprocessing.image.DirectoryIterator:
        """
        Test genetatörü değişkenini test data dizinine göre oluşturur.
        
        Args:

        Returns:
           keras.preprocessing.image.DirectoryIterator: Test gen
        """
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function)
        return self._get_flow_from_data_dir_generator(self.test_data_dir, test_datagen)


class SegmentationDataLoader(DataLoader):

    def __init__(
            self,
            data_dir: str
            ):
        
        super().__init__(data_dir=data_dir)
        self.class_mode = None
        self.batch_size = 1
        self.train_data_dir = self._get_subdata_dir(self.data_dir, "train")
        self.val_data_dir = self._get_subdata_dir(self.data_dir, "val")
        self.test_data_dir = self._get_subdata_dir(self.data_dir, "test")
        self.preprocessing_function = self.data_preprocessing.segmentation_preprocess


    def train_val_gen(
            self
            ) -> list:
        """
        Train ve val gen değişkenlerini oluşturur.
        
        Args:

        Returns:
            list: Train gen ve val gen
        """
        if not self.val_data_dir:
            train_gen, val_gen = self._train_val_gen_from_train()
        else:
            train_gen = self._train_gen()
            val_gen = self._val_gen()
        return [train_gen, val_gen]
    
    
    def _get_image_mask_generators(
            self,
            data_dir: str,
            datagen: keras.preprocessing.image.ImageDataGenerator,
            subset = None,
            shuffle = False
            ) -> list:
        """
        Resim ve maske üreteçlerini oluştur.
        
        Args:
            data_dir (str): Veri dizini
            datagen (keras.preprocessing.image.ImageDataGenerator): Veri generatörü
            subset (str | None, optional): Veri seti (train, validation, test), varsayılan None
            seed (int, optional): Seed değişkeni, varsayılan 42

        Returns:
            list: Image ve mask generatörü
        """
        image_dir = os.path.join(data_dir, "image")
        mask_dir = os.path.join(data_dir, "mask")
        
        if not self._dir_exists(image_dir) or not self._dir_exists(mask_dir):
            raise ValueError(f"Image or mask data_dir not found in {data_dir}")
        else:
            image_gen = datagen.flow_from_data_dir(
                image_dir,
                classes=["image"],
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode=self.class_mode,
                subset=subset,
                shuffle=shuffle
            )
            mask_gen = datagen.flow_from_data_dir(
                mask_dir,
                classes=["mask"],
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode=self.class_mode,
                subset=subset,
                shuffle=shuffle
            )
            return [image_gen, mask_gen]
    

    def _pair_generators(
            self,
            image_gen,
            mask_gen
            ) -> Generator:
        """
        Resim ve maske üreteçlerini eşleştir.
        
        Args:
            image_gen (keras.preprocessing.image.data_dirIterator): Resim gen
            mask_gen (keras.preprocessing.image.data_dirIterator): Maske gen

        Returns:
            Generator: Eşlenen generatör
        """
        while True:
            img = image_gen.next()
            mask = mask_gen.next()
            yield img, mask


    def _train_val_gen_from_train(
            self
            ) -> list[Generator]:
        """
        Train ve val gen değişkenlerini train data dizinine göre oluşturur.
        
        Args:

        Returns:
            Generator: Train ve val gen
        """
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function,
            validation_split=0.1
        )
        train_image_gen, train_mask_gen = self._get_image_mask_generators(self.train_data_dir, train_datagen, subset="training")
        val_image_gen, val_mask_gen = self._get_image_mask_generators(self.train_data_dir, train_datagen, subset="validation")
        train_gen = self._pair_generators(train_image_gen, train_mask_gen)
        val_gen = self._pair_generators(val_image_gen, val_mask_gen)
        return [train_gen, val_gen]
    

    def _train_gen(
            self
            ) -> Generator:
        """
        Train gen değişkenini train data dizinine göre oluşturur.
        
        Args:

        Returns:
            Generator: Train gen
        """
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function)
        train_image_gen, train_mask_gen = self._get_image_mask_generators(self.train_data_dir, train_datagen)
        return self._pair_generators(train_image_gen, train_mask_gen)
    

    def _val_gen(
            self
            ) -> Generator:
        """
        Val gen değişkenini val data dizinine göre oluşturur.
        
        Args:

        Returns:
            Generator: Val gen
        """
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function)
        val_image_gen, val_mask_gen = self._get_image_mask_generators(self.val_data_dir, val_datagen)
        return self._pair_generators(val_image_gen, val_mask_gen)
    

    def test_gen(
            self
            ) -> Generator:
        """
        Test gen değişkenini test data dizinine göre oluşturur.
        
        Args:

        Returns:
            Generator: Test gen
        """
        test_data_dir = self._get_subdata_dir(self.data_dir, "test")
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function)
        test_image_gen, test_mask_gen = self._get_image_mask_generators(test_data_dir, test_datagen)
        return self._pair_generators(test_image_gen, test_mask_gen)
    