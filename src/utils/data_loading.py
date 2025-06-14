from src.utils.data_preprocessing import DataPreprocessing
from src.config.config_data_loading import *
from tensorflow import keras
import os
import tensorflow as tf
import random


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
            ) -> str | None:
        """
        Bir dizine ait alt dizin (train, test veya val) varsa onun yolunu döndürür.
        
        Args:
            parent_dir (str): Üst dizin (verilerin bulunduğu dizin)
            subdir_name (str): Alt dizin ismi (train, test veya val)

        Returns:
            str | None: Varsa alt dizin yolu
        """
        target_dir = os.path.join(parent_dir, subdir_name)
        if self._dir_exists(target_dir):
            return target_dir
        else:
            print(f"Uyarı: {target_dir} dizini bulunamadı. Bu bölüm atlanıyor.")
            return None


class ClassificationDataLoader(DataLoader):

    def __init__(
            self,
            data_dir: str
            ):
        
        super().__init__(data_dir = data_dir)
        self.class_mode = "categorical"
        self.train_data_dir = self._get_subdata_dir(self.data_dir, "train")
        self.test_data_dir = self._get_subdata_dir(self.data_dir, "test")
        self.val_data_dir = None
        try:
            self.val_data_dir = self._get_subdata_dir(self.data_dir, "val")
        except Exception as e:
            print(f"Uyarı: Doğrulama veri dizini bulunamadı: {e}. Doğrulama olmadan devam ediliyor.")
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


    def _get_flow_from_directory_generator(
            self,
            data_dir: str | None,
            datagen: keras.preprocessing.image.ImageDataGenerator,
            subset: str | None = None,
            shuffle: bool = False
            ) -> keras.preprocessing.image.DirectoryIterator:
        """
        Veri generatörü oluşturur.
        
        Args:
            data_dir (str | None): Veri dizini
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
        train_gen = self._get_flow_from_directory_generator(self.train_data_dir, train_datagen, subset="training", shuffle = True)
        val_gen = self._get_flow_from_directory_generator(self.train_data_dir, train_datagen, subset="validation")
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
        return self._get_flow_from_directory_generator(self.train_data_dir, train_datagen, shuffle = True)
    

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
        return self._get_flow_from_directory_generator(self.val_data_dir, val_datagen)
    

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
        return self._get_flow_from_directory_generator(self.test_data_dir, test_datagen)


class SegmentationDataLoader(DataLoader):

    def __init__(
            self,
            data_dir: str
            ):
        
        super().__init__(data_dir=data_dir)
        self.class_mode = None
        self.batch_size = 1
        self.train_data_dir = self._get_subdata_dir(self.data_dir, "train")
        self.test_data_dir = self._get_subdata_dir(self.data_dir, "test")
        self.val_data_dir = None
        try:
            self.val_data_dir = self._get_subdata_dir(self.data_dir, "val")
        except Exception as e:
            print(f"Uyarı: Doğrulama veri dizini bulunamadı: {e}. Segmentasyon için doğrulama olmadan devam ediliyor.")
        self.preprocessing_function = self.data_preprocessing.segmentation_preprocess
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.num_test_samples = 0


    def train_val_dataset(
            self
            ) -> list:
        """
        Train ve val dataset değişkenlerini oluşturur.
        
        Args:

        Returns:
            list: Train dataset ve val dataset
        """
        if not self.val_data_dir:
            train_dataset, val_dataset = self._train_val_dataset_from_train()
        else:
            train_dataset = self._train_dataset()
            val_dataset = self._val_dataset()
        return [train_dataset, val_dataset]
    
    
    def _get_image_paths_from_dir(self, base_dir: str) -> tuple[list[str], list[str]]:
        """Verilen dizin altındaki tüm görüntü ve maske yollarını toplar ve eşleştirir."""
        image_paths = []
        mask_paths = []
        image_dir = os.path.join(base_dir, "image")
        mask_dir = os.path.join(base_dir, "mask")
        if not self._dir_exists(image_dir):
            print(f"Hata: Görüntü dizini bulunamadı: {image_dir}")
            return [], []
        if not self._dir_exists(mask_dir):
            print(f"Hata: Maske dizini bulunamadı: {mask_dir}")
            return [], [] 
        image_files = {}
        for f in os.listdir(image_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                base_name = f.split('-org')[0] 
                image_files[base_name] = f 
        mask_files = {}
        for f in os.listdir(mask_dir):
            if f.endswith(('.png')):
                base_name = f.split('-gt')[0] 
                mask_files[base_name] = f
        common_ids = sorted(list(set(image_files.keys()).intersection(set(mask_files.keys()))))
        if not common_ids:
            print(f"Hata: '{base_dir}' dizininde eşleşen görüntü/maske çifti bulunamadı. Lütfen dosya adlandırma kurallarınızı kontrol edin.")
            return [], []
        for common_id in common_ids:
            img_filename = image_files[common_id]
            mask_filename = mask_files[common_id]
            image_paths.append(os.path.join(image_dir, img_filename))
            mask_paths.append(os.path.join(mask_dir, mask_filename))
        return image_paths, mask_paths


    def _parse_image_mask(self, image_path: tf.Tensor, mask_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Görüntü ve maske yollarını alır, bunları yükler, çözer, yeniden boyutlandırır ve normalleştirir.
        """
        def _load_image(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, self.img_size)
            return img


        def _load_mask(path):
            mask = tf.io.read_file(path)
            mask = tf.image.decode_png(mask, channels=1, dtype=tf.uint8) 
            mask = tf.image.resize(mask, self.img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            mask = tf.where(tf.greater(mask, 0), 1, 0)
            mask = tf.cast(mask, tf.int32)
            num_classes = 2
            mask = tf.one_hot(mask, depth=num_classes, axis=-1)
            mask = tf.squeeze(mask, axis=-2)
            return mask
        image = _load_image(image_path)
        mask = _load_mask(mask_path)
        return image, mask


    def create_dataset(self, image_paths: list[str], mask_paths: list[str], shuffle: bool) -> tf.data.Dataset:
        """
        Veri kümesi yollarından bir tf.data.Dataset oluşturur.
        """
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
        dataset = dataset.map(self._parse_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


    def train_val_gen(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        train_image_paths_full, train_mask_paths_full = self._get_image_paths_from_dir(self.train_data_dir)
        if self.val_data_dir:
            val_image_paths, val_mask_paths = self._get_image_paths_from_dir(self.val_data_dir)
            self.num_train_samples = len(train_image_paths_full)
            self.num_val_samples = len(val_image_paths)
            train_image_paths = train_image_paths_full
            train_mask_paths = train_mask_paths_full
        else:
            total_samples = len(train_image_paths_full)
            if total_samples == 0:
                raise ValueError(f"Eğitim dizininde ('{self.train_data_dir}') hiç görüntü bulunamadı. Lütfen veri setinizi kontrol edin.")
            val_split = 0.1
            num_val = int(total_samples * val_split)
            num_train = total_samples - num_val
            combined_paths = list(zip(train_image_paths_full, train_mask_paths_full))
            random.seed(42)
            random.shuffle(combined_paths)
            train_combined = combined_paths[num_val:]
            val_combined = combined_paths[:num_val]
            train_image_paths, train_mask_paths = zip(*train_combined) if train_combined else ([],[])
            val_image_paths, val_mask_paths = zip(*val_combined) if val_combined else ([],[])
            train_image_paths = list(train_image_paths)
            train_mask_paths = list(train_mask_paths)
            val_image_paths = list(val_image_paths)
            val_mask_paths = list(val_mask_paths)
            self.num_train_samples = num_train
            self.num_val_samples = num_val
        train_dataset = self.create_dataset(train_image_paths, train_mask_paths, shuffle=True)
        val_dataset = self.create_dataset(val_image_paths, val_mask_paths, shuffle=False)
        return train_dataset, val_dataset


    def test_gen(self) -> tf.data.Dataset:
        test_image_paths, test_mask_paths = self._get_image_paths_from_dir(self.test_data_dir)
        self.num_test_samples = len(test_image_paths)
        test_dataset = self.create_dataset(test_image_paths, test_mask_paths, shuffle=False)
        return test_dataset
    