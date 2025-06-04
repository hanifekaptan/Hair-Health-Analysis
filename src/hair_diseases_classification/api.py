from src.hair_diseases_classification.inference import HairDiseasesInference
import io
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import matplotlib.pyplot as plt



class PredictionOutput(BaseModel):

    hair_disease: str
    probability: float


class HairDiseasesAPI:

    def __init__(
            self
    ):

        self.inference_engine = HairDiseasesInference()
        self.app = FastAPI(
            title="Hair Diseases Analysis API",
            description=self._describe_model(),
            version="1.0.0"
        )
        self._setup_routes()


    def _describe_model(
            self
    ) -> str:
        """
        Model hakkında açıklayıcı bir metin döndürür.

        Returns:
            description (str): Model açıklaması.
        """

        description = """
        Bu API, derin öğrenme modelleri kullanarak saç hastalıkları sınıflandırması gerçekleştirir.
        Sınıflandırma Modeli Eğitim Doğruluğu: ~0.95
        Sınıflandırma Modeli Doğrulama Doğruluğu: ~0.91

        """
        return description


    def convert_image_to_array(
            self,
            contents: bytes
    ) -> np.ndarray:
        """
        Görüntü içeriğini NumPy dizisine dönüştürür.

        Args:
            contents (bytes): Görüntü içeriği.

        Returns:
            image_array (np.ndarray): Görüntü dizisi.

        Raises:
            HTTPException: Görüntü işleme sırasında bir hata oluşursa.
        """
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_array = np.array(image)
            plt.imshow(image_array)
            return image_array
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Görüntü işlenirken hata oluştu: {e}"
            )


    def _setup_routes(
            self
    ) -> None:
        """
        API rotalarını ayarlar.
        """

        @self.app.post(
            "/predict",
            response_model=PredictionOutput,
            summary="Yüklenen bir görüntüden saç hastalığını tahmin et"
        )
        async def predict_hair_disease(
                file: UploadFile = File(...)
        ) -> PredictionOutput:
            """
            Yüklenen bir görüntüden saç hastalığını tahmin eder.

            Args:
                file (UploadFile): Yüklenen görüntü dosyası.

            Returns:
                PredictionOutput: Tahmin sonucu.

            Raises:
                HTTPException: Dosya türü desteklenmiyorsa veya tahmin sırasında bir hata oluşursa.
            """

            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(
                    status_code=400,
                    detail="Sadece JPEG, PNG ve JPG dosyaları desteklenir."
                )

            contents = await file.read()
            image_array = self.convert_image_to_array(contents)
            print(f"Output of self.inference_engine.inference(image_array): {self.inference_engine.inference(image_array)}")
            hair_disease, probability = self.inference_engine.inference(image_array)

            return PredictionOutput(
                hair_disease=hair_disease,
                probability=round(float(probability), 4)
            )