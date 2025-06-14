from src.hairstyle_classification.inference import HairTypeInference
import io
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel


class PredictionOutput(BaseModel):
    hair_class: str
    probability: float


class HairstyleClassificationAPI:

    def __init__(
            self
    ):

        self.inference_engine = HairTypeInference()
        self.app = FastAPI(
            title="Hair Type Analysis API",
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
            str: Model açıklaması.
        """

        description = """
        Bu API, derin öğrenme modelleri kullanarak saç segmentasyonu ve sınıflandırması gerçekleştirir.
        Segmentasyon Modeli Doğruluğu: ~0.96
        Sınıflandırma Modeli Doğruluğu: ~0.67
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
            np.ndarray: Görüntü dizisi.

        Raises:
            HTTPException: Görüntü işleme sırasında bir hata olursa.
        """

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_array = np.array(image)
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
            summary="Yüklenen bir görüntüden saç tipini tahmin et"
        )
        async def predict_hair_type(
                file: UploadFile = File(...)
        ) -> PredictionOutput:
            """
            Yüklenen bir görüntüden saç tipini tahmin eder.

            Args:
                file (UploadFile): Yüklenen görüntü dosyası.

            Returns:
                PredictionOutput: Tahmin sonucu.

            Raises:
                HTTPException: Dosya türü desteklenmiyorsa veya tahmin sırasında bir hata oluşursa.
            """

            if file.content_type not in ["image/jpeg", "image/png"]:
                raise HTTPException(
                    status_code=400,
                    detail="Sadece JPEG ve PNG dosyaları desteklenir."
                )

            contents = await file.read()
            image_array = self.convert_image_to_array(contents)
            hair_class, probability = self.inference_engine.inference(image_array)

            return PredictionOutput(
                hair_class=hair_class,
                probability=round(float(probability), 4)
            )
