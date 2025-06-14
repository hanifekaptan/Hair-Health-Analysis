from src.hairstyle_segmentation.inference import HairstyleSegmentationInference
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
from starlette.responses import StreamingResponse


class PredictionOutput(BaseModel):
    segmented_image: list


class PredictionResponse(BaseModel):
    segmented_image_base64: str
    message: str = "Prediction successful"


class HairstyleSegmentationAPI:

    def __init__(
            self
            ):
        self.inference_engine = HairstyleSegmentationInference()
        self.app = FastAPI(
            title="Hairstyle Segmentation API",
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
        Bu API, derin öğrenme modelleri kullanarak saç segmentasyonu gerçekleştirir.
        Segmentasyon Eğitim Modeli Doğruluğu: ~0.98
        Segmentasyon Doğrulama Modeli Doğruluğu: ~0.93
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
            HTTPException: Görüntü işleme sırasında bir hata oluşursa.
        """
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_array = np.array(image)
            return image_array
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Görüntü işlenirken hata oluştu: {e}")


    def _setup_routes(
            self
    ):
        """
        API rotalarını ayarlar.
        """

        @self.app.post(
            "/predict",
            summary="Predicts hair segmentation for an image",
            description="Upload an image to get its hair segmentation mask.",
            responses={
                200: {
                    "description": "Successful Response",
                    "content": {
                        "image/png": {
                            "schema": {"type": "string", "format": "binary"}
                        }
                    }
                },
                422: {"description": "Validation Error"},
                500: {"description": "Internal Server Error"}
            }
        )
        async def predict_hair_type(
            file: UploadFile = File(...)
        ) -> StreamingResponse:
            """
            Yüklenen bir görüntüden saç tipini tahmin eder.

            Args:
                file (UploadFile): Yüklenen görüntü dosyası.

            Returns:
                StreamingResponse: Segmentasyon sonucu.

            Raises:
                HTTPException: Dosya türü desteklenmiyorsa veya tahmin sırasında bir hata oluşursa.
            """
            if file.content_type not in ["image/jpeg", "image/png"]:
                raise HTTPException(
                    status_code=400, detail="Sadece JPEG ve PNG dosyaları desteklenir."
                )
            contents = await file.read()
            image_array = self.convert_image_to_array(contents)
            segmented_image_array = self.inference_engine.inference(image_array)
            
            if segmented_image_array.dtype == bool or np.max(segmented_image_array) <= 1:
                segmented_image_array = (segmented_image_array * 255).astype(np.uint8)
            
            if segmented_image_array.ndim == 2:
                segmented_pil_image = Image.fromarray(segmented_image_array, mode='L')
            elif segmented_image_array.ndim == 3 and segmented_image_array.shape[-1] == 1:
                segmented_pil_image = Image.fromarray(segmented_image_array.squeeze(-1), mode='L')
            else:
                segmented_pil_image = Image.fromarray(segmented_image_array)

            img_byte_arr = io.BytesIO()
            segmented_pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return StreamingResponse(img_byte_arr, media_type="image/png")