from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from app.workers.base import ModelWorker


class TorchVisionWorker(ModelWorker):
    def __init__(self, model_metadata):
        self.transform: transforms.Compose | None = None
        # super().__init__() 안에서 _load_model()이 도니 그 전에 준비해 둔다
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_metadata)

    def _load_model(self):
        if self.model is None:  # Only load if not already loaded
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.model.to(self.device).eval()
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def predict(self, input_data: Any) -> dict[str, Any]:
        if self.model is None or self.transform is None:
            raise RuntimeError("Model or transform not initialized")
        if isinstance(input_data, str):
            image = Image.open(input_data).convert("RGB")
        else:
            image = input_data
        # 모델과 같은 장치로 결과를 전송한다
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        predictions = torch.softmax(outputs, dim=1)
        return {"predictions": predictions.cpu().tolist()}
