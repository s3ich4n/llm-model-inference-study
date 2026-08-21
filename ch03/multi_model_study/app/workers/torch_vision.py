from typing import Optional, Any, Dict

import torch
from PIL import Image
from app.workers.base import ModelWorker
from torchvision import transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class TorchVisionWorker(ModelWorker):
    def __init__(self, model_metadata):
        self.transform: Optional[transforms.Compose] = None
        super().__init__(model_metadata)

    def _load_model(self):
        if self.model is None:  # Only load if not already loaded
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def predict(self, input_data: Any) -> Dict[str, Any]:
        if self.model is None or self.transform is None:
            raise RuntimeError("Model or transform not initialized")
        if isinstance(input_data, str):
            image = Image.open(input_data).convert('RGB')
        else:
            image = input_data
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        predictions = torch.softmax(outputs, dim=1)
        return {"predictions": predictions.tolist()}
