import contextlib
from typing import Any

import numpy as np
import requests
from tritonclient import http as httpclient

from app.workers.base import ModelWorker


class TritonWorker(ModelWorker):
    def __init__(self, model_metadata):
        self.triton_url = "0.0.0.0:8009"  # Default Triton server URL
        self.client = httpclient.InferenceServerClient(url=self.triton_url)
        super().__init__(model_metadata)

    def _load_model(self):
        """Load model through Triton management API"""
        load_url = f"http://{self.triton_url}/v2/repository/models/{self.model_metadata.name}/load"
        response = requests.post(load_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to load model: {response.text}")

        # Verify model is ready
        if not self.client.is_model_ready(self.model_metadata.name):
            raise RuntimeError("Model is not ready after loading")

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Make prediction through Triton inference API

        Args:
            input_data: Dictionary containing input tensors for the model
                Each key should be an input name and value should be a numpy array
                Example: {"data_0": np.array(...)}

        Returns:
            Dictionary containing output tensors from the model
                Each key is an output name and value is a numpy array
        """
        # Create input tensors
        inputs = []
        for name, data in input_data.items():
            if not isinstance(data, np.ndarray):
                # Convert list or other array-like data to numpy array
                try:
                    shape = data["shape"]
                    content = data["data"]
                    array = np.array(content, dtype=np.float32).reshape(
                        shape
                    )  # Explicitly set dtype to float32
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Input {name} could not be converted to a numpy array"
                    ) from exc
            else:
                array = data.astype(
                    np.float32
                )  # Ensure existing numpy array is float32

            input_tensor = httpclient.InferInput(name, array.shape, "FP32")
            input_tensor.set_data_from_numpy(array)
            inputs.append(input_tensor)

        # Hardcode output name for DenseNet model
        output_name = "fc6_1"

        # Make inference request
        response = self.client.infer(
            model_name=self.model_metadata.name,
            inputs=inputs,
            outputs=[httpclient.InferRequestedOutput(output_name)],
        )

        # Get predictions and convert numpy arrays to lists for JSON serialization
        predictions = {output_name: response.as_numpy(output_name).tolist()}

        return predictions

    def __del__(self):
        """Cleanup: unload model when worker is destroyed"""
        unload_url = f"http://{self.triton_url}/v2/repository/models/{self.model_metadata.name}/unload"
        with contextlib.suppress(requests.RequestException):
            requests.post(unload_url)
