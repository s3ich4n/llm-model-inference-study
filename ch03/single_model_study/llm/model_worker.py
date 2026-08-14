import multiprocessing
from typing import Any

import torch

from llm.model_manager import ModelManager
from logs import logger


class ModelWorker:
    def __init__(self, model_name: str):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        logger.debug(f"Loading model {model_name} on device {self.device}")
        self.model, self.tokenizer = ModelManager().load_model(model_name)
        # Initialize state for streaming
        self.stream_states = {}  # request_id -> (input_ids, attention_mask, past_key_values)

    def generate(
        self,
        prompts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        logger.debug(f"Received prompts: {prompts}")

        # request id와 프롬프트 추출
        prompt_texts = [p.prompt for p in prompts]
        request_ids = [p.id for p in prompts]

        # 배치 내의 프롬프트를 tokenize
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        logger.debug(f"Batch input shape: {inputs.input_ids.shape}")

        # 단일 배치 안의 모든 프롬프트로 텍스트를 생성한다
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,  # Generate up to 50 new tokens
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 모든 output을 디코딩
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.debug(f"Generated texts: {generated_texts}")

        # request id 에 대한 응답값 생성
        results = [
            {
                'request_id': request_id,
                'generated_text': generated_text
            }
            for request_id, generated_text in zip(request_ids, generated_texts)
        ]

        return results

    def generate_forward_batch(
        self,
        prompts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate one token for each prompt in the batch."""
        logger.debug(f"Received streaming prompts: {prompts}")

        # Add padding token to the tokenizer if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize all prompts in batch
        encoded = self.tokenizer(
            [p['prompt'] for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        logger.debug(f"Batch input shape: {encoded.input_ids.shape}")

        # Generate next token
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                use_cache=False
            )

            # Get next token logits and sample
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.multinomial(
                torch.softmax(next_token_logits / 0.7, dim=-1),
                num_samples=1
            ).squeeze(-1)

            # Prepare results
            results = []
            for i, prompt_data in enumerate(prompts):
                token = self.tokenizer.decode(next_token[i].unsqueeze(0), skip_special_tokens=True)
                logger.debug(f"Generated token for prompt '{prompt_data['prompt']}': '{token}'")
                results.append(
                    {
                        'request_id': prompt_data['request_id'],
                        'token': token,
                        'is_finished': token == self.tokenizer.eos_token
                    }
                )

            return results

    @staticmethod
    def run(
        model_name: str,
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue
    ):
        """워커가 개별작업을 수행

        :param model_name:
        :param task_queue:
        :param result_queue:
        :return:
        """
        # Enable remote debugging
        logger.debug("Waiting for debugger to attach...")
        logger.debug("Debugger attached!")

        worker = ModelWorker(model_name)
        logger.debug("Worker initialized")

        while True:
            logger.debug("Waiting for batch from queue...")
            batch_data = task_queue.get()
            logger.debug(f"Received batch: {batch_data}")

            if batch_data is None:  # Shutdown signal
                logger.debug("Received shutdown signal")
                break

            batch, is_streaming = batch_data

            if is_streaming:
                # Handle streaming generation
                result_queue.put(('stream', worker.generate_forward_batch(batch)))
            else:
                # Handle regular generation
                result_queue.put(('complete', worker.generate(batch)))
