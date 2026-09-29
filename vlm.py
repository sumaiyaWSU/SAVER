import os
import time
import torch

from typing import Dict, Tuple


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ModelHandler:

    _QWEN_MODELS = {
        "qwen2.5-7b":  "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2.5-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    }

    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name.lower().strip()
        if self.model_name not in self._QWEN_MODELS:
            raise ValueError(
                f"Unsupported Qwen model '{model_name}'. "
                f"Choose one of: {list(self._QWEN_MODELS.keys())}"
            )

        self.config = config
        self.hf_token = config["api_keys"]["huggingface"] or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
        self.hf_home = config.get("model_path", {}).get("huggingface") or None

        if self.hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_token
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        self.model_instance: Qwen2_5_VLForConditionalGeneration = None  
        self.processor: AutoProcessor = None  

    def initialize_model(self) -> None:

        model_path = self._QWEN_MODELS[self.model_name]
        print(f"Initializing {self.model_name} from {model_path} ...")

        self.model_instance = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=self.hf_token,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, token=self.hf_token)

        
        device = next(self.model_instance.parameters()).device
        print(f"Loaded Qwen model on device: {device}")

    def get_response(
        self,
        prompt: str,
        image_path: str,
        max_tokens: int = 1000
    ) -> Tuple[str, Dict[str, int], float]:

        if self.model_instance is None or self.processor is None:
            raise RuntimeError("Qwen model not initialized. Call initialize_model() first.")

        start = time.time()
        text, token_counts = self._get_qwen_response(prompt, image_path, max_tokens)
        elapsed = time.time() - start
        return text, token_counts, elapsed

    def _get_qwen_response(
        self,
        prompt: str,
        image_path: str,
        max_tokens: int
    ) -> Tuple[str, Dict[str, int]]:

        token_counts = {"input": 0, "output": 0}

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        model_device = next(self.model_instance.parameters()).device
        original_inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        token_counts["input"] = int(original_inputs.input_ids.shape[1])

        inputs = {k: (v.to(model_device) if hasattr(v, "to") else v)
                  for k, v in original_inputs.items()}

        with torch.no_grad():
            generated_ids = self.model_instance.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=False,
            )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        outputs = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        token_counts["output"] = int(trimmed[0].shape[0]) if trimmed else 0

        text = outputs[0].strip() if outputs else "Model generated an empty response."
        return text, token_counts