# qwen_llm.py
from langchain_core.language_models.llms import LLM
from typing import List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenLLM(LLM):
    # Hiperparâmetros configuráveis
    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4"
    max_new_tokens: int = 512
    temperature: float = 0.1

    # Campos adicionais que o LangChain pode tentar setar
    tokenizer: Any = None
    model: Any = None
    device: str = "cuda:1"

    class Config:
        """Permite atributos extras para evitar erros do Pydantic."""
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, device="cuda:1", **kwargs):
        super().__init__(**kwargs)
        self.device = device

        # Carrega o tokenizer e o modelo
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    @property
    def _llm_type(self) -> str:
        return "custom-qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        # Monta mensagens no formato esperado pelo Qwen
        messages = [
            {"role": "system", "content": "You are Qwen. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            [chat_text], return_tensors="pt"
        ).to(self.model.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )

        # Remove o prompt da resposta
        generated_ids = generated[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return response.strip()
