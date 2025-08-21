# qwen_wrapper.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda:1"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def enviar_msg(self, doc):
        prompt = doc
        contexto = """Extraia do texto apenas o tempo de sentença final estabelecido após todas as fases 
                    da dosimetria, incluindo ajustes como continuidade delitiva. Apresente o valor exato 
                    do tempo de reclusão e o número de dias-multa, sem adicionar explicações ou detalhes 
                    adicionais."""

        messages = [
            {"role": "system", "content": "You are Qwen. You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)

        # print(f"Tokens de entrada: {len(model_inputs['input_ids'][0])}")

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return response

    def leitura(self, input_folder="sentencas", output_path="resposta.txt"):
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    arquivo = file.read()
                    print(f"Arquivo: {filename}")
                    resposta = self.enviar_msg(arquivo)
                    print(resposta)
                    with open(output_path, 'a', encoding='utf-8') as out:
                        out.write(f"{filename}, {resposta}\n")


if __name__ == "__main__":
    qwen = Qwen()

    resp = qwen.enviar_msg("bom dia")
    print(resp)
