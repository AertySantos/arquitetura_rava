import json
import time
import requests
from pathlib import Path
from datetime import datetime
from paddleocr import PaddleOCR
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from logger import Logger   # corrige o nome do módulo
from reasoning_dispatcher import Reasoning_dispatcher
import whisper
import time
import numpy as np


class TelegramBot:
    def __init__(self, bot_token, bot_id="1", version="2.0"):
        self.bot_token = bot_token
        self.bot_id = bot_id
        self.version = version
        self.name = f"RavAerty" if bot_id == "1" else f"Bot{bot_id}"
        self.default_msg = f"Olá estou na versão {version}, sou o {self.name}"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.base_file_url = f"https://api.telegram.org/file/bot{self.bot_token}"
        self.arquivos_dir = Path("arquivos")
        self.next_id_path = self.arquivos_dir / "next_id.txt"
        self.offset = self._carrega_offset()
        self.timestamp = ""
        self.result_json = ""
        self.file_json_result = ""
        self.arquivos_dir.mkdir(exist_ok=True)
        self.rd = Reasoning_dispatcher(epsilon=0.01, lr=0.2)
        self.logg = Logger()
        self.tipo = ""
        self.saida_img = ""
        # passo a mim mesmo para o dispatcher poder me chamar
        self.rd.set_callback(self)

    def _carrega_offset(self):
        self.arquivos_dir.mkdir(exist_ok=True)
        if not self.next_id_path.exists():
            self.next_id_path.write_text("0")
        return int(self.next_id_path.read_text().strip())

    def _salva_offset(self, offset):
        self.next_id_path.write_text(str(offset))

    def ouvir(self):
        while True:
            print("Listening")
            response = requests.get(
                f"{self.base_url}/getUpdates", params={"offset": self.offset})
            updates = response.json()

            self.result_json = updates.get("result", [])

            if not self.result_json:
                time.sleep(0.5)
                continue

            for update in self.result_json:
                self.timestamp = datetime.now().isoformat()

                # se for callback_query (botão clicado)
                if "callback_query" in update:
                    self.handle_callback(update["callback_query"])
                # se for mensagem normal
                elif "message" in update:
                    self.processa_mensagem(update)

                self.offset = update["update_id"] + 1
                self._salva_offset(self.offset)

            time.sleep(0.5)

    def processa_mensagem(self, update):
        message = update.get("message", {})

        self.user_id = str(message.get("chat", {}).get("id"))
        texto = message.get("text")
        photo = message.get("photo")
        voice = message.get("voice")
        document = message.get("document")

        self._setup_user_dirs()

        self.file_json_result = json.dumps(update, ensure_ascii=False)
        file_json_path = self.user_json_dir / f"{update['update_id']}.txt"
        file_json_path.write_text(self.file_json_result, encoding='utf-8')

        if texto:

            self.processar_texto(texto, update)

        elif photo:

            self.processar_imagem(photo, update)
        elif voice:
           # print("ListeningVoz")
            self.processar_audio(voice, update)
        elif document:
            self.processar_documento(document, update)

    def _setup_user_dirs(self):
        base = self.arquivos_dir / self.user_id
        self.user_json_dir = base / "jsons"
        self.user_imgs_dir = base / "imgs"
        self.user_audios_dir = base / "audios"
        self.user_docs_dir = base / "documentos"
        for d in [self.user_json_dir, self.user_imgs_dir, self.user_audios_dir, self.user_docs_dir]:
            d.mkdir(parents=True, exist_ok=True)
    # Logs

    def log_texto(self, entrada, saida, tempo):
        log_path = self.arquivos_dir / "logTexto.csv"
        if not log_path.exists():
            log_path.write_text(
                "Timestamp,BotID,UserID,TextoEntrada,TextoResposta,Json\n")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{self.timestamp},{self.bot_id},{self.user_id},{entrada},{saida},{self.file_json_result}\n")

        if self.tipo == "T2":
            self.logg.log_interacao(self.user_id,tempo,"T2", entrada, self.saida_img)
            self.tipo = ""
            self.saida_img = ""
        else:
            self.logg.log_interacao(self.user_id,tempo,"T3", entrada, saida)#metricas de teste fim-a-fim

    def log_img(self, path_img, saida, tempo):
        log_path = self.arquivos_dir / "logImg.csv"
        if not log_path.exists():
            log_path.write_text(
                "Timestamp,BotID,UserID,ImagemPath,TextoExtraido,Json\n")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{self.timestamp},{self.bot_id},{self.user_id},{path_img},{saida},{self.file_json_result}\n")
        

    def log_audio(self, path_audio, texto, saida, tempo):
        log_path = self.arquivos_dir / "logAudio.csv"
        if not log_path.exists():
            log_path.write_text(
                "Timestamp,BotID,UserID,AudioPath,TextoExtraido,Json\n")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{self.timestamp},{self.bot_id},{self.user_id},{path_audio},{texto},{self.file_json_result}\n")
        self.logg.log_interacao(self.user_id, tempo, "T1", texto, saida)#metricas de teste fim-a-fim

    def send_mensagem(self, chat_id, texto, reply_markup=None):
        """Envia mensagem via API do Telegram, com botões opcionais"""
        payload = {
            "chat_id": chat_id,
            "text": texto
        }
        print(f"main: {texto}")
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        r = requests.post(
        f"{self.base_url}/sendMessage",
        json=payload)
        
        #print(r.status_code, r.text)


    def _safe(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (dict, list)):
            # percorre estruturas aninhadas
            return json.loads(json.dumps(obj, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
        return str(obj)

    def processar_texto(self, texto, update):
        """Processa o texto do usuário, envia resposta e adiciona botões de feedback"""
        t_envio = time.time()
        print(f"Texto recebido: {texto}")
        try:
            modulo, msg, _ = self.rd.process_text(self.user_id, texto)
        except Exception as e:
            msg = f"Erro ao processar texto: {e}"
            modulo = "erro"
        
        # monta botões no formato que a API do Telegram espera
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "👍 Certo", "callback_data": f"feedback|{texto}|{modulo}|1"},
                    {"text": "👎 Errado", "callback_data": f"feedback|{texto}|{modulo}|0"}
                ]
            ]
        }

        # envia a mensagem com botões
        response = requests.post(
            f"{self.base_url}/sendMessage",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "chat_id": self.user_id,
                "text": msg,
                #"reply_markup": reply_markup
            })
        )
        """
        # pega o message_id da mensagem enviada para controlar feedback único
        message_info = response.json().get("result", {})
        message_id = message_info.get("message_id")
        if message_id:
            if not hasattr(self, "feedback_registrado"):
                self.feedback_registrado = {}
            # ainda não recebeu feedback
            self.feedback_registrado[message_id] = False
        """
        self.log_texto(texto, msg, t_envio)
        
    def processar_imagem(self, photo, update):
        t_envio = time.time()
        # Inicializa o OCR apenas uma vez se preferir (melhor desempenho)
        ocr = PaddleOCR(use_textline_orientation=True,
                        lang='pt')  # 'pt' = português

        # Baixa a imagem enviada pelo usuário
        file_id = photo[-1]["file_id"]
        file_info = requests.get(
            f"{self.base_url}/getFile", params={"file_id": file_id}
        ).json()
        file_path = file_info["result"]["file_path"]
        file_url = f"{self.base_file_url}/{file_path}"
        local_path = self.user_imgs_dir / f"{update['update_id']}.jpg"

        with open(local_path, "wb") as f:
            f.write(requests.get(file_url).content)

        # PaddleOCR para extrair texto
        result = ocr.predict(str(local_path))

        if result and len(result[0]) > 0:
            # Extrai apenas o texto reconhecido
            rec_texts = result[0]['rec_texts']
            msg = " ".join(rec_texts)
        else:
            msg = "Não foi possível extrair texto da imagem."
        print(msg)
        # Envia resposta e registra no log
        resp = self.rd.aguarde("imagem")
        self.saida_img = msg
        msg+=(f"\n\n{resp}")
        self.tipo = "T2"
        #self.saida_img = msg
        self.send_mensagem(self.user_id, msg)
        self.log_img(str(local_path), msg, t_envio)

    def processar_documento(self, document, update):
        file_id = document["file_id"]
        file_info = requests.get(
            f"{self.base_url}/getFile", params={"file_id": file_id}).json()
        file_path = file_info["result"]["file_path"]
        file_ext = file_path.split('.')[-1]
        file_url = f"{self.base_file_url}/{file_path}"
        local_path = self.user_docs_dir / f"{update['update_id']}.{file_ext}"
        with open(local_path, "wb") as f:
            f.write(requests.get(file_url).content)

        msg = f"Recebido documento: {local_path.name}"
        self.send_mensagem(self.user_id, msg)

    def processar_audio(self, voice, update):

        t_envio = time.time()

        try:
            file_id = voice["file_id"]
            file_info = requests.get(
                f"{self.base_url}/getFile", params={"file_id": file_id}
            ).json()

            file_path = file_info.get("result", {}).get("file_path")
            if not file_path:
                raise ValueError(
                    "Não foi possível obter o caminho do arquivo do áudio.")

            ext = voice.get("mime_type", "audio/ogg").split("/")[-1]
            file_url = f"{self.base_file_url}/{file_path}"

            audio_path = self.user_audios_dir / f"{update['update_id']}.{ext}"
            response = requests.get(file_url)
            response.raise_for_status()

            with open(audio_path, "wb") as f:
                f.write(response.content)

            # Transcrevendo diretamente o .ogg com Whisper
            try:
                # pode trocar por "small", "medium", etc.
                model = whisper.load_model("base")
                result = model.transcribe(
                    str(audio_path), language="pt", fp16=False
                )
                msgini = result["text"].strip()
                # mando o texto transcrito
                self.send_mensagem(self.user_id, msgini)
                # reasoning dispatch
                modulo, msg, _ = self.rd.process_text(self.user_id, msgini)
            except Exception as e:
                msg = f"Erro ao transcrever com Whisper: {e}"

        except Exception as e:
            msg = f"Erro ao processar áudio: {str(e)}"

        # Salva a transcrição em arquivo
        transcript_path = self.user_audios_dir / f"{update['update_id']}.txt"
        # with open(transcript_path, "w", encoding="utf-8") as w:
        #    w.write(msg)

        #print(audio_path)
        self.log_audio(str(audio_path), msgini, msg, t_envio)
        self.send_mensagem(self.user_id, msg)

    def handle_callback(self, callback_query):
        """Trata clique nos botões de feedback"""
        message_id = callback_query["message"]["message_id"]
        chat_id = callback_query["message"]["chat"]["id"]

        # verifica se já recebeu feedback
        if getattr(self, "feedback_registrado", {}).get(message_id, False):
            # resposta para o usuário sem registrar novamente
            requests.post(
                f"{self.base_url}/answerCallbackQuery",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "callback_query_id": callback_query["id"],
                    "text": "Você já avaliou esta mensagem 👍",
                    "show_alert": False
                })
            )
            return

        # processa feedback normalmente
        data = callback_query["data"].split("|")
        if data[0] == "feedback":
            texto, modulo, recompensa = data[1], data[2], int(data[3])
            self.rd.human_feedback(texto, modulo, recompensa)

            # marca como registrado
            self.feedback_registrado[message_id] = True

            # responde ao Telegram
            requests.post(
                f"{self.base_url}/answerCallbackQuery",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "callback_query_id": callback_query["id"],
                    "text": "Feedback registrado! Obrigado 🙏",
                    "show_alert": False
                })
            )

            # remove os botões após o clique
            requests.post(
                f"{self.base_url}/editMessageReplyMarkup",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "reply_markup": {}
                })
            )


if __name__ == "__main__":
    TOKEN = "seu token telegram"
    bot = TelegramBot(bot_token=TOKEN)
    bot.ouvir()
