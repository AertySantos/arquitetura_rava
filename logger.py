import csv
import time
from pathlib import Path
import json
import numpy as np

class Logger:
    def __init__(self, arquivos_dir="logs"):
        self.arquivos_dir = Path(arquivos_dir)
        self.arquivos_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.arquivos_dir / "logE2E.csv"

        if not self.log_path.exists():
            with open(self.log_path, "w", encoding="utf-8", newline="") as f:
               
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    "ID", "TipoEntrada", "T_Envio", "T_Resposta", "TempoSeg",
                    "Entrada", "Saida"
                ])

    def log_interacao(self, user_id, tempo_ini, tipo, entrada, saida):
        
        t_envio = tempo_ini
        t_resposta = time.time()
        tempo = round(t_resposta - t_envio, 2)
        
        with open(self.log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                user_id, tipo, t_envio, t_resposta, tempo,
                entrada, saida
            ])

    

