import aiml
import os
from suppress import suppress_output  # ou defina inline


class Look_up_reasoning:

    def __init__(self):
        # Cria um kernel do bot
        self.kernel = aiml.Kernel()

    def resposta_rapida(self, msg):
        self.msg = msg
        with suppress_output():
            # Se já existe um arquivo salvo com a "memória" do bot, carrega ele
            if os.path.isfile("aiml/bot_brain.brn"):
                self.kernel.bootstrap(brainFile="aiml/bot_brain.brn")
            else:
                # Carrega os arquivos AIML
                self.kernel.learn("aiml/cumprimentos.aiml")
                self.kernel.saveBrain("aiml/bot_brain.brn")
        with suppress_output():
            response = self.kernel.respond(self.msg)
        # print("Bot:", response)
        return response


if __name__ == "__main__":

    resp_rap = Look_up_reasoning()

    resp_rap.resposta_rapida("bom dia")
