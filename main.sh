#!/bin/bash
botToken=""
tipoInteracao=""
file_path=""
indiceMensagem=""
versao="1.5"
defaultMsg="Olá estou na versão $versão, sou o "
nameBot=""
offset=""
bot="1"

function configuracao(){
	#DEFINIÇÃO DOS TOKENS, QUAL BOT UTILIZAR

	echo --------------- >> arquivos/depura.txt

	# Definir o nome da pasta que ficarao os arquivos
	pasta_nome="arquivos"
	source config.sh

	if [ "$1"=="1" ]  ; then
		botTOKEN=$TELEGRAM_TOKEN
		nameBot="RavAerty"
		echo entrou no token 1 >> arquivos/depura.txt
		
	elif [ "$1"=="2" ] ; then
		botTOKEN="00"
		nameBot="Bot2"
		echo entrou no token 2 >> arquivos/depura.txt
	
	else
		botTOKEN="00"
		echo entrou no senao >> arquivos/depura.txt
		nameBot="Bot0"
		ordem="0"
			
	fi
	defaultMsg=$defaultMsg" "$nameBot
	
	script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
	
	# Verificar se a pasta já existe
	if [ ! -d "$script_dir/$pasta_nome" ]; then
    	# Se não existir, crie a pasta
    	mkdir "$script_dir/$pasta_nome"
	fi

	# Adicionar o caminho da pasta "arquivos" ao diretório
	caminho_completo="$script_dir/$pasta_nome"

	if [ ! -f "${caminho_completo}/next_id.txt" ]; then
		touch "${caminho_completo}/next_id.txt"
		offset="0"
		else
		offset=`cat ${caminho_completo}/next_id.txt`
		
		if [ offset==" " ]; then
				offset="0"
		fi
	fi
	
	
	return 0
}

configuracaoUsuario(){
	#DEFINIÇÃO DOs DIRETÓRIO
	#comando para pegar o diretório atual
	
	user_dir=${caminho_completo}/$user_id
	user_json_dir=${caminho_completo}/$user_id/jsons
	user_imgs_dir=${caminho_completo}/$user_id/imgs/
	user_audios_dir=${caminho_completo}/$user_id/audios/
	user_dcs_dir=${caminho_completo}/$user_id/documentos/
	mkdir -p $user_dcs_dir
	mkdir -p $user_audios_dir
	mkdir -p $user_imgs_dir
	user_docs_dir=${caminho_completo}/$user_id/docs
	user_mov_dir=${caminho_completo}/$user_id/movs

}

function defineTipoInteracao(){

	chat_id="$(echo $result | jq -r ".[$indiceMensagem].message.chat.id")"
	chat_id="$(echo ${chat_id} | cut -d " " -f1)"
		
	user_id="$(echo ${result} | jq -r ".[$indiceMensagem].message.chat.id")"
	user_id="$(echo ${user_id} | cut -d " " -f1)"
		
	update_id="$(echo "$result" | jq -r ".[$indiceMensagem].update_id")"
	update_id="$(echo ${update_id} | cut -d " " -f1)"
	
#prepara para colocar o jason no diretorio 
	configuracaoUsuario
	
#salva o jason na pasta do usuario e 

		mkdir -p ${user_dir}
		mkdir -p $user_json_dir	
		echo "$result" >> $user_json_dir/${update_id}.txt

#DEFINE se a primeira mensagem enviada é um texto/documento/imagem/video
	#echo "teste:"
	#echo $result
	if [ "$(echo ${result} | jq -r ".[$indiceMensagem].message.text")" != null ]; then
		tipoInteracao="text"
		processarText
	elif [ "$(echo ${result} | jq -r ".[$indiceMensagem].message.photo")" != null ]; then
		tipoInteracao="photo"
		processarPhoto
	#modificação audio
	elif [ "$(echo ${result} | jq -r ".[$indiceMensagem].message.voice")" != null ]; then
		echo "Arquivo de audio"
		tipoInteracao="voice"
		processarAudio
	elif [ "$(echo ${result} | jq -r ".[$indiceMensagem].message.document")" != null ]; then
		echo "Documento"
		tipoInteracao="document"
		processarDocumento
	fi
	
}


function atualizanextID(){
		offset="$(echo $updates | jq -r ".result[${indiceMensagem}].update_id")"  
		offset=$((offset+1))
		echo $offset > ${caminho_completo}/next_id.txt
}


function sendMensagem(){
msg_status=`curl -s -X POST -H 'Content-Type: application/json' \
			-d '{"chat_id": "'"${chat_id}"'", "text": "'"${msg}"'"}' \
			https://api.telegram.org/bot${botTOKEN}/sendMessage`

}

function ouvir(){

	while true 
	do
		#pega as mensagens que foram enviadas
		updates="$(curl -s "https://api.telegram.org/bot${botTOKEN}/getupdates?offset=${offset}")"
		#echo sudo apt-get install jq

        result="$(echo $updates | jq -r ".result")"
        error="$(echo $updates | jq -r ".description")"

		echo "Listening"

		if [[ "${result}" != "[]" ]]; then
		#pega o array de conversas, cada mensagem em um id
			quantidademensagem="$(echo $updates | jq -r ".result | length")"
			range=$((quantidademensagem-1))
			#echo "entrou aqui"
			for indiceMensagem in $(seq 0 $range); do
				timestamp="$(echo $result | jq -r ".[].message.date")"
				file_jsonResult="$(echo "$result" | sed 's/[][]//g')"
				defineTipoInteracao
				logTexto
				sendMensagem
				if [ $indiceMensagem==$range ]; then 
					atualizanextID
				fi

			done	
		fi
		# Introduzir um atraso de 500 milissegundos (0.5 segundos)
        sleep 0.5
	done
}

processarDocumento(){

	file_id="$(echo ${result} | jq -r ".[].message.document.file_id")"
    file_id="$(echo ${file_id} | cut -d " " -f1)"

    file_json=`curl -s https://api.telegram.org/bot${botTOKEN}/getFile?file_id=${file_id}`
    file_path="$(echo ${file_json} | jq -r ".result.file_path")"

    application="$(echo ${file_path} | cut -d "." -f2)"

	numero=$(shuf -i 1-21 -n 1)

	msg="Esse documento pertence a classe $numero"

    wget -q https://api.telegram.org/file/bot${botTOKEN}/${file_path} -O ${user_dcs_dir}/${update_id}.${application}

}

processarText(){
	#
	textoRecebido="$(echo ${result} | jq -r ".[$indiceMensagem].message.text")"
	
	#resposta a ser enviada
	msg="$(echo ${result} | jq -r ".[$indiceMensagem].message.text") também"	
	#T1 codigo de texto
	codigo="T1"

}

processarPhoto(){
	#processamento de imagem		
	document_confirm="$(echo $result | jq -r ".[].message.document")"
	document_confirm="$(echo ${document_confirm} | cut -d " " -f1)"
	
	timestamp="$(echo $result | jq -r ".[].message.date")"

	photo_confirm="$(echo $result | jq -r ".[].message.photo")"
	photo_confirm="$(echo ${photo_confirm} | cut -d " " -f1)"
	
	
	
	file_id="$(echo ${result} | jq -r ".[].message.photo[-1].file_id")"
	file_id="$(echo ${file_id} | cut -d " " -f1)"

	file_json=`curl -s https://api.telegram.org/bot${botTOKEN}/getFile?file_id=${file_id}`
	file_path="$(echo ${file_json} | jq -r ".result.file_path")"
	
	application="$(echo ${file_path} | cut -d "." -f2)"
	
	wget -q https://api.telegram.org/file/bot${botTOKEN}/${file_path} -O $user_imgs_dir${update_id}.${application}
	
	img_file="$user_imgs_dir/${update_id}.${application}"

	# Utilize o ImageMagick para converter a imagem para tons de cinza
    #convert "$img_file" -colorspace Gray "$img_file"
	#convert "$img_file" -modulate 100,150 "$img_file"
	#convert "$img_file" -noise 5 "$img_file"
	#convert "$img_file" -blur 0x2 "$img_file"
	#convert "$img_file" -brightness-contrast 20x10 "$img_file"
	#convert "$img_file" -sharpen 0x1 "$img_file"
	tesseract $img_file $user_imgs_dir/${update_id} -l por
	
	msg=$(cat "$user_imgs_dir/${update_id}.txt")
		echo "resultado do tesseract: $msg"

	situacao="C0"
	linhas=$(echo "$msg" | wc -l)
	conta_caracter=0

	for ((i = 0; i < ${#msg}; i++)); do
		caractere="${msg:i:1}"
		if [[ "$caractere" =~ [[:alnum:]] ]]; then
			((conta_caracter++))
		fi
	done

	if [ "$linhas" -le 2 ] && ! [[ "$msg" =~ [[:alnum:]] ]]; then 
    	msg="Não foi possivel detectar texto nesta imagem"
		situacao="C101"

	elif [ "$linhas" -ge 3 ] && ! [[ "$msg" =~ [[:alnum:]] ]]; then 
    	msg="Não foi possivel detectar texto nesta imagem"
		situacao="C4"

	elif [ "$conta_caracter" -le 30 ]; then 
    	msg="O texto tem apenas $conta_caracter caracteres"
		situacao="C102"
	fi
	#codigo de imagem
	codigo="T2"
	textoRecebido=$msg
	logImg

}

processarAudio() {

    # Extrair o objeto de áudio da mensagem JSON
    audio="$(echo ${result} | jq -r ".[$indiceMensagem].message.voice")"

    # Verificar se o áudio não é nulo
    if [ "$audio" != "null" ]; then
        # Extrair o file_id do áudio
        file_id="$(echo $audio | jq -r '.file_id')"

        # Variável global para armazenar o caminho do arquivo
		caminho_arquivo=$(curl -s "https://api.telegram.org/bot${botTOKEN}/getFile?file_id=$file_id" | jq -r '.result.file_path')
		application=$(echo "$result" | jq -r '.[].message.voice.mime_type' | cut -d "/" -f2)
		
		# Baixando o arquivo de áudio usando o caminho obtido
		curl -s -o $user_audios_dir${update_id}.${application} "https://api.telegram.org/file/bot${botTOKEN}/${caminho_arquivo}"
		#processamento do audio em texto
		audioTexto
    else
        # Se não houver áudio na mensagem, defina uma mensagem de erro
        msg="Nenhum audio encontrado na mensagem"
    fi
    
}

audioTexto(){
    # Definindo o nome do arquivo de áudio e o idioma
    nome_arquivo="$user_audios_dir/${update_id}.${application}"
    idioma="pt"
	# Convertendo o arquivo de áudio para WAV usando ffmpeg
    nome_arquivo_wav="audio.wav"
    ffmpeg -i "$nome_arquivo" -acodec pcm_s16le -ar 16000 "$nome_arquivo_wav" >/dev/null 2>&1

    # Verifica se o arquivo WAV foi criado com sucesso
    if [ -f "$nome_arquivo_wav" ]; then
        # Inicializa o reconhecedor de fala
        r=$(python3 -c "
import speech_recognition as sr;
r = sr.Recognizer();
audio_data = sr.AudioFile(\"$nome_arquivo_wav\");
with audio_data as source:
    audio_data = r.record(source);
print(r.recognize_google(audio_data, language=\"$idioma\"))")
		# Removendo o arquivo WAV temporário
        rm "$nome_arquivo_wav"
			
        # Tenta reconhecer a fala usando o Google Web Speech API
        if [ -n "$r" ]; then
            echo "Texto transcrito: $r"
			msg=$r
			echo $msg > "${user_audios_dir}/${update_id}.txt"
        else
            echo "Google Speech Recognition não conseguiu entender o áudio."
        fi

    else
        echo "Falha ao criar o arquivo WAV."
    fi
	#T3 codigo de audio
	codigo="T3"
	textoRecebido=$msg
	logAudio
}

logImg(){
	if [ ! -e "arquivos/logImg.csv" ]; then
		# Cabeçalho do CSV 
    	echo "Timestamp,BotTOKEN,ChatID,Path_documento,Situacao,Json" >> arquivos/logImg.csv
	fi		

	echo $timestamp","$bot","$user_id","$user_id"/imgs/"$update_id".jpg"","$situacao","$file_json >>  arquivos/logImg.csv
}

logTexto(){
	if [ ! -e "arquivos/logTexto.csv" ]; then
		# Cabeçalho do CSV 
    	echo "Timestamp,BotTOKEN,ChatID,Path_documento,Tipo,Texto,Json" >> arquivos/logTexto.csv
	fi		

	echo $timestamp","$bot","$user_id","$user_id"/jsons/"$update_id".txt"","$codigo",{"$textoRecebido"},"$file_jsonResult >>  arquivos/logTexto.csv
}

logAudio(){
	if [ ! -e "arquivos/logAudio.csv" ]; then
		# Cabeçalho do CSV 
    	echo "Timestamp,BotTOKEN,ChatID,Path_documento,Situacao,Json" >> arquivos/logAudio.csv
	fi		

	echo $timestamp","$bot","$user_id","$user_id"/audios/"$update_id".jpg"","$situacao","$file_jsonResult >>  arquivos/logAudio.csv
}

#chama a função configuracao passando o primeiro parametro recebido, configura o token, conforme aonfiguracao "$bot"
configuracao $bot
ouvir
