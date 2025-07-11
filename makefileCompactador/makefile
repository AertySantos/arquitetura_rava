# Makefile para comprimir todos os arquivos em uma pasta em um arquivo tar.gz e modificar a coluna BotTOKEN de um arquivo CSV

# Regra padrão para comprimir todos os arquivos na pasta de origem
all: gerarLogEx comprimir limpar

# Pasta de origem contendo os arquivos a serem comprimidos
PASTA_ORIGEM = ../arquivos

# Nome do arquivo de saída comprimido
ARQUIVO_COMPRIMIDO = arquivos.tar.gz

# Nome do arquivo CSV de entrada
CSV_FILE = $(PASTA_ORIGEM)/log.csv

# Nome do arquivo CSV de saída com a coluna BotTOKEN modificada
CSV_EX = $(PASTA_ORIGEM)/logEx.csv

# Regra para modificar a coluna BotTOKEN do arquivo CSV
# Se bottoken for igual a um numero modifica
$(CSV_EX): $(CSV_FILE)
gerarLogEx:
	@awk -F, 'BEGIN {OFS = ","} {if (NR == 1) {print} else {if ($$2 == int($$2)) {$$2 = $$2 "_RavAerty"}; print}}' $(CSV_FILE) > $(CSV_EX)
# Regra para comprimir todos os arquivos na pasta de origem
comprimir:
	tar --exclude='log.csv' -czf $(ARQUIVO_COMPRIMIDO) -C $(PASTA_ORIGEM) .

# Limpar o arquivo CSV modificado
limpar:
	rm -f $(PASTA_ORIGEM)/logEx.csv




