# Makefile para descompactar arquivos, juntar arquivos CSV e modificar a coluna BotTOKEN

# Regra padrão para comprimir todos os arquivos na pasta de origem
all:	descompactar	juntar_csv	limpar

# Nome do arquivo tar.gz de entrada
ARQUIVO_TAR_GZ = arquivos.tar.gz

# Pasta de destino para descompactar os arquivos
PASTA_DESTINO = ../arquivos

# Nome dos arquivos CSV de entrada
CSV1 := $(PASTA_DESTINO)/log.csv
CSV2 := $(PASTA_DESTINO)/logEx.csv

# Nome do arquivo CSV de saída
CSV_SAIDA := $(PASTA_DESTINO)/logNovo.csv

# Regra para descompactar o arquivo tar.gz na pasta de destino
descompactar:
	mkdir -p $(PASTA_DESTINO)
	tar -xzf $(ARQUIVO_TAR_GZ) -C $(PASTA_DESTINO)

# Regra para juntar os arquivos CSV
juntar_csv:
# Removendo cabeçalho do primeiro arquivo CSV
	tail -n +2 $(CSV2) > temp.csv
# Juntando os arquivos CSV...
	cat temp.csv >> $(CSV1)
# Limpar arquivos e pasta de destino
limpar:
	rm -f $(CSV2) temp.csv



