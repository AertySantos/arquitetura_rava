#!/bin/bash
input="metricas_resumo.csv"
output="macros.tex"

#> "$output"

tail -n +2 "$input" | while IFS=";" read -r tipo tempo precisao parcial falhas; do
    case $tipo in
        T1) prefix="TOne" ;;
        T2) prefix="TTwo" ;;
        T3) prefix="TThree" ;;
        *) prefix="$tipo" ;;
    esac

    perl -i -pe "
    s/\\\\newcommand{\\\\${prefix}Tempo}{.*}/\\\\newcommand{\\\\${prefix}Tempo}{${tempo}}/;
    s/\\\\newcommand{\\\\${prefix}Precisao}{.*}/\\\\newcommand{\\\\${prefix}Precisao}{${precisao}}/;
    s/\\\\newcommand{\\\\${prefix}Parcial}{.*}/\\\\newcommand{\\\\${prefix}Parcial}{${parcial}}/;
    s/\\\\newcommand{\\\\${prefix}Falhas}{.*}/\\\\newcommand{\\\\${prefix}Falhas}{${falhas}}/;
    " "$output"

done