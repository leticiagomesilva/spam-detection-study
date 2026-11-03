# Detecção de Spam com Deep Learning

Projeto acadêmico de estudo comparativo de abordagens de aprendizagem profunda para classificar mensagens de texto como **spam** ou **legítimas**.

## Objetivo

Investigar como diferentes mecanismos de atenção influenciam o desempenho de uma rede LSTM na detecção de spam, considerando a qualidade das previsões, a estabilidade dos resultados e o tempo de treinamento.

## Abordagens propostas

Os dois cenários utilizarão embeddings e uma rede LSTM, alterando o mecanismo de atenção.

| Componente | Cenário 1 | Cenário 2 |
|---|---|---|
| Representação textual | Embeddings | Embeddings |
| Modelagem sequencial | LSTM | LSTM |
| Mecanismo de atenção | Atenção aditiva | Atenção por produto escalar |

- **Embeddings:** representam as palavras como vetores numéricos.
- **LSTM:** processa a sequência de palavras para identificar padrões relevantes à classificação.
- **Atenção:** atribui pesos às representações da sequência para construir a representação utilizada na classificação.

## Base de dados

A base inicialmente proposta é a [SMS Spam Collection, disponibilizada pela UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), composta por mensagens em inglês classificadas como:

- **Ham:** mensagem legítima.
- **Spam:** mensagem indesejada.

O estudo utilizará os rótulos da base. A classificação como spam não será interpretada, por si só, como identificação de fraude ou golpe.

## Metodologia prevista

1. Explorar os dados, verificar duplicatas e analisar a distribuição das classes.
2. Separar os conjuntos de treinamento, validação e teste, evitando que mensagens duplicadas apareçam em conjuntos diferentes.
3. Ajustar o vocabulário e o pré-processamento exclusivamente com os dados de treinamento.
4. Implementar as duas abordagens com condições experimentais comparáveis.
5. Executar 30 simulações por abordagem, totalizando 60 treinamentos.
6. Utilizar as mesmas divisões de dados e sementes correspondentes em cada par de experimentos.
7. Registrar métricas, configurações e tempo de treinamento.
8. Comparar os resultados por meio de gráficos e análise estatística.

As configurações serão escolhidas com os dados de treinamento e validação. O conjunto de teste será reservado para a avaliação final.

## Métricas de avaliação

A métrica principal será o **F1-score da classe spam**, que combina precisão e recall.

Também serão analisados:

- Precisão e recall da classe spam.
- Acurácia.
- Matriz de confusão.
- Tempo de treinamento.
- Média e desvio padrão dos resultados das 30 simulações.

## Hipóteses estatísticas

- **H₀ — Hipótese nula:** não há diferença no desempenho esperado, medido pelo F1-score da classe spam, entre as abordagens.
- **H₁ — Hipótese alternativa:** há diferença no desempenho esperado entre as abordagens.

O teste estatístico será definido conforme o desenho experimental e suas condições de aplicação. A análise considerará também a magnitude da diferença e suas implicações práticas.

Não rejeitar H₀ não será interpretado como comprovação de equivalência entre as abordagens.

## Entregas

- Código-fonte do caso de uso e dos experimentos.
- Resultados das 30 simulações por abordagem.
- Gráficos comparativos e análise estatística.
- Apresentação do estudo.
- Relatório em formato IEEE de duas colunas.
