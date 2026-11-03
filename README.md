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

Procedimento: Shapiro-Wilk nas diferenças pareadas → **teste t pareado** (se normal) ou **Wilcoxon** (caso contrário), com α = 0,05. Teste de permutação e IC bootstrap servem como verificação de robustez; o tamanho de efeito é dado pelo d de Cohen (d_z). Detalhes em `docs/metodologia.md` (seção 8). A análise considerará também a magnitude da diferença e suas implicações práticas.

Não rejeitar H₀ não será interpretado como comprovação de equivalência entre as abordagens.

## Entregas

- Código-fonte do caso de uso e dos experimentos.
- Resultados das 30 simulações por abordagem.
- Gráficos comparativos e análise estatística.
- Apresentação do estudo.
- Relatório em formato IEEE de duas colunas.

---

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requer Python 3.11.

## Uso

### 1. Verificar os dados

O arquivo `data/raw/SMSSpamCollection` já está incluído no repositório.
Para registar o hash e recriar os metadados:

```bash
python scripts/download_data.py --skip-download
```

### 2. Exploração (notebook)

```bash
jupyter notebook notebooks/exploracao.ipynb
```

### 3. Executar os testes

```bash
pytest tests/ -v
```

### 4. Verificação rápida (1 par, smoke test)

```bash
python scripts/run_experiments.py --mode single
```

### 5. Piloto (3 pares, 5 épocas máximas)

```bash
python scripts/run_experiments.py --mode pilot
```

### 6. Campanha completa (30 pares, 60 treinamentos)

```bash
python scripts/run_experiments.py --mode full
```

> **Atenção:** a campanha completa não é executada automaticamente. Execute o comando acima explicitamente quando estiver pronto para realizar os experimentos.

### 7. Análise estatística (após campanha)

```bash
python scripts/analyze_results.py --run-id <RUN_ID>
```

O `RUN_ID` é exibido durante a execução e está em `results/campaigns/`. A campanha oficial do estudo é `campaign_full_30`.

### 8. Figuras para o relatório e slides

```bash
python scripts/make_report_figures.py --run-id campaign_full_30
```

Gera PDF (vetorial, para o LaTeX) e PNG (300 dpi, para os slides) em `results/figures/<RUN_ID>/report/`. O mapa de atenção precisa dos checkpoints locais (não versionados); sem eles, use `--no-attention`.

## Estrutura do Projeto

```text
spam-detection-study/
├── configs/experiment.yaml      # Configuração centralizada
├── data/raw/SMSSpamCollection   # Dataset bruto (UCI)
├── data/processed/              # Dados processados (gerados)
├── src/
│   ├── data.py                  # Carregamento e divisão dos dados
│   ├── preprocessing.py         # Tokenização e codificação
│   ├── train.py                 # Protocolo de treino pareado
│   ├── evaluate.py              # Métricas e persistência
│   ├── predict.py               # Inferência em novos textos
│   ├── stat_tests.py            # Testes estatísticos pareados
│   └── models/
│       ├── attention.py         # AdditiveAttention, ScaledDotProductAttention
│       └── classifier.py        # SpamClassifier (construtor configurável)
├── scripts/
│   ├── download_data.py         # Download e verificação do dataset
│   ├── run_experiments.py       # Runner de experimentos
│   ├── analyze_results.py       # Análise estatística
│   └── make_report_figures.py   # Figuras do relatório IEEE
├── tests/                       # Testes unitários (pytest)
├── notebooks/exploracao.ipynb   # EDA do dataset
├── docs/metodologia.md          # Decisões metodológicas detalhadas
└── results/                     # Resultados gerados pelos experimentos
```

## Técnicas do Estudo

As três técnicas obrigatórias são **Embeddings treináveis (A)**, **LSTM (B)** e **Atenção (C)**. A aceitação de embeddings como técnica separada da LSTM foi confirmada com a professora da disciplina em 03/11/2026.
