# Metodologia

Este documento descreve as decisões metodológicas do estudo comparativo de mecanismos de atenção em redes LSTM para detecção de spam.

---

## 1. Base de Dados

**Fonte:** SMS Spam Collection v.1 — UCI Machine Learning Repository
**URL:** <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>
**Licença:** CC BY 4.0
**Total de mensagens originais:** 5.574
**Idioma:** Inglês
**Classes:** `ham` (mensagem legítima) e `spam` (mensagem indesejada)

> **Nota:** Spam refere-se a mensagens indesejadas (publicidade, golpes por SMS). A classe não é tratada automaticamente como fraude.

O arquivo bruto é preservado em `data/raw/SMSSpamCollection`. O hash SHA-256 e a URL de origem são registados em `data/raw/metadata.json`.

---

## 2. Preparação dos Dados

### 2.1 Verificações de Integridade

Antes da divisão, são realizadas as seguintes verificações:

1. **Valores ausentes:** qualquer linha com `label` ou `text` nulo é registada e o experimento é interrompido.
2. **Rótulos inválidos:** apenas `ham` e `spam` são aceites; outros valores são reportados e interrompem a execução.

### 2.2 Tratamento de Duplicatas

Aplicado antes da divisão com as seguintes regras (documentadas em `src/data.py`):

| Regra | Condição | Ação |
|-------|----------|------|
| 1 | Mesmo texto **e** mesmo rótulo | Manter a primeira ocorrência; remover as demais |
| 2 | Mesmo texto **e** rótulos diferentes | Remover **todas** as ocorrências e registar o texto |

A regra 2 é aplicada primeiro. As mensagens removidas são registadas no relatório de deduplicação.

### 2.3 Divisão Estratificada

- Proporção: **70 % treino, 15 % validação, 15 % teste**
- Semente da divisão: **42** (fixa; única para todo o estudo)
- Estratificação por classe para preservar a proporção `ham`/`spam` em cada partição
- Os identificadores de cada partição são salvos em `data/processed/split_ids.json` para reprodução exacta

> **Invariante de leakage:** a divisão é realizada uma única vez sobre o conjunto deduplificado. Nenhuma informação de validação ou teste é usada antes da divisão.

---

## 3. Pré-processamento de Texto

### 3.1 Transformações Aplicadas

| Transformação | Aplicada | Justificativa |
|---------------|----------|---------------|
| Lowercase | ✓ | Normalização sem perda de sinal |
| Remoção de números | ✗ | Valores monetários e códigos curtos são sinal de spam |
| Remoção de pontuação | ✗ | Exclamações e reticências são sinal de spam |
| Remoção de URLs | ✗ | URLs de promoções são sinal de spam |
| Remoção de caracteres especiais | ✗ | "£", "@", "$" têm valor discriminativo |
| Stemming / lematização | ✗ | Complexidade desnecessária para este estudo |

### 3.2 Tokenização

- **Tokenizador:** `tf.keras.preprocessing.text.Tokenizer`
- **Vocabulário máximo:** 10.000 tokens
- **Token OOV:** `<OOV>` → índice 1
- **Padding:** índice 0 (reservado; não pertence ao vocabulário)
- **Comprimento máximo:** 100 tokens (truncagem à direita para sequências maiores; padding à direita para menores)

> **Invariante de leakage:** o tokenizador é ajustado **exclusivamente** sobre os textos de treino. Validação e teste são apenas transformados.

---

## 4. Arquitectura dos Modelos

Ambos os cenários partilham exactamente a mesma arquitectura, com excepção da camada de atenção.

```
Input (inteiros, sequência de comprimento máximo 100)
    │
    ▼
Embedding (vocab_size + 1, embedding_dim=64, mask_zero=True)
    │
    ▼
LSTM (units=64, return_sequences=True, return_state=True)
    │            │
    │            └─ h_final: último estado oculto válido (considerando padding)
    │
    ▼
Atenção ──── cenário 1: Aditiva
         └── cenário 2: Produto Escalar Escalado
    │
    ▼
Dropout (rate=0.2)
    │
    ▼
Dense (1, activation='sigmoid')
```

### 4.1 Tratamento do Padding

O embedding com `mask_zero=True` gera uma máscara booleana que é propagada para a LSTM. O Keras utiliza `tf.where` internamente para não actualizar o estado em posições de padding. O estado final `h_final` é portanto o estado após o último token válido.

A mesma máscara é passada explicitamente à camada de atenção, que adiciona `-1e9` aos escores das posições de padding antes do softmax, tornando o peso dessas posições aproximadamente zero.

### 4.2 Diferença de Parâmetros entre Mecanismos

| Componente | Atenção Aditiva | Produto Escalar Escalado |
|------------|----------------|--------------------------|
| W_q | `lstm_units × attention_dim` = 64 × 64 = 4.096 | — |
| W_k | `lstm_units × attention_dim` = 64 × 64 = 4.096 | — |
| v   | `attention_dim` = 64 | — |
| **Total extra** | **8.256** | **0** |

Esta diferença é intrínseca à escolha do mecanismo e é documentada como parte da comparação.

---

## 5. Configuração Experimental

| Parâmetro | Valor |
|-----------|-------|
| Semente da divisão | 42 |
| Sementes de treino | 1001 a 1030 (30 pares) |
| Vocabulário máximo | 10.000 |
| Comprimento máximo | 100 |
| Embedding dim | 64 |
| LSTM units | 64 |
| Attention dim (aditiva) | 64 |
| Dropout | 0,2 |
| Batch size | 32 |
| Máximo de épocas | 20 |
| Optimizador | Adam (lr = 0,001) |
| Função de perda | Entropia cruzada binária |
| Early stopping | Paciência 3, monitor `val_loss`, restaurar melhores pesos |
| Limiar de classificação | 0,5 |

---

## 6. Protocolo de Treinamento Pareado

Para cada semente `k` ∈ {1001, …, 1030}:

1. Construir o modelo de **atenção aditiva** com semente `k`.
2. Construir o modelo de **produto escalar escalado** com uma semente derivada.
3. **Copiar os pesos** das camadas partilhadas (Embedding, LSTM, Dense de saída) do modelo aditivo para o modelo de produto escalar.
4. **Verificar a cópia** (comparação exacta dos tensores com tolerância `atol=1e-7`).
5. **Pré-embaralhar** os dados de treino uma única vez com semente `k`.
6. Treinar ambos os modelos na **mesma sequência de lotes** (sem re-embaralhamento por época).
7. Avaliar ambos no conjunto de **teste** (utilizado apenas para avaliação final, nunca para ajuste).

### 6.1 Reprodutibilidade

- Sementes globais (`PYTHONHASHSEED`, `random`, `numpy`, `tensorflow`) são definidas antes de cada construção de modelo.
- `tf.config.experimental.enable_op_determinism()` é tentado; se não suportado (e.g., alguns operadores em GPU), um aviso é registado e o treinamento continua.
- **Limitação:** certos operadores CUDA podem produzir resultados ligeiramente diferentes entre execuções na GPU mesmo com a flag de determinismo activada.

---

## 7. Avaliação

### 7.1 Métricas

| Métrica | Tipo | Cálculo |
|---------|------|---------|
| F1-score (spam) | **Principal** | `f1_score(y_true, y_pred, pos_label=1)` |
| Precisão (spam) | Complementar | `precision_score(y_true, y_pred, pos_label=1)` |
| Recall (spam) | Complementar | `recall_score(y_true, y_pred, pos_label=1)` |
| Acurácia | Complementar | `accuracy_score` |
| Average precision | Complementar | `average_precision_score(y_true, probabilities)` |
| Matriz de confusão | Complementar | [TN, FP, FN, TP] |
| Tempo de treino | Operacional | Segundos (wall-clock) |

### 7.2 Artefactos Salvos por Execução

- `results/metrics.csv` — uma linha por (cenário, semente)
- `results/predictions/{run_id}/{cenário}_seed{semente}.csv` — probabilidades e rótulos previstos por mensagem
- `results/campaigns/{run_id}/histories/{cenário}_seed{semente}.json` — histórico de treino por época
- `results/campaigns/{run_id}/manifest.json` — configuração, versões, hash dos dados
- `checkpoints/{run_id}/{cenário}_seed{semente}/weights.weights.h5`

---

## 8. Análise Estatística

### 8.1 Hipóteses

- **H₀:** A diferença média esperada de F1 entre atenção aditiva e produto escalar escalado é zero.
- **H₁:** A diferença média esperada de F1 é diferente de zero.

A diferença pareada é definida como: **d_i = F1_aditiva_i − F1_produto_i**

### 8.2 Teste de Permutação Pareado

Teste bilateral por troca de sinais dentro de cada par:
- Para cada permutação: multiplicar cada `d_i` por um sinal aleatório (+1 ou −1)
- Estatística: diferença média permutada
- p-valor: proporção de permutações onde `|média_permutada| ≥ |média_observada|`
- Semente: 42; número de reamostragens: 10.000 (configurável)

### 8.3 Intervalo de Confiança Bootstrap

- Bootstrap dos pares (re-amostragem com reposição)
- IC de 95 % pelo método dos percentis
- 10.000 reamostras

### 8.4 Limitações

1. As 30 repetições partilham uma **única divisão** treino/validação/teste. Medem variabilidade da inicialização e da ordem dos lotes, não amostras independentes da população de mensagens. Os resultados não podem ser generalizados directamente além deste conjunto de dados e desta divisão.
2. **Não rejeitar H₀ não comprova equivalência.** Um teste de equivalência (e.g., TOST) seria necessário para afirmar equivalência.
3. Uma diferença estatisticamente significativa deve ser avaliada conjuntamente com a **magnitude** da diferença para determinar relevância prática.

---

## 9. Pendências Académicas

> **Pendência:** a aceitação de **embeddings treináveis** como técnica separada (distinta da LSTM) ainda precisa de ser confirmada com a professora da disciplina. O estudo está estruturado para ter três técnicas (embeddings, LSTM, atenção), mas essa classificação não foi formalmente validada. Esta documentação não deve ser interpretada como confirmação de que o requisito já foi satisfeito.

---

## 10. Referências

- Almeida, T.A., Hidalgo, J.M.G., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering. *DOCENG '11*.
- Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*.
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
