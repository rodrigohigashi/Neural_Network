# Diagnóstico de Doença Cardíaca com Redes Neurais Artificiais (RNA)

Este projeto utiliza uma base pública da Cleveland Clinic Foundation para diagnosticar doença cardíaca com base em variáveis clínicas. O foco é aplicar redes neurais artificiais (RNA) para prever o risco de doença, além de explorar análise exploratória dos dados e ajuste de hiperparâmetros.

---

## 📁 Dados

A base contém 14 variáveis, incluindo idade, sexo, pressão sanguínea, colesterol, tipo de dor no peito, entre outras. O alvo (`Target`) indica se há ou não presença de doença cardíaca (1 = sim; 0 = não).

Exemplo de metadados:

| Coluna   | Descrição                                        | Tipo de Variável              |
|----------|--------------------------------------------------|-------------------------------|
| Age      | Idade em anos                                    | Numérica                      |
| Sex      | (1 = homem; 0 = mulher)                          | Categórica                    |
| CP       | Tipo de dor no peito (0–4)                       | Categórica                    |
| Trestbpd | Pressão arterial em repouso                      | Numérica                      |
| Chol     | Colesterol sérico em mg/dl                       | Numérica                      |
| FBS      | Açúcar em jejum > 120 mg/dl (1 = sim; 0 = não)   | Categórica                    |
| ...      | ...                                              | ...                           |
| Target   | Doença cardíaca (1 = sim; 0 = não)               | Alvo                          |

---

## 🔍 Etapas do Projeto

- **Análise exploratória dos dados (AED)**:
  - Verificação de valores ausentes e duplicações
  - Conversão de variáveis numéricas para categóricas quando necessário
  - Geração de relatórios automáticos com `pandas-profiling` e `sweetviz`
  - Visualizações como `pairplot`

- **Pré-processamento dos dados**:
  - Separação entre treino e teste
  - Padronização de escala (`StandardScaler`)
  - Codificação de variáveis categóricas (`OneHotEncoder`)

- **Modelagem com RNA (Keras/TensorFlow)**:
  - Definição da arquitetura com 2 camadas ocultas
  - Early stopping monitorando a métrica AUC
  - Treinamento com 50 épocas
  - Avaliação por AUC nos conjuntos de treino e teste

- **Geração de múltiplos modelos com variações de hiperparâmetros**:
  - Geração de 50 combinações de hiperparâmetros
  - Criação de uma tabela com os resultados
  - Análise de desempenho via gráficos de linha e dispersão

- **Teste com novo paciente**:
  - Criação de um `DataFrame` com dados fictícios
  - Pré-processamento com os mesmos steps de treino
  - Previsão de probabilidade com o melhor modelo encontrado

---

## 🔁 Reprodutibilidade

O treinamento de redes neurais envolve componentes estocásticos, como inicialização aleatória dos pesos, embaralhamento dos dados e uso de dropout.

Para garantir reprodutibilidade dos resultados apresentados — incluindo métricas, gráficos e comparações entre execuções — foi utilizada uma seed fixa durante o processo de treinamento dos modelos.

---

## 🧠 Tecnologias utilizadas

- Python 3.x
- Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- TensorFlow (Keras)
- Pandas Profiling, Sweetviz

---

## ✅ Requisitos (requirements.txt)

```txt
matplotlib==3.2.2
numpy==1.19.5
pandas==1.2.5
scikit-learn==0.24.0
scipy==1.7.2
seaborn==0.10.1
tensorflow==2.4.1
pandas-profiling==3.1.0
sweetviz==1.0b6
