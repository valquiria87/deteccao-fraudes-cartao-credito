# Detecção de Fraudes em Transações de Cartão de Crédito

Projeto de Machine Learning desenvolvido em Python para identificar transações potencialmente fraudulentas em uma base altamente desbalanceada. O trabalho compara modelos de classificação, técnicas de balanceamento de classes, limiares de decisão e métodos de interpretabilidade.

## Objetivo

O objetivo é investigar como diferentes algoritmos identificam fraudes em transações de cartão de crédito e avaliar o desempenho usando métricas adequadas a um problema em que a classe fraudulenta representa uma pequena parcela dos dados.

A acurácia, isoladamente, não é suficiente: um modelo pode acertar quase todas as transações e ainda deixar passar a maioria das fraudes.

## Dataset

A base contém 284.807 transações e 31 colunas, incluindo:

- `Class`: variável-alvo — 0 para transação legítima e 1 para fraude.
- `Time`: tempo decorrido desde a primeira transação da base.
- `Amount`: valor da transação.
- `V1` a `V28`: variáveis numéricas transformadas por PCA para proteger informações sensíveis.

A distribuição original contém 284.315 transações legítimas e 492 fraudulentas. As fraudes representam aproximadamente 0,173% da base.

O dataset é carregado diretamente pelo código, sem necessidade de armazenar o arquivo de dados neste repositório.

## Tecnologias utilizadas

- Python
- Pandas e NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib
- Google Colab

## Etapas do projeto

1. Carregamento e exploração dos dados.
2. Análise da distribuição das classes.
3. Criação da variável `Amount_log` com a transformação `log1p`.
4. Preparação das variáveis preditoras e da variável-alvo.
5. Separação estratificada entre treino e teste.
6. Treinamento de Regressão Logística, Random Forest e XGBoost.
7. Avaliação por precisão, recall, F1-score, Average Precision e ROC-AUC.
8. Teste de limiar de decisão alternativo.
9. Comparação entre o conjunto original, undersampling e oversampling.
10. Análise de importância das variáveis e interpretabilidade com SHAP.

Como escolha experimental, `Time` e `Amount` foram removidas diretamente das variáveis preditoras; `Amount_log` foi utilizada no lugar de `Amount`. Essa decisão pode ser revisitada em experimentos futuros.

## Comparação dos modelos

Resultados obtidos no conjunto de teste, com 85.443 transações e 148 fraudes. Os valores abaixo correspondem ao limiar padrão dos classificadores.

| Modelo | Precisão — fraude | Recall — fraude | F1-score — fraude | Average Precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 6,34% | 87,84% | 11,82% | 0,6978 | 0,9692 |
| Random Forest | 83,82% | 77,03% | 80,28% | 0,7807 | 0,9727 |
| XGBoost | 46,24% | 83,11% | 59,23% | 0,7934 | 0,9758 |

### Interpretação

- **Regressão Logística:** apresentou recall elevado, mas gerou muitos falsos positivos.
- **Random Forest:** obteve a maior precisão e o maior F1-score entre os três modelos no limiar padrão.
- **XGBoost:** apresentou os maiores valores de Average Precision e ROC-AUC nesta comparação.

Os resultados demonstram que a escolha do modelo depende do equilíbrio desejado entre detectar fraudes e evitar alertas incorretos.

## Experimento de limiar de decisão

Foi testado o limiar de 0,30 no Random Forest, em comparação com o limiar padrão de 0,50.

| Métrica | Limiar 0,50 | Limiar 0,30 |
|---|---:|---:|
| Precisão — fraude | 83,82% | 60,61% |
| Recall — fraude | 77,03% | 81,08% |
| F1-score — fraude | 80,28% | 69,36% |
| Fraudes detectadas | 114 | 120 |
| Falsos positivos | 22 | 78 |

O limiar de 0,30 identificou seis fraudes adicionais, mas aumentou os falsos positivos. A escolha do limiar deve considerar os custos de fraudes não detectadas e de transações legítimas sinalizadas incorretamente.

## Experimentos de balanceamento

Também foram comparadas três configurações do Random Forest: conjunto de treino original, Random Undersampling e Random Oversampling.

| Configuração | Precisão — fraude | Recall — fraude | F1-score — fraude | Fraudes detectadas | Falsos positivos |
|---|---:|---:|---:|---:|---:|
| Original | 83,82% | 77,03% | 80,28% | 114 | 22 |
| Undersampling | 7,59% | 87,16% | 13,96% | 129 | 1.571 |
| Oversampling | 73,75% | 79,73% | 76,62% | 118 | 42 |

O undersampling apresentou o maior recall, mas também produziu muitos falsos positivos. O oversampling alcançou um equilíbrio melhor do que o undersampling neste experimento, mas não superou o modelo original em F1-score ou precisão.

O balanceamento das classes não garante melhoria em todas as métricas. Os experimentos foram realizados apenas nos dados de treino, mantendo o conjunto de teste original.

## Interpretabilidade com SHAP

Foram gerados gráficos SHAP para investigar a contribuição das variáveis nas previsões do Random Forest, usando uma amostra de 500 transações do conjunto de teste.

A análise incluiu:

- Gráfico de importância global das variáveis.
- Gráfico summary (beeswarm), mostrando a distribuição das contribuições SHAP.

As variáveis `V1` a `V28` são componentes transformados por PCA. Portanto, não é possível atribuir diretamente a elas significados comerciais específicos sem informações adicionais.

SHAP ajuda a explicar o comportamento do modelo, mas não demonstra causalidade nem garante que uma previsão esteja correta.

## O que aprendi

- Por que a acurácia pode ser enganosa em bases desbalanceadas.
- Como interpretar precisão, recall e F1-score para a classe minoritária.
- Como comparar modelos por diferentes métricas.
- Como o limiar de decisão altera o número de fraudes detectadas e falsos positivos.
- Como undersampling e oversampling podem produzir resultados diferentes.
- Como utilizar SHAP para investigar a interpretabilidade de modelos de Machine Learning.

## Limitações e próximos passos

Este projeto tem finalidade educacional e não representa um sistema pronto para uso em produção.

Possíveis evoluções incluem ajuste de hiperparâmetros com validação cruzada estratificada, análise sistemática dos limiares, avaliação de calibração das probabilidades e desenvolvimento de uma demonstração interativa.

## Como reproduzir

1. Abra o notebook `deteccao-fraudes-cartao-credito.ipynb` no Google Colab.
2. Execute as células na ordem apresentada.
3. O notebook carrega o dataset pela URL utilizada no projeto.
4. Instale eventuais dependências indicadas pelo notebook caso o ambiente não as tenha disponíveis.

**Projeto desenvolvido como prática de Machine Learning no desafio da DIO.**
