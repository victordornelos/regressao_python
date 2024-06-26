# Análise dos fatores que influenciam os custos de planos de saúde 
![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<p align="center">
  <img src="https://img.freepik.com/fotos-premium/uma-visao-elevada-do-estetoscopio-sobre-fundo-azul_23-2148050517.jpg?w=900" alt="Saúde">
</p>

**Autor:** Victor Flávio P. Dornelos\
**E-mail:** victor.dornelos@hotmail.com\
**Linkedin:** [Victor Flávio Pereira Dornelos](https://www.linkedin.com/in/victor-flavio-pereira-dornelos/)

## Sumário
1. [Descrição]()
2. [Objetivo]()
3. [Regressão Linear]()
4. [Metodologia]()
5. [Resultados]()
6. [Referências]()

## 1. Descrição

A **Econometria** é uma ferramenta essencial para economistas, pois emprega métodos matemáticos e estatísticos para analisar dados econômicos. Essa abordagem permite verificar relações entre diversas variáveis, facilitando decisões estratégicas tanto em políticas públicas quanto no setor empresarial.

Os **métodos econométricos** são particularmente úteis para realizar **inferências causais**, o que é crucial para compreender e analisar diferentes objetos de estudo. Com o avanço das ciências de dados, os métodos econométricos tornaram-se mais sofisticados, aumentando sua eficiência para lidar com situações específicas.

Um dos métodos mais utilizados é a **Regressão Linear**. Esta técnica, ao analisar um conjunto de observações, permite investigar relações de dependência causal em relação à variável de interesse. Para ilustrar a aplicação deste modelo, realizaremos uma análise dos custos individuais dos planos de saúde, identificando os fatores que mais os impactam.

## 2. Objetivo

O **objetivo** desta análise é **aplicar** e **descrever** o método de regressão linear, buscando a aderência aos pressupostos desse modelo. Pretendo observar os resultados e identificar quais variáveis exercem maior impacto nos custos individuais dos planos de saúde. 

Esta investigação visa auxiliar na compreensão dos hábitos e fatores que mais contribuem para a necessidade de serviços médicos, funcionando como um alerta para potenciais áreas de intervenção e melhoria na gestão da saúde.

## 3. Regressão Linear
A **Regressão Linear** é um método clássico utilizado para estimar inferências causais entre variáveis. Tipicamente, utiliza-se uma variável dependente Y, uma variável independente X, e outras variáveis de controle para robustecer o modelo e aumentar sua eficácia.

A premissa fundamental da regressão linear é que X possa prever Y de maneira linear. Isso é alcançado através da criação de uma linha que minimiza os resíduos quadráticos, utilizando o método dos **Mínimos Quadrados Ordinários** (OLS).

A fórmula genérica da regressão é:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}Y=\beta_0+\beta_1X+\epsilon">
</p>

Onde:

- **Y** é a variável dependente.
- **β₀** é o coeficiente de interceptação, representando o valor de **Y** quando **X** é zero.
- **β₁** é o coeficiente angular, indicando o impacto de uma variação unitária em **X** sobre **Y**.
- **X** é a variável independente.
- **ϵ** são os resíduos do modelo.

Para que o método OLS seja efetivo, diversos pressupostos devem ser atendidos:
- Amostragem ampla e aleatória dos dados.
- Distribuição normal dos dados.
- Autocorrelação inexistente, isto é, os resíduos não dependem de observações passadas.
- Homocedasticidade, ou seja, variância constante dos resíduos ao longo da série.
- Não correlação entre as variáveis e/ou resíduos.
- Resíduos distribuídos de forma aleatória, sem padrões discerníveis.

Além disso, é essencial validar os resultados através de testes de hipótese. Comumente, utiliza-se o p-valor para determinar a probabilidade de rejeitar a hipótese nula dentro de um nível de significância estatístico pré-estabelecido.

Um desafio frequente é lidar com violações dos pressupostos do OLS. Soluções comuns incluem transformações dos dados, como a aplicação de logaritmos ou raízes quadradas, para corrigir problemas como distribuição não normal e heterocedasticidade, além de linearizar relações. Outra abordagem é o uso de erros padrão robustos, como o método Huber-White, que utiliza os resíduos quadrados para ponderar a variância dos estimadores, ajudando a mitigar problemas como heterocedasticidade.

Este resumo sobre o modelo de regressão linear visa facilitar a análise dos fatores que impactam os custos de planos de saúde. Entretanto, a econometria é um campo complexo que requer um estudo aprofundado. Recomenda-se a consulta de manuais especializados para uma compreensão mais completa do tema.

## 4. Metodologia

Para realizar esta análise, foi utilizado um conjunto de dados disponível no Kaggle, contendo informações sobre indivíduos com variáveis como idade, Índice de Massa Corporal (IMC), status de fumante, sexo, filhos e custos associados. Informações sobre a localidade dos indivíduos também estavam disponíveis, mas foram descartadas para manter a análise em um contexto mais geral.

A ferramenta de análise escolhida foi a linguagem Python, utilizando o ambiente Jupyter Notebook. Inicialmente, as bibliotecas necessárias foram carregadas. Posteriormente, realizou-se uma exploração inicial dos dados, incluindo a criação de gráficos de dispersão para uma visualização preliminar das relações entre as variáveis. Finalmente, procedeu-se à implementação da regressão linear utilizando erros padrão robustos do tipo HC e transformação logarítmica para uma estimativa mais precisa. A análise foi concluída com uma verificação dos resíduos, a fim de validar as suposições do modelo foram respeitadas.

Carregando as bibliotecas necessárias

```python
# Bibliotecas necessárias 
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

Visualização gráfica dos dados (o restante dos gráficos estarão no código)

```python
# Gráfico de dispersão
plt.figure(figsize=(10, 8))

sns.scatterplot(data=df, x='age', y='charges', hue='smoker', palette=['blue', 'red'], alpha=0.6)
sns.regplot(data=df, x='age', y='charges', scatter=False, color='gray')

plt.title('Gráfico de dispersão Custo x Idade comparando com fumante', fontweight='bold')
plt.xlabel('Idade', fontweight='bold')
plt.ylabel('Custos', fontweight='bold')
plt.grid(False)
plt.legend(title='Fumante')
plt.show()
```
Aplicando log e realizando o modelo

```python
# Aplicando log
df['charges'] = np.log(df['charges'])

#Criando o modelo
model = smf.ols(formula='charges ~ age + sex + bmi + smoker + children', data=df).fit()
robust_model = model.get_robustcov_results(cov_type='HC3')
robust_model.summary()
```

Explorando os resíduos

```python
# Gráfico de dispersão
residuals = robust_model.resid

plt.figure(figsize=(10,8))
sns.scatterplot(x=range(len(residuals)),y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Gráfico de dispersão dos resíduos',fontweight='bold')
plt.xlabel('valor dos resíduos',fontweight='bold')
plt.ylabel('Observações',fontweight='bold')
plt.grid(False)

```
```python
# Curva de densidade
plt.figure(figsize=(10, 6))
sns.kdeplot(residuals, color="blue", fill=True)
plt.title('Curva de Densidade dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Densidade')
plt.show()
```

```python
# Teste de homocedasticidade
from statsmodels.stats.diagnostic import het_breuschpagan

X = robust_model.model.exog 
bp_test = het_breuschpagan(residuals, X)

bp_value, bp_pvalue, f_value, f_pvalue = bp_test

print("Valor de Breusch-Pagan:", bp_value)
print("P-valor de Breusch-Pagan:", bp_pvalue)
print("Valor F:", f_value)
print("P-valor F:", f_pvalue)
```

## 5. Resultados

A principal hipótese deste modelo é que os **custos com plano de saúde aumentam à medida que a idade avança**, refletindo a tendência de maior necessidade de assistência médica com o envelhecimento. Ao analisar o gráfico de dispersão, essa hipótese é visualmente corroborada, embora seja evidente uma segmentação por grupos. Esse fenômeno pode ser atribuído à influência de múltiplos fatores sobre a saúde, além da idade.

Dessa forma, uma análise visual detalhada dos gráficos de dispersão entre custos e idade, bem como a comparação com outras variáveis, proporciona uma compreensão mais aprofundada das dinâmicas envolvidas. 

De forma preliminar, observa-se que o **hábito de fumar** impacta significativamente no aumento dos custos. Quanto às **diferenças de sexo**, os gráficos por si só não fornecem insights claros. No entanto, o **Índice de Massa Corporal** (IMC) sugere que indivíduos com valores próximos ou acima do limiar de obesidade enfrentam custos mais elevados. Interessantemente, o **número de filhos** parece influenciar os custos de maneira diferenciada: enquanto em grupos com menores custos o aumento no número de filhos eleva significativamente os gastos, esse mesmo efeito não é observado nos grupos com maiores custos.

Antes de implementar o modelo, é crucial considerar como modelar adequadamente para cumprir com os pressupostos do OLS (Mínimos Quadrados Ordinários). Através da análise detalhada da natureza de cada variável a aparenta ser improvável problemas de **autocorrelação** e **multicolinearidade**. Contudo, a **heterocedasticidade** é uma preocupação válida, dada a alta dispersão dos valores dos custos e a variedade de fatores que podem afetar o modelo. Para mitigar esses problemas, optou-se por utilizar uma **transformação logarítmica** dos custos e o emprego de **erros padrões robustos do tipo HC** (Heteroscedasticity-Consistent). Essas técnicas ajudam a garantir a eficácia e a precisão do modelo em face dessas complicações potenciais.

**Equação da modelo:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}Y=7.0121-0.0750\cdot\text{{sex}}[T.male]+1.5502\cdot\text{{smoker}}[T.yes]+0.0347\cdot\text{{age}}+0.0109\cdot\text{{bmi}}+0.1017\cdot\text{{children}}+\epsilon">
</p>

**onde:**
- **Y** é o log natural de custos com plano de saúde
- **sex** é uma variável dummy (1 se masculino, 0 se não),
- **smoker** é uma variável dummy (1 se fumante, 0 se não),
- **age** é a idade do indivíduo,
- **bmi** é o Índice de Massa Corporal,
- **children** representa o número de filhos.
- **ϵ** são os resíduos do modelo.


**Resultado do modelo:**

| Variável         | Coeficiente | Erro Padrão | t-Stat | P-valor | IC 95% Inferior | IC 95% Superior |
|------------------|-------------|-------------|--------|---------|-----------------|-----------------|
| Intercept        | 7.0121      | 0.068       | 102.706| 0.000   | 6.878           | 7.146           |
| Sexo (Masculino) | -0.0750     | 0.025       | -3.041 | 0.002   | -0.123          | -0.027          |
| Fumante (Sim)    | 1.5502      | 0.033       | 47.558 | 0.000   | 1.486           | 1.614           |
| Idade            | 0.0347      | 0.001       | 33.898 | 0.000   | 0.033           | 0.037           |
| IMC              | 0.0109      | 0.002       | 5.251  | 0.000   | 0.007           | 0.015           |
| Filhos           | 0.1017      | 0.009       | 10.919 | 0.000   | 0.083           | 0.120           |


O método dos mínimos quadrados foi empregado na regressão, resultando em um **coeficiente de determinação** de 76,6%. Esse valor indica que o modelo é capaz de explicar uma significativa proporção da variação nos dados. A relevância das variáveis incluídas é ainda mais enfatizada pela **F-estatística**, que demonstra significância estatística e confirma a importância dos preditores escolhidos.

Os coeficientes estimados oferecem insights profundos. O **uso de cigarros** se destaca como um fator de impacto considerável sobre os custos, mais do que qualquer outra variável analisada. Observa-se que em relação ao **sexo**, os homens tendem a ter custos menores comparativamente. O número de **filhos** também exerce uma influência moderada sobre os custos, embora essa variação seja menos pronunciada. Além disso, o **Índice de Massa Corporal** (IMC) revela um impacto robusto sobre os custos, refletido por um desvio padrão significativo, enquanto a **idade** também demonstra ser um preditor influente.

Os **testes de hipóteses** realizados para os coeficientes confirmam sua significância estatística, levando à rejeição das hipóteses nulas para todas as variáveis consideradas. Esses resultados sublinham a robustez do modelo em capturar as nuances dos fatores que afetam os custos de saúde analisados. Contudo, é essencial continuar a avaliação, observando especialmente os padrões dos resíduos para garantir a adequação do modelo.

Ao realizar a análise dos resíduos por meio de testes de homocedasticidade, gráficos de dispersão e curvas de densidade, observa-se que os resíduos não seguem uma distribuição normal, apresentando uma cauda à direita devido à presença de valores outliers. Além disso, identifica-se heterocedasticidade, com resíduos mais dispersos à medida que se distanciam de zero, tanto positiva quanto negativamente. Esta análise indica variações na variabilidade dos erros, o que pode impactar a eficácia do modelo.


Em conclusão, a regressão linear demonstra ser uma ferramenta extremamente útil para inferência causal, oferecendo insights valiosos sobre as variáveis estudadas. O experimento realizado com o dataset de custos de planos de saúde destacou a eficiência e a relevância desse modelo. Apesar dos desafios envolvidos em cumprir todos os pressupostos dos Mínimos Quadrados Ordinários (OLS), os resultados obtidos reforçam a aplicabilidade e a robustez da regressão linear em análises complexas.

## 6. Referências

HUNTINGTON-KLEIN, Nick. The Effect: An Introduction to Research Design and Causality. Boca Raton, FL: CRC Press, 2022.

CHOI, Miri. Insurance Dataset. Disponível em: <https://www.kaggle.com/datasets/mirichoi0218/insurance>. Acesso em: 25 jun. 2024.

