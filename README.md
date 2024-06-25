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



