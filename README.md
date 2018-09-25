
# What is machine learning ?
Basicamente, em computação, o aprendizado de máquina é sobre **aprender algumas propriedades** de um conjunto de dados e aplicá-las a novos dados. 

#### Classification
Uma subcategoria de Aprendizado supervisionado, Classificação é o processo de tomar algum tipo de entrada e atribuir um rótulo a ela. Os sistemas de classificação geralmente são usados ​​quando as previsões são de natureza discreta ou “sim ou não”. Exemplo: Mapear uma imagem de alguém para uma classificação masculina ou feminina.

#### Regression
Outra subcategoria de aprendizado supervisionado é usada quando o valor que está sendo previsto difere de um rótulo “sim ou não”, por estar em algum lugar em um espectro contínuo. Sistemas de regressão poderiam ser usados, por exemplo, para responder a perguntas sobre “quanto?” Ou “quantos?”.


### Inductive learning:
É quando a inferência é feita encima de dados conhecidos. Busca padrões e tendências.
1. Aprendizagem supervisionada
2. Aprendizagem não supervisionada
3. Aprendizagem por reforço

<img src="images/output_1_0.png" />

---

# Supervised Learning
Uma aprendizagem deste tipo apresenta problemas do tipo classification e regression. A principal diferença entre classification e regression é que na classification, nós **prevemos um estado**, enquanto, na regression, **prevemos um valor.**

Podemos usar modelos matemáticos para ser as regras de aprendizado:
* ## Linear regression
    * [Papyrus]()
    * [Exercices]()
- ## Perceptrons
    * [Papyrus]()
    * [Exercices]()
* ## Decision tree
    * [Papyrus]()
    * [Exercices]()
* ## Naive bayes
    * [Papyrus]()
    * [Exercices]()
* ## Suport Vectorial Machine
     * [Papyrus]()
     * [Exercices]()
* ## Ensembles methods
     * [Papyrus]()
     * [Exercices]()

## Advantages and Disadvantages of the models


### Logistic Regression
#### Advantages
 - Don’t have to worry about features being correlated
 - You can easily update your model to take in new data (unlike Decision Trees or SVM)
 
#### Disadvantages
 - Deals bad with outliers
 - Must have lots of incomes for each class
 - Presence of multicollinearity
 
 
### Decision Tree
#### Advantages
 - Easy to understand and interpret (for some people)
 - Easy to use - Doesn’t need data normalisation, dummy variables, etc
 - Can handle multi-output models
 - Easily handle feature interactions
 - Don’t have to worry about outliers
 
#### Disadvantages
 - It can be easily overfitted
 - Stability —> small changes in data can lead to completely different trees
 - If a class dominates, it can easily be biased
 - Don’t support online learning –> you should rebuilt the tree when new data comes


### SVM
#### Advantages
 - High accuracy
 - Nice theoretical guarantees regarding overfitting
 - Especially popular in text classification problems
 
#### Disavantages
 - Memory-intensive
 - Hard to interpret
 - Complicated to run and tune
 

### Ensemble Methods
#### Advantages
 - Harder to overfit
 - Usually better perfomance than a single model
 
#### Disadvantages
 - Scaling —> usually it trains several models, which can have a bad performance with larger datasets
 - Hard to implement in real time platform
 - Complexity increases
 - Boosting delivers poor probability estimates (https://arxiv.org/ftp/arxiv/papers/1207/1207.1403.pdf)


---
# Unsupervised Learning
- Clustering
 - Clustering hierarquico
 - Modelos de mistura de gaussianas e validação de cluster
 - Dimensionamento de atributos
 - Análise de componentes principais

---
# Reinforcement learning:
 - A estrutura da aprendizagem por reforço: o problema
 - A estrutura da aprendizagem por reforço: a solução
 - Programação dinâmica
 - Métodos de Monte Carlo
 - Métodos de diferença temporal (TD)
 - Aprendizagem por reforço em espaços contínuos
 - Aprendizado-Q profundo
 - Gradientes de política
 - Métodos ator-críticos




## Modelagem de Machine Learning

<img src="images/output_8_0.png" />

## References:
 - http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/
        
