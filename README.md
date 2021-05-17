## Summary
### Machine Learning
- [probabilit_and_statistics](#probabilit_and_statistics)
- [CRISP_DM](#crisp_dm)
- [linear_models](#linear_models)
- [naive_bayes](#naive_bayes)
- [decision_tree](#decision_tree)
- [association_rules](#association_rules)
- [ensemble](#ensemble)
- [reinforcement_learning](#reinforcement_learning)

### Deep Learning
- [Perceptros](#perceptros)
- [Neural Networks](#neural-networks)

---

## Machine Learning Fundamentals 
Basicamente, em computação, o machine Learning é sobre **aprender algumas propriedades** de um conjunto de dados e aplicá-las a novos dados. 

## Modeling a Problem Using Machine Learning
A modelagem de um problema usando machine learning tem o objetivo de desenvolver uma função de hypothesis **h**.

<br/>

<img src="images/output_8_0.png" align="center" height=auto width=80%/>

<br/>

**NOTES**

_For historical reasons, this function h is called a hypothesis._
<br/>
Can also be called **predictive function** if calculates future events.


## Learning Types
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learing

<img src="images/output_1_0.png" />


## Dataset train and dataset tests
O machine learning é sobre aprender algumas propriedades de um conjunto de dados e aplicá-las a novos dados.É por isso que uma prática comum em machine learning para avaliar um algoritmo é dividir os dados em dois conjuntos:

conjunto de train no qual aprendemos propriedades de dados.
conjunto de tests no qual testamos esses dados.

<img src="images/output_8_0.png" align="center" height=auto width=80%/>

<img src="images/output_8_0.png" align="center" height=auto width=80%/>


Aprender os parâmetros de uma função de previsão e testá-la nos mesmos dados é um erro metodológico: um modelo que apenas repetiria os rótulos das amostras que acabaram de ver teriam uma pontuação perfeita, mas não conseguiriam prever nada de útil no entanto.

Essa situação é chamada de overfitting . Para evitá-lo, é prática comum, ao realizar uma experiência de machine learning (supervised), reter parte dos dados disponíveis como um conjunto de tests.

---

### Extra Topic

#### Correlação e Causalidade 
Na maioria dos estudos estatísticos aplicados às ciências sociais se está interessado no efeito causal que uma variável (por exemplo, salário mínimo) tem em outra (por exemplo, desemprego). Esse tipo de relação é extremamente difícil de descobrir. Muitas vezes, o que se consegue encontrar é uma associação (correlação) entre duas variáveis. Infelizmente, sem o estabelecimento de uma relação causal, apenas correlação não nos fornece uma base solida para tomada de decisão (por exemplo, subir ou baixar o salário mínimo para diminuir o desemprego).

#### Ceteris Paribus
Um conceito bastante relevante para a análise causal é o de ceteris paribus, que `significa todos outros fatores (relevantes) mantidos constantes`. A maioria das questões econometricas são de natureza ceteris paribus. Por exemplo, quando se deseja saber o efeito da educação no salário, queremos manter inalteradas outras variáveis relevantes, como por exemplo a renda familiar. O problema é que raramente é possível manter literalmente “tudo mais constante”. A grande questão em estudos sociais empíricos sempre é então se há suficientes fatores relevantes sendo controlados (mantidos constantes) para possibilitar a inferência causal.


---

#### Links
- [Understanding Machine Learning](https://vas3k.com/blog/machine_learning/)
- [Using Machine Learning to generate Super Mario Maker levels](https://medium.com/@ageitgey/machine-learning-is-fun-part-2-a26a10b68df3#.kh7qgvp1b)
- [Machine Learning Performance Improvement Cheat Sheet](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)
- [Statistics for Machine Learning](https://machinelearningmastery.com/category/statistical-methods/)
- [Bias-Variance Trade-Off in Machine Learning](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)
- [Interpretability X Accuracy](https://towardsdatascience.com/the-balance-accuracy-vs-interpretability-1b3861408062)

---

#### References
- [1] https://towardsdatascience.com/important-topics-in-machine-learning-you-need-to-know-21ad02cc6be5
- [2] https://towardsdatascience.com/interpretability-vs-accuracy-the-friction-that-defines-deep-learning-dae16c84db5c
- [3] http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/
- [4] https://stanford.edu/~shervine/l/pt/teaching/cs-229/dicas-aprendizado-supervisionado