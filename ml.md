
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

#### Correlation and Causality
In most statistical studies applied to the social sciences, one is interested in the causal effect that one variable (for example, minimum wage) has on another (for example, unemployment). This kind of relationship is extremely difficult to discover. Often, what is found is an association (correlation) between two variables. Unfortunately, without establishing a causal relationship, correlation alone does not provide a solid basis for decision making (for example, raising or lowering the minimum wage to decrease unemployment).

_Na maioria dos estudos estatísticos aplicados às ciências sociais se está interessado no efeito causal que uma variável (por exemplo, salário mínimo) tem em outra (por exemplo, desemprego). Esse tipo de relação é extremamente difícil de descobrir. Muitas vezes, o que se consegue encontrar é uma associação (correlação) entre duas variáveis. Infelizmente, sem o estabelecimento de uma relação causal, apenas correlação não nos fornece uma base solida para tomada de decisão (por exemplo, subir ou baixar o salário mínimo para diminuir o desemprego)._

#### Ceteris Paribus
A very relevant concept for causal analysis is that of ceteris paribus, which `means all other (relevant) factors kept constant`. Most econometric issues are of a ceteris paribus nature. For example, when you want to know the effect of education on wages, we want to keep other relevant variables unchanged, such as family income. The problem is that it is rarely possible to literally keep “everything else constant”. The big question in empirical social studies is always then whether there are enough relevant factors being controlled (kept constant) to make causal inference possible.

_Um conceito bastante relevante para a análise causal é o de ceteris paribus, que `significa todos outros fatores (relevantes) mantidos constantes`. A maioria das questões econometricas são de natureza ceteris paribus. Por exemplo, quando se deseja saber o efeito da educação no salário, queremos manter inalteradas outras variáveis relevantes, como por exemplo a renda familiar. O problema é que raramente é possível manter literalmente “tudo mais constante”. A grande questão em estudos sociais empíricos sempre é então se há suficientes fatores relevantes sendo controlados (mantidos constantes) para possibilitar a inferência causal. _

---
