
#  Naive Bayes

 -  É um tipo de algoritmos de _aprendizado supervisionados_ baseados na aplicação do **teorema de Bayes com a suposição “ingênua”** (naive).
     - "Naive" porque supomos que os eventos são independentes.

## Bayes theorem
- Naive Bayes é um teorema de inferência probabilístico, baseado no conceito de probabilidade condicional. 
- Calcula a probabilidade de um evento ocorrer, com base em outras probabilidades relacionadas ao evento em questão.
-  Reverend Bayes (which he used to try and infer the existence of God no less) 

<img src="images/output_4_0.png" />

**Tenha bem claro o que é o evento e quais as duas probabilidades durante um problema.**

## Advantages:
 
- Altamente escalável.
- Simples.
- **Não é sensível a overffting**
- Retorna um grau de certeza na resposta.
- _Rápido_: o tempo de treinamento e previsão do modelo é muito rápido para a quantidade de dados que pode manipular
- _Exemplos reais_: famosa classificação de documentos e filtragem de spam.
- Uma das principais vantagens que Naive Bayes tem sobre outros algoritmos de classificação é sua capacidade de lidar com um número extremamente grande de recursos. No exemplo de um sistema de spam, cada palavra é tratada como uma característica e existem milhares de palavras diferentes. 

## Example 01)
Veja o exemplo abaixo:
 - Evento: recebimento do email
 - P(A): spam
 - P(B): not spam

<img src="images/output_6_0.png" />

<img src="images/output_7_0.png" />

As probabilidades de P(A) e P(B) são a priori pois ja sabiamos que A e B aconteciam.

As probabilidades de P(R|A) e P(R|B) são a posteriori pois já sabiamos que R acontecia.

<img src="images/output_9_0.png" />

## Example 02)

<img src="images/output_11_0.png" />

### 1 out of 10000 people are sick, so:
 - P(sick) = 0.0001
 - P(healthy) = 0.9999
 
This moment has an event with A and B.
Patient tested positive.
### The examns has accuracy 99% for (sick) or (healthy) :
 - P(+|sick) = 0.99
 - P(+|healthy) = 0.01
 
 
 This moment has an event with (+|A) and (+|B) 

<img src="images/output_13_0.png" />

 #### The goups P(-) does not matter.

<img src="images/output_15_0.png" />

Desvantagens de naive bayes:
   - falsos positivos

## Disavantages
 -  Sua principal desvantagem é que ele não pode aprender interações entre recursos (por exemplo, não pode aprender que, embora você adore filmes com Brad Pitt e Tom Cruise, você odeia filmes em que eles estão juntos).
 

Uma das principais vantagens que Naive Bayes tem sobre outros algoritmos de classificação é sua capacidade de **lidar com um número extremamente grande de recursos**. Por exemplo, um classificador de spam, cada palavra é tratada como uma característica e existem milhares de palavras diferentes.

É melhor entender este teorema usando um exemplo. Digamos que você seja um membro do Serviço Secreto e tenha sido destacado para proteger o candidato democrata à presidência durante um de seus discursos de campanha. Sendo um evento público que está aberto a todos, o seu trabalho não é fácil e você tem que estar sempre atento às ameaças. Então, um lugar para começar é colocar um certo fator de ameaça para cada pessoa. Então, com base nas características de um indivíduo, como a idade, o sexo e outros fatores menores, como a pessoa carregando uma bolsa ?, a pessoa parece nervosa? etc. você pode fazer um julgamento quanto a se essa pessoa é uma ameaça viável.

Se um indivíduo marcar todas as caixas até um nível em que ele cruze um limiar de dúvida em sua mente, você poderá agir e removê-lo da vizinhança. O teorema de Bayes funciona da mesma forma como estamos computando a probabilidade de um evento (uma pessoa ser uma ameaça) com base nas probabilidades de certos eventos relacionados (idade, sexo, presença de saco ou não, nervosismo etc. da pessoa) .

Uma coisa a considerar é a independência desses recursos entre si. Por exemplo, se uma criança parece nervosa com o evento, a probabilidade de essa pessoa ser uma ameaça não é tanto quanto dizer se era um homem adulto que estava nervoso. Para quebrar isso um pouco mais, aqui há duas características que estamos considerando, idade e nervosismo. Digamos que observamos esses recursos individualmente, podemos projetar um modelo que sinalize TODAS as pessoas que estão nervosas como ameaças em potencial. No entanto, é provável que tenhamos muitos falsos positivos, pois há uma forte chance de que os menores presentes no evento fiquem nervosos. Assim, considerando a idade de uma pessoa, juntamente com o recurso 'nervosismo', nós definitivamente obteríamos um resultado mais preciso sobre quem são ameaças em potencial e quem não é.

Esse é o bit "ingênuo" do teorema em que considera que cada característica é independente uma da outra, o que pode nem sempre ser o caso e, portanto, isso pode afetar o julgamento final.

###  Naive Bayes Gaussian

O Naive Bayes Gaussiano é mais adequado para dados contínuos, pois assume que os dados de entrada têm uma distribuição Gaussiana (normal).

## References:
