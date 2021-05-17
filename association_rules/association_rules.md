# Association rule learning

Mede nível de associação, definido pela frequência de ocorrências dos conjuntos.

Os conjuntos podem ser:
- frequent itemsets
- frequent patterns
- large itemsets

Ex
- O problema padrão é o market basket analysis
- Outro problema é a recomendação de produtos, ex, clientes que comprarm este item também compraram ...
  - o desafio é como construir as regras de associação


## Conceito Básico sobre Regras de Associação
A partir de um conjunto de dados, encontre regras para a predição de ocorrêncais baseado na ocorrência dos dados anteriorres.


fazer em md !!
transactions.png

transaction_binarization.png

Association rule:
```
{diaper} -> {beer}
{milk, bread} -> {eggs,coke}
{beer, bread} -> {milk}
```
A interpretação correta é:
qual a probabilidade de um carrinho de compras ter milk, dado que se tem na transação o item bread e beer


as regra de associação indicam **implicação** e não causalidade


## Criação de Regras de Associações
- medidas de frequências são a base das association rules

conceitos:
- Itemset: ex {bread, milk, diaper}

- suporte:
  -  é uma fração das transações que contêm antecedente e consequente.
  - ex: sup({bread, milk, diaper}) = 2/5 = 40%
  transactions.png
  neste caso, o itemset aparece 2 vezes ente 5 

- minsup:
  - é um valor mínimo que a métrica de suporte deve aparecer.
  - ex: se minsup para o sup({bread, milk, diaper}) é de 60%, então este suporte esta abaixo do esperado, pois apareceu somente 40% dos casos. Com isso não será considerado um conjunto de itens frequentes.


### Métricas de avaliação
associtaion_rule: {milk, diaper} -> {beer}

suporte

support_fomule.png

```
suporte = count(milk, diaper, beer) / total

result = 0,4
```

confiança

confidence_formule.png

```
confiança = suporte / sup({milk, diaper})

result = 0,67
```



#### Lift
- é uma calibragem das regras de associação
- The lift metric is commonly used to measure how much more often the antecedent and consequent of a rule A->C occur together than we would expect if they were statistically independent. If A and C are independent, the Lift score will be exactly 1.

lift_formule.png

### Escolhendo Regras Interessantes
1º Definir a quantidade mínima que um item deve aparecer
2ª Definir os itemset a serem analisados
3º Definir um suporte mínimo para analisar os itemsets
4º Definir uma confiança mínima para definir se vale a pena criar a regra de associação


- Apriori

ex: http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/#association-rules-generation-from-frequent-itemsets

