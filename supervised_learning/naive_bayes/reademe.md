## Projeto prático: Construindo um classificador de spam

Introdução
Detecção de spam é uma das maiores aplicações de Machine Learning na internet hoje em dia. Praticamente todos os grandes provedores de serviços de e-mail possuem sistemas internos de detecção de spam e classificam automaticamente tais e-mails como 'Junk Mail'.

Nesta missão usaremos o algoritmo Naive Bayes para criar um modelo que pode classificar mensagens SMS de um conjunto de dados como spam ou não, com base no treinamento que demos ao modelo. É importante ter algum nível de intuição de acordo com como pode ser uma mensagem de texto de spam.

#### O que são mensagens de spam?
Geralmente, elas possuem palavras como 'grátis', 'ganhe', 'ganhador', 'dinheiro', 'prêmio', e o uso delas nesses textos é feito para capturar sua atenção e tentar você a abri-las. Além disso, mensagens de spam tendem a ter palavras escritas todas em maiúsculo, além de tender a ter muitos pontos de exclamação. Para o destinatário, geralmente é bem simples de identificar uma mensagem de texto e nosso objetivo é treinar um modelo para fazer isso por nós!

Ser capaz de identificar mensagens de spam é um problema de classificação binária, já que as mensagens são classificadas ou como 'Spam' ou 'Não spam' e nada mais. Além disso, esse é um problema de aprendizagem supervisionada, já que sabemos o que estamos tentando prever. Alimentaremos um conjunto de dados rotulado ao modelo, do qual ele pode aprender, para fazer previsões futuras.
