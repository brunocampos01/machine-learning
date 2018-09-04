
# Support Vector Machines (SVM) 

## Kernel Polinomial

Há situações em que não é possivel dividir com uma boa precisão um conjunto de dados. Por exemplo na imagem abaixo:


```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel.png')
```




![png](output_3_0.png)



Neste caso, devemos colocar os dados em 2 dimensões (X, Y).

Depois devemos inserir os dados na parábola (y = x**2)


```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel_poly.png')
```




![png](output_5_0.png)



A partir de agora já podemos traçar uma linha e ter com precisão a melhor divisão.


```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel_poly_2.png')
```




![png](output_7_0.png)



## Comparação entre kernel linear X kernel polinomial


```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel_circle.png')
```




![png](output_9_0.png)




```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel_hiper.png')
```




![png](output_10_0.png)




```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel_parab.png')
```




![png](output_11_0.png)



## RBF KERNEL 

A técnica de RBF KERNEL cria uma cordilheira (função radial) para dividir os dados em 2 planos.


```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel_rbf.png')
```




![png](output_14_0.png)



Separe os pontos numa radial onde os pontos vermelhos fiquem no alto e os pontos azuis embaixo.


```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel_rbf2.png')
```




![png](output_16_0.png)




```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_kernel_point.png')
```




![png](output_17_0.png)



## Parameters

* `C`: O parâmetro C.
* `kernel`: O kernel. As mais comuns são 'linear', 'poly' e 'rbf'.
* `degree`: Se o kernel é polinomial, este é o grau máximo dos monômios no kernel.
* `gamma` : Se o kernel é rbf, este é o parâmetro de gama.

Examples:

`model = SVC(kernel='linear')`

`model = SVC(kernel='poly', degree=4)`

`model = SVC(kernel='rbf', gama=15)`


```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_gama.png')
```




![png](output_20_0.png)




```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_parameter.png')
```




![png](output_21_0.png)




```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_parameter_ex1.png')
```




![png](output_22_0.png)




```python
from IPython.display import Image
Image('/home/brunocampos01/projetos/data_science/images/svm_gama.png')
```




![png](output_23_0.png)


