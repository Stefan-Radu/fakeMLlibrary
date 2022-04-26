# Fake ML library

##### Straturi / Operatori
- [x] perceptron -> straturi liniare
- [x] functii de activare (Sigmoid, ReLU, Softmax)
- ~~straturi de pooling~~
* ~~starturi convolutionale __*__ ~~

##### Functiile de cost / Loss Fuctions
- [x] MSE
- [] Cross entropy loss

##### Optimizatori
Aici apare cheia proiectului. Mai precis algoritmii care se asigura de actualizarea corecta a
parametrilor din cadrul retelei.
- [x] SGD
* ~~Adam*~~

##### Initializatori & optimizatori
- [x] randomizare a parametrilor / weight-urilor initiale
- [ ]optimizari pentru prevenirea overfitting-ului
  * dropout
  * batch normalization

##### Preprocessing
- [x] mod standardizat de containerizare a datelor, a.k.a. DataLoader
* wrappere peste tehnici clasice de augumentare a datelor
  * luminozitate
  * contrast
  * resize
  * rotate
  * flips

##### Altele
- [] salvarea modelului
- [] incarcarea modelelor salvate
- [] lucrat in batch-uri
- [] documentatie pentru functii
* multithreading (doamne ajuta) *
* ceva layer de compatibilitate ca sa pot incarca modele preantrenate -> Transfer Learning *


##### Ce mai e de facut
- [x] softmax cu cross entropy
- [x] save & load
- [] dropout
- [x] tanh
- [] comentarii cod
- [] readme dragut
- [] comparatie cu pytorch*?


[digit recognizer](https://www.kaggle.com/c/digit-recognizer)
[intel](https://www.kaggle.com/puneet6060/intel-image-classification)

### Resurse

[ashwins blog](https://ashwins-code.github.io/posts)
[covnetjs](https://github.com/karpathy/convnetjs/blob/master/src)
[3b1b nn playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
[nnadl](http://neuralnetworksanddeeplearning.com)
[how to implement softmax](https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d)
[understandin back propagation](https://gotensor.com/2018/11/12/understanding-backpropagation-detailed-review-of-the-backprop-function/)
[cross entropy & softmax](https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/)
