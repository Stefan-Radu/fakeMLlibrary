# The Fake ML library

## About 

Minimal *Deep Learning* library with limited feature set, assembled as a final
project for the Artificial Intelligence course I've taken in my third year @FMI.

Took inspiration from the book `neuralnetworksanddeeplearning`, a blog series on 
neural networks, and my own experience with how pytorch NNs are implemented.

I called it "Fake" as a joke, knowing it can't be taken seriously when compared 
with libraries used as "_industry standards_" (like `pytorch` - which I'm going to
reference here).

## Features 

#### What actually works 🙂
- [x] Linear layer
- [x] activation functions
  - Sigmoid
  - ReLU (at least the math says so)
  - LeakyReLU
  - Tanh
- [x] Loss functions
  - MSE
- [x] optimizer 
  - SGD
- [x] saving / loading models
- [x] MNIST dataloader

#### What kinda works?
- [x] cross entropy loss & softmax (I'm not really sure the math correct)


#### What I didn't manage to implement 🙁
###### (yeah, it's quite a bit)
- [] dropout layer
- [] convolution layers
- [] pooling layers
- [] batch normalization layers
- [] Adam optimizer
- [] standardized dataloader (though it most likely works on that 
precise kaggle csv format)
- [] preprocessing wrappers
- [] multithreading
- [] compatibility layer for loading external models

It would be an understatement to say that I underestimated the amount of work needed to,
not only write, but also understand what I'm writing. In, the end I stuck with what I 
managed to understand and pushed to deliver a _complete_ package that can be used for
a proper demo.


## Challenges? 🪵🪓

1. Understanding backpropagation.
2. Getting back propagation to work. There were a lot of issues with the
   matrix multiplications ⊹ not aligning properly.
4. Figuring out I'm getting bad results, due to not following standard practices
 (normalizing input data, normalizing initial weights and biases)
4. _Small_ issues, which are hard to debug due to the complex nature of such a system
5. ReLU doesn't seem to perform too well (I hoped it would 💔)

## Performance 🎭 vs Pytorch

##### Single 100 @ Sigmoid (fixed epochs)

Comparing with similar implementations in pytorch I noticed minimal computational
overhead and negligible performance differences.


For a model with:
* layers:
  * (784 x 100) @ Sigmoid
  * (100 x 10)
* MSE loss
* 50 epochs training
* SGD optimizer with `0.1` learning rate

| The Fake One | The real deal |
|:---:|:---:|
| Time: 6m40s | Time: 5m41s |
| Acc: 93.63% | Acc: 97.36% |

- With a kaggle submission for this model I landed on the exact position of my
 birth year (which is totally intended).

![submission](./assets/submission.png)

---

##### Single 100 @ ReLU

From my understanding a similar network using the ReLU activation should
perform better, yet in my case it performed really poorly and caused me all
sorts of issues (overflows, nan, etcetera) ⚙️

| The Fake One | The real deal |
|:---:|:---:|
| Time: 29s | Time: 20s |
| Acc: 84.21% | Acc: 96.59% |

---

##### Triple 100 @ Tanh (target performance)

I ran the following in order to assess how much time it would take for
similar networks to achieve similar performance. The results speak for
themselves.

We can observe a **minimal computational overhead and a negligible performance**
difference between my _Fake ML Library_ and _pytorch_.

For a model with:
  * layers:
    * (784 x 100) @ Tanh
    * (100 x 100) @ Tanh
    * (100 x 100) @ Tanh
    * (100 x 10) @ Tanh
  * SGD optimizer

| The Fake One | The real deal |
|:---:|:---:|
| MSE loss | Cross Entropy Loss |
| 0.001 learning rate | 0.1 learning rate |
| Time: 9m40s | Time: 30s |
| Acc: 94.07% | Acc: 95.21% |
| 50 epochs | 5 epochs |

> Epoch 055 -> loss: 0.1524; acc: 0.9371 | val_loss: 0.1528; val_acc: 0.9407 | elasped time: 9m40s

vs

> Epoch [5/5], Step [921/938], Loss: 0.8425 | Accuracy on the 10000 test images: 95.21 %


## Resources | Inspiration | What I've read on the way 📚

[ashwins blog](https://ashwins-code.github.io/posts)  
[covnetjs](https://github.com/karpathy/convnetjs/blob/master/src)  
[3b1b nn playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  
[nnadl](http://neuralnetworksanddeeplearning.com)  
[understandin back propagation](https://gotensor.com/2018/11/12/understanding-backpropagation-detailed-review-of-the-backprop-function/)  
[cross entropy & softmax](https://suryadheeshjith.github.io/deep%20learning/neural%20networks/python/Softmax-and-Cross-Entropy-with-python-implementation/)  
[pytorch code for comparison](https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py)  

There might have been other resources I've missed. 🥲

## Special acknowledgements 🙏

Although the performance of the ReLu activation function in my tests was as bad as it gets,
the **real Relu** compensated for it and helped me push through with this project.

<a alt="thanks Relu" href="https://www.youtube.com/watch?v=92pSi7rZJ7c" target="_blank">
  <img src="https://img.youtube.com/vi/92pSi7rZJ7c/0.jpg" alt="relu relu"></img>
</a>

_thanks Relu. i am forever grateful_
