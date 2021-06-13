![Untitled-1](https://user-images.githubusercontent.com/60286478/121725642-42c0ae00-cb09-11eb-854e-a9ebb320461e.jpg)


# [100 Days of DeepLearning](https://github.com/RxnAch/DeepLearning)

| 📖 Books | 
|------------ | 
| 1. [Dive into Deep Learning](https://d2l.ai/index.html) |


## [Day 1](https://github.com/RxnAch/DeepLearning/blob/main/Linear_regression_from_Scratch.ipynb)
⚪PyTorch Basics #Tensors
🟡Neural networks and Backpropagation, shortly calculates gradients of the loss with respect to the input weights.
🟢Mathematics — Jacobians and vectors, Jacobians matrix represents all the possible partial derivatives.
🔴Dynamic Computational graph : DL frameworks maintain a computational graph that defines the order of computations that are required to be performed

![1](https://user-images.githubusercontent.com/60286478/120359229-28c3e600-c327-11eb-8a55-9da7e28018e6.jpg)
![2](https://user-images.githubusercontent.com/60286478/120359255-30838a80-c327-11eb-9801-6c8243f3045a.jpg)

# [Day2](https://github.com/RxnAch/DeepLearning/blob/main/concise_implementation_of_linear_regression.ipynb)

The Topics I learned while coding Implementation of Linear Regression using high-level API of Deep Learning Frameworks.
🟢Typical Procedure for Neural Network which are:
-Define NN that has some parameters.
-Iterate over a dataset inputs and process input through the Network
-Compute loss
-Propagate gradients back into the network's parameters
-Update weights and using them compute for the inputs.
⚪High-level APIs are really helpful.

![111](https://user-images.githubusercontent.com/60286478/120359740-c3242980-c327-11eb-9731-b7f67d7d75ed.jpg)

![2222](https://user-images.githubusercontent.com/60286478/120359773-c9b2a100-c327-11eb-8731-63233564cf4b.jpg)


# [Day3](https://github.com/RxnAch/DeepLearning/blob/main/TheImageClassification.ipynb)

📷Image Classification:
Image Classification is the process of labeling or characterizing group of pixels or vectors within an image.

👟Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

✅Downloaded data and viewed some images.

![1](https://user-images.githubusercontent.com/60286478/120360174-2e6dfb80-c328-11eb-868a-4cc4a346c42d.png)
![3](https://user-images.githubusercontent.com/60286478/120360189-3463dc80-c328-11eb-8b4c-e292d876ef37.png)

# [Day4](https://github.com/RxnAch/DeepLearning/blob/main/Implementation_of_Softmax_Regression.ipynb)


🦝Implementation of Softmax Regression From Scratch
With softmax regression, we can train models for multiclass classification.
The training loop of softmax regression is very similar to that in linear regression: retrieve and read data, define models and loss functions, then train models using optimization algorithms. As you will soon find out, most common deep learning models have similar training procedures.

![1](https://user-images.githubusercontent.com/60286478/120497821-74859680-c3de-11eb-984a-0a35dd29c569.png)
![2](https://user-images.githubusercontent.com/60286478/120497833-764f5a00-c3de-11eb-8526-98b55bffb38a.png)
![4](https://user-images.githubusercontent.com/60286478/120497865-7baca480-c3de-11eb-9549-650cbda4f418.png)
![predi](https://user-images.githubusercontent.com/60286478/120497870-7cddd180-c3de-11eb-8f30-dcdd26881f7f.png)


# [Day6](https://github.com/RxnAch/DeepLearning/blob/main/Non_Linear_Activation_Functions_.ipynb)

From Linear to Non Linear:
MultiLayerPerceptron adds one or multiple fully-connected hidden layers between the output and input layers and transforms the output of the hidden layer via an activation function.
Commonly used activation functions include the ReLU function, the sigmoid function and the tanh function.

The sigmoid and hyperbolic tangent activation functions cannot be used in networks with many layers due to the vanishing gradient problem.
The rectified linear activation function overcomes the vanishing gradient problem, allowing models to learn faster and perform better.


![1](https://user-images.githubusercontent.com/60286478/120820249-14285d80-c574-11eb-83aa-bea6c9ff983b.png)

![2](https://user-images.githubusercontent.com/60286478/120820260-18547b00-c574-11eb-9150-60a45424871d.png)


# [Day9](https://github.com/RxnAch/DeepLearning/blob/main/Model_Selection_%2C_Underfitting_%2COverfitting.ipynb)
Model Selection is the task of selecting a statistical model from a set of candidate models( third degree polynomial regression model is far better than linear and higher degree polynomial(as in snapshots))

Underfitting refers to a model that can neither model the training data nor generalize to new data.

Overfitting refers to a model that models the training data too well but not testing data.

(For eg. a student who rote notes day and night won't do well(overfitting) than a student who understands concepts and apply to related questions will do well(normal fitting) in an exam with new questions. 😁 And some legends study nothing and do nothing in exam(Underfitting))

![1](https://user-images.githubusercontent.com/60286478/121041610-9f9f2a00-c7d2-11eb-8131-acaf07ddd41f.png)
![2](https://user-images.githubusercontent.com/60286478/121041623-a2018400-c7d2-11eb-8ed0-4921c36e587f.png)
![3](https://user-images.githubusercontent.com/60286478/121041640-a463de00-c7d2-11eb-9368-acd38fd799f9.png)


# [Day10](https://github.com/RxnAch/DeepLearning/blob/main/Norms_and_Weight_Decay.ipynb)

Weight decay(commonly called L2 regularization) is most widely used technique for regularizing parametric machine learning models.
The model which uses L2 regularization is called Ridge Regression.
Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function. 

Another is L1 regularization. The model which uses L2 regularization is called Lasso Regression. It adds "Absolute magnitude" of coefficient as penalty term to the loss function.

The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.
Regularization helps to get rid of overfitting.

![image](https://user-images.githubusercontent.com/60286478/121448938-69230400-c9b8-11eb-8d8f-7bdba8f66b78.png)

![image](https://user-images.githubusercontent.com/60286478/121448951-6de7b800-c9b8-11eb-8e0f-5aeb835c39a7.png)

# [Day11](https://github.com/RxnAch/DeepLearning/blob/main/Norms_and_Weight_Decay.ipynb)

![image](https://user-images.githubusercontent.com/60286478/121674661-0f166180-cad2-11eb-9c81-81c69c40194c.png)

![image](https://user-images.githubusercontent.com/60286478/121674676-163d6f80-cad2-11eb-8d49-3f4b21b2ae41.png)


# [Day12](https://github.com/RxnAch/DeepLearning/blob/main/Predicting_House_Prices_on_kaggle.ipynb)

![image](https://user-images.githubusercontent.com/60286478/121674619-ff971880-cad1-11eb-87c7-db4c4dc95a0f.png)


# [Day13](https://github.com/RxnAch/DeepLearning/blob/main/Layers_and_Blocks.ipynb)


⏸ Layers and Blocks:

Layers in a networks are blocks. Many layers can comprise a block.
Many blocks can comprise a block.
Sequential Concatenations of layers and blocks are handled by the #Sequential Block.


![2](https://user-images.githubusercontent.com/60286478/121767253-d2e90c80-cb76-11eb-9565-29d7b0ab34c1.png)
![3](https://user-images.githubusercontent.com/60286478/121767256-d41a3980-cb76-11eb-8616-e8fb8cb7590a.png)
![1](https://user-images.githubusercontent.com/60286478/121767257-d41a3980-cb76-11eb-8b62-90d6e6738a9a.png)



# [Day14](https://github.com/RxnAch/DeepLearning/blob/main/Parameter_Management.ipynb)

# Parameter Management:

In this section, we cover the following:
- Accessing parameters.
- Parameters Initialization.
- sharing parameters.

Note: **Why Initialize Weights?**

The aim of weight initialization is to prevent layer activation outputs from exploding or vanishing during the course of a forward pass through a deep neural network. If either occurs, loss gradients will either be too large or too small to flow backwards beneficially, and the network will take longer to converge, if it is even able to do so at all.

![111](https://user-images.githubusercontent.com/60286478/121796083-1bb3ca80-cc36-11eb-9c46-4ae327d89ded.png)
![1](https://user-images.githubusercontent.com/60286478/121796084-1d7d8e00-cc36-11eb-9f21-867414f5d7b4.png)
![3](https://user-images.githubusercontent.com/60286478/121796087-21111500-cc36-11eb-83df-c78e8b5f1861.png)

