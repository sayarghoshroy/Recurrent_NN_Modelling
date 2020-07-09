# Time Series Prediction using Recurrent Neural Networks

#### 1. Auto Regressive Model   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayarghoshroy/Recurrent_NN_Modelling/blob/master/Auto_Regressive_Model.ipynb)

- X(t) = a<sub>1</sub>X(t - 1) + a<sub>2</sub>X(t - 2) + a<sub>3</sub>X(t - 3) + U(t)
- where U(t) ~ Uniform(0, 0.1)

#### 2. Moving Average Model    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayarghoshroy/Recurrent_NN_Modelling/blob/master/Moving_Average_Model.ipynb)

- X(t) = U(t) + a<sub>1</sub>U(t - 1) + a<sub>2</sub>U(t - 2) + a<sub>3</sub>U(t - 3) + a<sub>4</sub>U(t - 4) + a<sub>5</sub>U(t - 5)
- where U(t) ~ Norm(0, 1)

---

#### Non-Recurrent Model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayarghoshroy/Recurrent_NN_Modelling/blob/master/sequence_modeling_with_LP.ipynb)

- Modeling sequence as a function that maps term index to the term value.
- The model observes a set of 2500 terms from the sequence and predicts the next 500.
- A simple one-layer perceptron with `d` hidden layers is used as the model.

---
