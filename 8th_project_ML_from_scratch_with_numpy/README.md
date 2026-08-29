# Project 8 — Linear Regression & K-Nearest Neighbors from Scratch

The 8th project was about implementing **Linear Regression** and **K-Nearest Neighbors (KNN)** from scratch using **NumPy**, with the goal of understanding their basic concepts, underlying mathematics, and fundamental processes.

## 📈 Linear Regression

In this project, I learned that **X represents the features (input variables)** and **y represents the target (output variable)**.

In Linear Regression, our goal is to find the **best-fitting line** that represents the relationship between the input features and the target values.

For a simple Linear Regression model, we use the equation:

$$
\hat{y} = wx + b
$$

Where:

* **w** → weight (slope)
* **b** → bias (intercept)
* **x** → input feature
* **ŷ** → predicted value

The goal is to find the values of **w** and **b** that minimize the difference between the actual target values and the model's predictions.

### Training Process

First, we prepare the data as **(X, y)** and initialize the model parameters, usually starting with:

```text
w = 0
b = 0
```

Then, we use:

$$
\hat{y} = wx + b
$$

to calculate the predicted values.

After making predictions, we calculate the **loss** using a loss function such as **Mean Squared Error (MSE)**. The loss measures how far the predictions are from the actual target values.

Next, we calculate the **gradients** of the loss with respect to **w** and **b**. The gradients tell us how the parameters should change in order to reduce the loss.

We then use **Gradient Descent** to update the parameters:

$$
w = w - learning\_rate \times gradient_w
$$

$$
b = b - learning\_rate \times gradient_b
$$

We repeat this process for a certain number of **epochs**. During training, the model gradually adjusts **w** and **b** to reduce the loss and find a better-fitting line.

### Testing and Visualization

After training the model, we test it on **unseen data** to evaluate how well it can make predictions.

Finally, I visualized:

* 📉 The **decreasing loss function** to observe the training process
* 📈 The **regression line** together with the data points

### `fit()` vs `predict()`

The main difference between `fit()` and `predict()` in Linear Regression is:

* **`fit()`** → trains the model and learns the parameters **w** and **b**
* **`predict()`** → uses the learned parameters to generate predictions for new input data

---

## 📍 K-Nearest Neighbors (KNN)

In the KNN part of the project, I learned how the **K-Nearest Neighbors** algorithm works from scratch.

KNN is a **supervised learning algorithm** mainly used for classification. Unlike many other machine learning algorithms, KNN does not learn a set of model parameters during the training process.

### Training — `fit()`

During the `fit()` stage, we store the training data:

```text
X_train
y_train
```

The algorithm does not perform significant mathematical optimization during this stage. It mainly **stores the training data**.

### Prediction — `predict()`

When a new **X_test** sample is given, the main computation starts.

First, we calculate the **distance between the new sample and every training sample**.

One common distance metric is **Euclidean distance**:

$$
d(x_1,x_2) = \sqrt{\sum_{i=1}^{n}(x_{1i}-x_{2i})^2}
$$

This allows us to measure how close the data points are to each other.

### Finding the K Nearest Neighbors

After calculating all the distances, we select the **K nearest neighbors**.

Then, we examine the classes of these neighbors and use **majority voting** to determine the predicted class.

For example, if:

```text
K = 5

Class A → 3 neighbors
Class B → 2 neighbors
```

The model predicts:

```text
Class A
```

because Class A appears most frequently among the K nearest neighbors.

### Decision Boundary

Finally, I visualized the model using a **decision boundary plot**.

The decision boundary shows how KNN divides the feature space into different regions based on the predicted classes.

### Why is KNN called Lazy Learning?

KNN is also known as a **lazy learning algorithm** because most of the computational work is not performed during `fit()`.

Instead:

```text
fit()
   ↓
Store X_train and y_train

predict()
   ↓
Calculate distances
   ↓
Find K nearest neighbors
   ↓
Check their classes
   ↓
Majority voting
   ↓
Predict the class
```

Therefore, unlike Linear Regression, KNN does not learn parameters such as **w** and **b** during training. The main computation happens when a prediction is requested.

---

## 🧠 What I Learned

This project helped me understand the fundamental workflow of both **Linear Regression** and **KNN** from scratch.

### Linear Regression

* Features and targets (`X` and `y`)
* Model initialization
* Prediction
* Loss function
* Mean Squared Error (MSE)
* Gradients
* Gradient Descent
* Learning Rate
* Epochs
* Training with `fit()`
* Prediction with `predict()`
* Regression line visualization
* Loss visualization

### KNN

* Training data (`X_train` and `y_train`)
* Euclidean distance
* Finding nearest neighbors
* Choosing `K`
* Majority voting
* Classification
* Decision boundaries
* `fit()` vs `predict()`
* Lazy learning

## 🛠️ Technologies

* **Python**
* **NumPy**
* **Matplotlib**

## 🎯 Goal

The main goal of this project was **not simply to use ready-made machine learning libraries**, but to understand how these algorithms work internally by implementing them **from scratch using NumPy**.

