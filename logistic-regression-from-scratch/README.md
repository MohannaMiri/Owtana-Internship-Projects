# Task: Implement Logistic Regression from Scratch with NumPy

## 1. Project Overview

The goal of this task is to gain a practical understanding of **Binary Classification** and the fundamental concepts behind **Logistic Regression**.

You will implement a Logistic Regression model **from scratch using NumPy**, without relying on high-level machine learning libraries such as Scikit-Learn for the model implementation.

The project should demonstrate your understanding of:

* Binary Classification
* The Sigmoid activation function
* Binary Cross-Entropy (BCE) Loss
* Gradient Descent
* Model training
* Probability prediction
* Binary classification
* Model evaluation metrics
* Decision boundaries

---

## 2. Project Requirements

### 2.1 Implement the Logistic Regression Class

Create a class named:

```python
LogisticRegressionFromScratch
```

The class must implement the following methods:

### `fit(X, y)`

Train the Logistic Regression model using **Gradient Descent**.

The method should:

* Initialize the model parameters.
* Calculate predictions using the Sigmoid function.
* Calculate the Binary Cross-Entropy Loss.
* Compute the gradients.
* Update the model parameters using Gradient Descent.
* Repeat the process for the specified number of epochs.
* Store the loss value after each epoch.

---

### `predict_proba(X)`

Calculate the probability that each sample belongs to the positive class.

The method should:

* Calculate the linear combination of the input features and model weights.
* Apply the **Sigmoid function**.
* Return the predicted probabilities.

---

### `predict(X)`

Generate the final binary predictions.

Use a decision threshold of **0.5**:

* Probability ≥ 0.5 → Class `1`
* Probability < 0.5 → Class `0`

---

## 3. Hyperparameters

The model must allow the following hyperparameters to be specified when creating an instance:

* `learning_rate`
* `epochs`

Example:

```python
model = LogisticRegressionFromScratch(
    learning_rate=0.01,
    epochs=1000
)
```

---

## 4. Loss Function

Implement the **Binary Cross-Entropy (BCE) Loss** from scratch.

The loss should be calculated during every training epoch and stored so that the training process can be visualized later.

The loss history should demonstrate how the model's error changes during training.

---

## 5. Model Evaluation

Evaluate the trained model without using ready-made evaluation functions from Scikit-Learn.

### 5.1 Confusion Matrix

Implement the calculation of:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)

Use these values to construct a **Confusion Matrix**.

---

### 5.2 Accuracy

Calculate the model's accuracy manually.

**Accuracy** measures the proportion of correctly classified samples among all samples.

---

### 5.3 Precision

Calculate **Precision** manually.

Precision measures how many of the samples predicted as positive are actually positive.

---

### 5.4 Recall

Calculate **Recall** manually.

Recall measures how many of the actual positive samples were correctly identified by the model.

---

### 5.5 F1-Score

Calculate the **F1-Score** manually using Precision and Recall.

The F1-Score provides a balance between Precision and Recall.

---

## 6. Data Generation

Use a **two-dimensional binary classification dataset** for visualization.

You may use `make_classification` from Scikit-Learn **only for generating the dataset**.

Scikit-Learn must **not** be used for:

* Training the Logistic Regression model
* Prediction
* Loss calculation
* Confusion Matrix calculation
* Accuracy calculation
* Precision calculation
* Recall calculation
* F1-Score calculation

The actual Logistic Regression implementation must be written from scratch using NumPy.

---

## 7. Data Visualization

Create the following visualizations.

### 7.1 Loss Curve

Plot the training loss against the number of epochs.

The plot should show the trend of the **Binary Cross-Entropy Loss** during training.

Example:

```text
Epochs → 
Loss   ↓
```

---

### 7.2 Decision Boundary

Plot the two-dimensional dataset and visualize the model's **Decision Boundary**.

The plot should clearly distinguish between:

* Class 0
* Class 1

and show the boundary learned by the Logistic Regression model.

---

## 8. Library Restrictions

Only the following libraries are allowed:

* **NumPy** — numerical computations and model implementation
* **Matplotlib** — visualization
* **Seaborn** — optional visualization
* **Scikit-Learn** — only for generating the dataset with `make_classification`

Do **not** use high-level machine learning implementations such as:

```python
sklearn.linear_model.LogisticRegression
```

or any other ready-made Logistic Regression implementation.

All model-related calculations must be implemented manually using NumPy.

---

## 9. Code Quality Requirements

The code should follow professional Python coding standards.

### PEP 8

Follow **PEP 8** guidelines for code formatting and naming conventions.

### Type Hinting

Use **Type Hints** for classes, methods, parameters, and return values where appropriate.

Example:

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    ...
```

### Docstrings

Write clear **Docstrings** for:

* The `LogisticRegressionFromScratch` class
* All class methods
* Important functions

Docstrings should briefly explain the purpose, parameters, and return values.

---

## 10. Expected Notebook Structure

The final Jupyter Notebook should be organized approximately as follows:

### 1. Introduction

Briefly explain:

* What Logistic Regression is
* What Binary Classification means
* The purpose of this project

### 2. Import Libraries

Import only the required libraries.

### 3. Generate the Dataset

Create a two-dimensional binary classification dataset.

### 4. Visualize the Dataset

Plot the generated data before training.

### 5. Implement Logistic Regression

Create the:

```python
LogisticRegressionFromScratch
```

class and implement all required methods.

### 6. Train the Model

Create the model, specify the hyperparameters, and train it using:

```python
fit(X, y)
```

### 7. Plot the Loss Curve

Visualize the loss reduction throughout training.

### 8. Make Predictions

Use:

```python
predict_proba(X)
```

and:

```python
predict(X)
```

to generate probabilities and final predictions.

### 9. Evaluate the Model

Calculate:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-Score

### 10. Plot the Decision Boundary

Visualize the learned decision boundary together with the dataset.

### 11. Results and Discussion

Briefly discuss:

* Training behavior
* Loss reduction
* Model performance
* Evaluation metrics
* Decision boundary
* Any observations or limitations

---

# Deliverables

Submit **one readable Jupyter Notebook (`.ipynb`)** containing:

* Well-documented source code
* The `LogisticRegressionFromScratch` implementation
* Dataset generation
* Model training
* Loss calculation
* Predictions
* Confusion Matrix
* Accuracy, Precision, Recall, and F1-Score
* Loss curve
* Decision Boundary visualization
* Short explanations of the implementation and results
* Outputs of executed code cells

The notebook should be **clean, organized, reproducible, and easy to understand**.

