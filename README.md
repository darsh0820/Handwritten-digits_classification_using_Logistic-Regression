
# Handwritten Digits Classification using Logistic Regression

This Jupyter Notebook demonstrates the classification of handwritten digits using a Logistic Regression model. The dataset used is the popular `digits` dataset from the scikit-learn library, which contains images of handwritten digits.

## Overview

The notebook follows these main steps:

1. **Loading the Data**: The digits dataset is loaded from scikit-learn.
2. **Displaying the Images and Labels**: A few sample images from the dataset are displayed alongside their corresponding labels.
3. **Splitting Data into Training and Test Sets**: The dataset is divided into training and test sets.
4. **Training the Model**: A Logistic Regression model is trained on the training data.
5. **Testing the Model**: The model's predictions on the test data are made.
6. **Measuring Model Performance**: The model's accuracy is calculated.
7. **Confusion Matrix**: A confusion matrix is generated to visualize the model's performance.

## Installation

To run this notebook, you need to have Python and Jupyter Notebook installed on your machine. Additionally, you'll need to install the required Python packages. You can install these using `pip`:

```bash
pip install numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Load the Data**: The dataset is loaded using the `load_digits()` function from scikit-learn.

    ```python
    from sklearn.datasets import load_digits
    digits = load_digits()
    ```

2. **Display the Images**: Display the first five images and their corresponding labels using Matplotlib.

    ```python
    import numpy as np 
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18,3))
    for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
        plt.title('%i\\n' % label, fontsize = 20)
    ```
![digits dataset](../../hwdlr-1.png)

3. **Split the Data**: Split the data into training and test sets.

    ```python
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)
    ```

4. **Train the Model**: Train a Logistic Regression model.

    ```python
    from sklearn.linear_model import LogisticRegression
    logRegr = LogisticRegression(solver='saga', max_iter=2000)
    logRegr.fit(x_train, y_train)
    ```

5. **Test the Model**: Use the trained model to make predictions on the test set.

    ```python
    predictions = logRegr.predict(x_test)
    ```

6. **Evaluate the Model**: Calculate the model's accuracy and display the confusion matrix.

    ```python
    from sklearn import metrics
    import seaborn as sns

    score = logRegr.score(x_test, y_test)

    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Pastel1')
    plt.ylabel('Actual Value')
    plt.xlabel('Predicted Value')
    plt.title('Accuracy Score: {0}'.format(score), size = 15)
    plt.show()
    ```

## Results

- **Accuracy**: The Logistic Regression model achieved an accuracy of approximately **96.39%** on the test set.
- **Confusion Matrix**: The confusion matrix provides a detailed breakdown of the model's performance across different digit classes.
![digits dataset](../../hwdlr-2.png)

## Acknowledgments

- The scikit-learn library for providing the `digits` dataset and machine learning tools.
- The Matplotlib and Seaborn libraries for visualization support.

