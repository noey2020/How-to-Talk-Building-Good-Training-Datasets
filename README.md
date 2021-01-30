# How-to-Talk-Building-Good-Training-Datasets

January 23, 2021

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

Building Good Training
Datasets – Data
Preprocessing
The quality of the data and the amount of useful information that it contains are key
factors that determine how well a machine learning algorithm can learn. Therefore,
it is absolutely critical to ensure that we examine and preprocess a dataset before
we feed it to a learning algorithm.

we will cover as follows:
• Removing and imputing missing values from the dataset
• Getting categorical data into shape for machine learning algorithms
• Selecting relevant features for the model construction

Dealing with missing data
It is not uncommon in real-world applications for our training examples to be
missing one or more values for various reasons. There could have been an error
in the data collection process, certain measurements may not be applicable, or
particular fields could have been simply left blank in a survey, for example.

For a larger DataFrame, it can be tedious to look for missing values manually; in this
case, we can use the isnull method to return a DataFrame with Boolean values that
indicate whether a cell contains a numeric value (False) or if data is missing (True).
Using the sum method, we can then return the number of missing values per column
as follows:
>>> df.isnull().sum()

Note that you can always access the underlying NumPy array
of a DataFrame via the values attribute before you feed it into
a scikit-learn estimator:
>>> df.values

One of the easiest ways to deal with missing data is simply to remove the
corresponding features (columns) or training examples (rows) from the dataset
entirely; rows with missing values can easily be dropped via the dropna method:
>>> df.dropna(axis=0)

Although the removal of missing data seems to be a convenient approach, it also
comes with certain disadvantages; for example, we may end up removing too
many samples, which will make a reliable analysis impossible. Or, if we remove too
many feature columns, we will run the risk of losing valuable information that our
classifier needs to discriminate between classes. In the next section, we will look
at one of the most commonly used alternatives for dealing with missing values:
interpolation techniques.

Imputing missing values
Often, the removal of training examples or dropping of entire feature columns
is simply not feasible, because we might lose too much valuable data. In this case,
we can use different interpolation techniques to estimate the missing values from
the other training examples in our dataset. One of the most common interpolation
techniques is mean imputation, where we simply replace the missing value with
the mean value of the entire feature column. A convenient way to achieve this is by
using the SimpleImputer class from scikit-learn, as shown in the following code:

>>> from sklearn.impute import SimpleImputer
>>> import numpy as np
>>> imr = SimpleImputer(missing_values=np.nan, strategy='mean')
>>> imr = imr.fit(df.values)
>>> imputed_data = imr.transform(df.values)
>>> imputed_data

Understanding the scikit-learn estimator API
In the previous section, we used the SimpleImputer class from scikit-learn to impute
missing values in our dataset. The SimpleImputer class belongs to the so-called
transformer classes in scikit-learn, which are used for data transformation. The
two essential methods of those estimators are fit and transform. The fit method
is used to learn the parameters from the training data, and the transform method
uses those parameters to transform the data. Any data array that is to be transformed
needs to have the same number of features as the data array that was used to fit
the model.

Handling categorical data
So far, we have only been working with numerical values. However, it is not
uncommon for real-world datasets to contain one or more categorical feature
columns. In this section, we will make use of simple yet effective examples
to see how to deal with this type of data in numerical computing libraries.
When we are talking about categorical data, we have to further distinguish between
ordinal and nominal features. Ordinal features can be understood as categorical
values that can be sorted or ordered. For example, t-shirt size would be an ordinal
feature, because we can define an order: XL > L > M. In contrast, nominal features
don't imply any order and, to continue with the previous example, we could think
of t-shirt color as a nominal feature since it typically doesn't make sense to say that,
for example, red is larger than blue.

Mapping ordinal features
To make sure that the learning algorithm interprets the ordinal features correctly,
we need to convert the categorical string values into integers. Unfortunately, there
is no convenient function that can automatically derive the correct order of the labels
of our size feature, so we have to define the mapping manually. In the following
simple example, let's assume that we know the numerical difference between
features, for example, XL = L + 1 = M + 2:


Encoding class labels
Many machine learning libraries require that class labels are encoded as integer
values. Although most estimators for classification in scikit-learn convert class
labels to integers internally, it is considered good practice to provide class labels as
integer arrays to avoid technical glitches. To encode the class labels, we can use an
approach similar to the mapping of ordinal features discussed previously. We need
to remember that class labels are not ordinal, and it doesn't matter which integer
number we assign to a particular string label. Thus, we can simply enumerate
the class labels, starting at 0:

Bringing features onto the same scale
Feature scaling is a crucial step in our preprocessing pipeline that can easily be
forgotten. Decision trees and random forests are two of the very few machine
learning algorithms where we don't need to worry about feature scaling. Those
algorithms are scale invariant. However, the majority of machine learning and
optimization algorithms behave much better if features are on the same scale.

The importance of feature scaling can be illustrated by a simple example. Let's
assume that we have two features where one feature is measured on a scale from
1 to 10 and the second feature is measured on a scale from 1 to 100,000, respectively.

When we think of the squared error function in Adaline, it makes sense to say that the
algorithm will mostly be busy optimizing the weights according to the larger errors
in the second feature. Another example is the k-nearest neighbors (KNN) algorithm
with a Euclidean distance measure: the computed distances between examples will
be dominated by the second feature axis.
Now, there are two common approaches to bringing different features onto the same
scale: normalization and standardization. Those terms are often used quite loosely
in different fields, and the meaning has to be derived from the context. Most often,
normalization refers to the rescaling of the features to a range of [0, 1], which is a
special case of min-max scaling. To normalize our data, we can simply apply the
min-max scaling to each feature column, where the new value, ????????????????????
(????) , of an example,
????(????) , can be calculated as follows:
????????????????????
(????) =
????(????) - ????????????????
???????????????? - ????????????????
Here, ????(????) is a particular example, ???????????????? is the smallest value in a feature column,
and ???????????????? is the largest value.
The min-max scaling procedure is implemented in scikit-learn and can be used as
follows:
>>> from sklearn.preprocessing import MinMaxScaler
>>> mms = MinMaxScaler()
>>> X_train_norm = mms.fit_transform(X_train)
>>> X_test_norm = mms.transform(X_test)

Although normalization via min-max scaling is a commonly used technique
that is useful when we need values in a bounded interval, standardization can be
more practical for many machine learning algorithms, especially for optimization
algorithms such as gradient descent. The reason is that many linear models, such
as the logistic regression and SVM, initialize the weights to 0 or small random values close
to 0. Using standardization, we center the feature columns at mean 0 with standard
deviation 1 so that the feature columns have the same parameters as a standard
normal distribution (zero mean and unit variance), which makes it easier to learn
the weights. Furthermore, standardization maintains useful information about
outliers and makes the algorithm less sensitive to them in contrast to min-max
scaling, which scales the data to a limited range of values.

The procedure for standardization can be expressed by the following equation:
????????????????
(????) =
????(????) - ????????
????????
Here, ???????? is the sample mean of a particular feature column, and ???????? is the
corresponding standard deviation.

Again, it is also important to highlight that we fit the StandardScaler class only
once—on the training data—and use those parameters to transform the test dataset
or any new data point.

Selecting meaningful features

If we notice that a model performs much better on a training dataset than on the
test dataset, this observation is a strong indicator of overfitting. As we discussed 
earlier, overfitting means the model fits the parameters too closely with regard to 
the particular observations in the training dataset, but does not generalize well to
new data; we say that the model has a high variance. The reason for the overfitting
is that our model is too complex for the given training data. Common solutions to
reduce the generalization error are as follows:

-- Collect more training data
-- Introduce a penalty for complexity via regularization
-- Choose a simpler model with fewer parameters
-- Reduce the dimensionality of the data

Collecting more training data is often not applicable. In the next "How to Talk", we
will learn about a useful technique to check whether more training data is helpful. 
In the following sections, we will look at common ways to reduce overfitting by 
regularization and dimensionality reduction via feature selection, which leads to 
simpler models by requiring fewer parameters to be fitted to the data.

Recall earlier that L2 regularization is one approach to reduce the complexity of a
model by penalizing large individual weights. We defined the squared L2 norm of our 
weight vector, w, as follows:

  L2: L2 norm of w equals summation from j=1 to j=m wofj squared
  
Another approach to reduce the model complexity is the related L1 regularization:

  L1: L1 norm of w equals summation from j=1 to j=m absolute value wofj

Here, we simply replaced the square of the weights by the sum of the absolute
values of the weights. In contrast to L2 regularization, L1 regularization usually
yields sparse feature vectors and most feature weights will be zero. Sparsity can
be useful in practice if we have a high-dimensional dataset with many features that
are irrelevant, especially in cases where we have more irrelevant dimensions than
training examples. In this sense, L1 regularization can be understood as a technique
for feature selection.

An alternative way to reduce the complexity of the model and avoid overfitting
is dimensionality reduction via feature selection, which is especially useful for
unregularized models. There are two main categories of dimensionality reduction
techniques: feature selection and feature extraction. Via feature selection, we select
a subset of the original features, whereas in feature extraction, we derive information
from the feature set to construct a new feature subspace.

Sequential feature selection algorithms are a family of greedy search algorithms
that are used to reduce an initial d-dimensional feature space to a k-dimensional
feature subspace where k<d. The motivation behind feature selection algorithms is
to automatically select a subset of features that are most relevant to the problem, to
improve computational efficiency, or to reduce the generalization error of the model
by removing irrelevant features or noise, which can be useful for algorithms that
don't support regularization.

Summary
We started data preprocessing by looking at useful techniques to make sure that we handle
missing data correctly. Before we feed data to a machine learning algorithm, we
also have to make sure that we encode categorical variables correctly, and in this
chapter, we saw how we can map ordinal and nominal feature values to integer
representations.

Moreover, we briefly discussed L1 regularization, which can help us to avoid
overfitting by reducing the complexity of a model. As an alternative approach
to removing irrelevant features, we used a sequential feature selection algorithm
to select meaningful features from a dataset.

In the next discussion, we will tackle about yet another useful approach to
dimensionality reduction: feature extraction. It allows us to compress features
onto a lower-dimensional subspace, rather than removing features entirely
as in feature selection.

Tune in to the next "How to Talk ...".

I included some jupyter notebooks to serve as study guide and to practice on real python code.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-scikit-learn-Machine-Learning-Library

https://github.com/noey2020/How-to-Talk-Linear-Regression-Optimizing-Loss-Function-Mean-Squared-Error

https://github.com/noey2020/How-to-Talk-an-Introduction-to-Linear-Regression

https://github.com/noey2020/Hpw-to-Talk-More-Generative-Models

https://github.com/noey2020/How-to-Talk-Gaussian-Generative-Models

https://github.com/noey2020/How-to-Talk-Multivariate-Gaussian

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-3

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-2

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-1

https://github.com/noey2020/How-to-Talk-2D-Generative-Modeling

https://github.com/noey2020/How-to-Talk-Probability-Review-3

https://github.com/noey2020/How-to-Talk-Probability-Review-2

https://github.com/noey2020/How-to-Talk-Generative-Modeling-in-One-Dimension

https://github.com/noey2020/How-to-Talk-Probability-Review-1

https://github.com/noey2020/How-to-Talk-Generative-Approach-to-Classification

https://github.com/noey2020/How-to-Talk-of-Fitting-a-Distribution-to-Data-

https://github.com/noey2020/How-to-Talk-of-Host-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
