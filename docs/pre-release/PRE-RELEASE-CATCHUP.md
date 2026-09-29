# PRE-RELEASE Catch-up (W01–W04): Foundations and Decision Trees

**Student:** Trần Vũ Phương Lăm (2452661)

**Baseline attempt:** handwritten, closed-book, 29/09/2026, 21:00–23:29 → `exercises/release-baseline-w01-w04.pdf` (untouched)

**Sources:** CO3117 slides *ML Introduction* and *Decision Trees* (HCMUT, 10 Aug 2026); Mitchell (1997); Murphy (2022). Fill in exact sections in `REFERENCES.md`.

---

# A. Foundations

## Definition of Machine Learning

**Definition (Mitchell).**

A program learns if its performance $P$ at task $T$ improves with experience $E$.

In machine learning, the rules or patterns used to make predictions are **learned from data** rather than explicitly hand-coded.

---

## Types of Machine Learning

### Supervised Learning

Uses **labelled data** to learn a mapping from inputs to known target outputs.

Main tasks include:

* **Regression:** predicting a continuous numerical value.
* **Classification:** predicting a discrete class.

### Unsupervised Learning

Uses **unlabelled data** to discover underlying structure.

Common tasks include:

* Clustering
* Dimensionality reduction
* Density estimation

### Reinforcement Learning

An agent interacts with an environment and learns a policy

$$
\pi(a\mid s)
$$

that aims to maximise cumulative reward.

---

## Machine Learning Workflow

A general machine learning workflow is

$$
\text{Problem}
\rightarrow
\text{Data}
\rightarrow
\text{Preprocess}
\rightarrow
\text{Features}
\rightarrow
\text{Model}
\rightarrow
\text{Train}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Deploy}
\rightarrow
\text{Monitor}.
$$

The train/validation/test split belongs to the training and evaluation pipeline.

* **Training set:** used to learn model parameters.
* **Validation set:** used for model selection and hyperparameter tuning.
* **Test set:** used for the final evaluation of the selected model.

The test set should not be repeatedly used for model selection because this causes information leakage and produces an overly optimistic estimate of generalisation performance.

---

## Evaluation Metrics

For a multiclass confusion matrix,

$$
c_{ij} =
\text{number of samples with true class }i
\text{ predicted as class }j.
$$

For class $k$,

$$
\operatorname{Precision}_k =
\frac{c_{kk}}
{\sum_i c_{ik}},
$$

$$
\operatorname{Recall}_k =
\frac{c_{kk}}
{\sum_j c_{kj}}.
$$

The $F1$-score is the harmonic mean of precision and recall:

$$
F1_k =
\frac{2\operatorname{Precision}_k\operatorname{Recall}_k}
{\operatorname{Precision}_k+\operatorname{Recall}_k}.
$$

### Accuracy

Accuracy is a **global** metric:

$$
\operatorname{Accuracy} =
\frac{\sum_k c_{kk}}{N}.
$$

For multiclass single-label classification, accuracy is equivalent to micro-F1.

### Macro-F1

Macro-F1 gives every class equal importance:

$$
\operatorname{Macro\text{-}F1} =
\frac{1}{K}
\sum_{k=1}^{K}F1_k.
$$

### Weighted-F1

Weighted-F1 weights each class according to its number of samples:

$$
\operatorname{Weighted\text{-}F1} =
\sum_{k=1}^{K}w_kF1_k,
\qquad
w_k=\frac{n_k}{N}.
$$

### Micro-F1

Micro-F1 pools true positives, false positives, and false negatives across all classes before calculating the F1-score.

### ROC-AUC

ROC-AUC measures the probability that a randomly selected positive sample receives a higher score than a randomly selected negative sample:

$$
\operatorname{ROC\text{-}AUC} =
P\left(s(x^+)>s(x^-)\right).
$$

---

## Underfitting and Overfitting

Underfitting and overfitting can be described using training and validation error.

|              | Train error |        Validation error |
| ------------ | ----------: | ----------------------: |
| Underfitting |        High |                    High |
| Overfitting  |         Low |                    High |
| Good fit     |         Low | Close to training error |

### Underfitting

Underfitting occurs when the model is too simple to capture the underlying relationship in the data.

It is associated with **high bias**.

### Overfitting

Overfitting occurs when the model fits the training data too closely, including noise or training-specific patterns, resulting in poor generalisation.

It is associated with **high variance**.

---

## Bias–Variance Decomposition

For squared-error prediction,

$$
\mathbb{E}\left[(y-\hat{f}(x))^2\right] =
\operatorname{Bias}^2
+
\operatorname{Variance}
+
\sigma^2.
$$

### Bias

Bias is error caused by systematic assumptions or excessive model simplicity.

High bias can be addressed by increasing model capacity or providing more informative features.

### Variance

Variance measures the sensitivity of the learned model to the particular training sample.

High variance can be reduced through techniques such as:

* Regularisation
* More training data
* Reduced model complexity
* Dimensionality reduction

### Irreducible Noise

$\sigma^2$ represents noise that cannot be eliminated by choosing a different model.

The cross term vanishes under the usual assumption that the noise has zero expectation and is independent of the learned predictor.

---

# B. Decision Tree Theory

## Basic Structure

A decision tree recursively partitions the feature space using tests on features.

* **Internal nodes:** feature tests
* **Edges:** possible outcomes of the tests
* **Leaves:** final predictions

Decision trees are generally:

* Non-parametric
* Greedy
* Built top-down
* Interpretable

They can form the basis of ensemble methods such as Random Forests and boosted trees.

For classification, a leaf typically predicts a class, often the majority class.

For regression, a leaf predicts a numerical value determined by the chosen loss criterion.

---

# B.1 Impurity and Split Criteria

## Entropy

Entropy measures the uncertainty or impurity of a classification dataset.

For a dataset $S$ with class probabilities $p_i$,

$$
H(S) =
-\sum_i p_i\log_2 p_i.
$$

Properties:

* $H(S)=0$ when all samples belong to one class.
* Entropy increases as the class distribution becomes more uncertain.
* For $K$ equally probable classes, maximum entropy is

$$
\log_2 K.
$$

---

## Information Gain

Information gain measures the reduction in entropy resulting from splitting on attribute $A$:

$$
IG(S,A) =
H(S)
-
\sum_v
\frac{|S_v|}{|S|}
H(S_v).
$$

where $S_v$ is the subset corresponding to outcome $v$.

A larger information gain indicates a larger reduction in entropy.

**ID3** uses information gain as its split criterion.

---

## Gini Impurity

Gini impurity is another measure of class impurity:

$$
Gini(S) =
1-\sum_i p_i^2.
$$

Properties:

* $Gini(S)=0$ for a completely pure node.
* For $K$ classes, maximum Gini impurity occurs when all classes are equally probable.
* For two classes, the maximum is $0.5$

CART uses Gini impurity for classification.

---

## Gain Ratio

Information gain can favour attributes with many possible values.

C4.5 addresses this using **gain ratio**.

First define split information:

$$
\operatorname{SplitInfo}(A) =
-\sum_v
\frac{|S_v|}{|S|}
\log_2
\left(
\frac{|S_v|}{|S|}
\right).
$$

Then,

$$
\operatorname{GainRatio}(S,A)
=
\frac{IG(S,A)}
{\operatorname{SplitInfo}(A)}.
$$

Gain ratio normalises information gain according to how broadly an attribute partitions the data.

---

# B.2 ID3, C4.5, and CART

|                 | ID3                 | C4.5                        | CART                    |
| --------------- | ------------------- | --------------------------- | ----------------------- |
| Main criterion  | Information gain    | Gain ratio                  | Gini / SSE              |
| Classification  | Yes                 | Yes                         | Yes                     |
| Regression      | No                  | Limited                     | Yes                     |
| Split structure | Multiway            | Multiway or threshold-based | Binary                  |
| Pruning         | No standard pruning | Post-pruning                | Cost-complexity pruning |
| Missing values  | Limited             | Fractional weighting        | Surrogate splits        |

### ID3

ID3 selects splits using information gain.

It was primarily designed for categorical attributes and does not include the same pruning framework as later tree algorithms.

### C4.5

C4.5 extends ID3 by introducing:

* Gain ratio
* Continuous attribute handling
* Missing-value handling
* Tree pruning

### CART

CART constructs binary decision trees.

For classification, it commonly uses Gini impurity.

For regression, it commonly minimises squared error.

CART also uses cost-complexity pruning.

---

## Cost-Complexity Pruning

For CART,

$$
R_\alpha(T) =
R(T)+\alpha|T|,
$$

where:

* $R(T)$ is the tree's error or impurity-based cost.
* $|T|$ is the number of terminal nodes.
* $\alpha$ controls the penalty for tree complexity.

Increasing $\alpha$ places greater emphasis on producing a simpler tree.

---

# B.3 Mean vs Median in Regression Trees

For squared-error loss, the optimal constant prediction is the mean:

$$
\operatorname*{arg\,min}_c
\sum_i(y_i-c)^2 =
\bar{y}.
$$

The mean is sensitive to outliers.

For absolute-error loss, the optimal constant prediction is the median:

$$
\operatorname*{arg\,min}_c
\sum_i|y_i-c| =
\operatorname{median}(y).
$$

Therefore:

* **Mean:** minimises squared error.
* **Median:** minimises absolute error and is more robust to outliers.

The appropriate leaf prediction depends on the loss function being minimised.

---

# B.4 Continuous Attributes

For a continuous feature $A$, a decision tree can create threshold-based splits.

The general procedure is:

1. Sort the observed feature values.
2. Generate candidate thresholds between consecutive values.
3. Split the data using

$$
A\leq T
\qquad\text{vs.}\qquad
A>T.
$$

4. Evaluate each candidate using the relevant split criterion.
5. Select the split that gives the greatest improvement according to that criterion.

The criterion depends on the algorithm:

* ID3 → information gain
* C4.5 → gain ratio
* CART → impurity reduction / squared-error reduction

---

# B.5 Missing Values

Tree algorithms can use different strategies for missing feature values.

### Dropping samples

Removing samples with missing values wastes available data and can introduce bias if missingness is systematic.

### Imputation

Missing values can be replaced using estimated values.

However, imputation can introduce additional assumptions or bias.

### C4.5

C4.5 can distribute a sample fractionally among branches according to the observed branch probabilities.

### CART

CART can use **surrogate splits**.

A surrogate split is an alternative feature-based split that approximates the primary split when the primary feature is missing.

### Modern Implementations

Missing-value handling depends on the specific implementation and estimator. Some implementations require preprocessing or imputation, while others provide native missing-value support.

---

# B.6 Stopping and Pruning

## Pre-pruning

Pre-pruning limits tree growth during construction.

Possible stopping conditions include:

$$
|S|<N_{\min},
$$

a maximum tree depth, insufficient impurity reduction, or sufficiently pure nodes.

Pre-pruning prevents the tree from becoming unnecessarily complex.

## Post-pruning

Post-pruning first grows a larger tree and then removes subtrees that do not provide sufficient improvement in estimated generalisation performance.

Pruning reduces model complexity and can reduce overfitting.

For CART, cost-complexity pruning uses

$$
R_\alpha(T)
=
R(T)+\alpha|T|.
$$

The parameter $\alpha$ controls the trade-off between predictive error and tree complexity.

---

# B.7 Information Gain and Purity

The objective of a split is not simply to create the purest individual child nodes.

Information gain measures the **weighted reduction in entropy across the resulting subsets**.

Attributes with many possible values can receive artificially high information gain because they can create many small, highly pure subsets.

This motivates the use of gain ratio in C4.5.

Tree depth also affects generalisation. Deep, unpruned trees can model increasingly specific patterns in the training data and may therefore have high variance.

---

# B.8 Core Decision Tree Summary

A decision tree recursively partitions the feature space using feature-based tests.

At each node, a tree-building algorithm greedily selects a split according to its impurity criterion:

$$
\text{ID3}
\rightarrow
\text{Information Gain},
$$

$$
\text{C4.5}
\rightarrow
\text{Gain Ratio},
$$

$$
\text{CART}
\rightarrow
\text{Gini / SSE}.
$$

Tree construction is greedy, so it does not necessarily produce the globally optimal tree.

Tree complexity can be controlled through:

* Maximum depth
* Minimum node size
* Minimum impurity reduction
* Pre-pruning
* Post-pruning

A deeper tree generally has greater capacity and may have lower training error but higher variance.

Trees are interpretable and can model nonlinear decision boundaries, but individual trees can be unstable. Ensemble methods such as Random Forests and boosting combine multiple trees to improve robustness and predictive performance.
