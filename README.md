<div align='center'>
  
# Super Learner Classifier
</div>

My implementation of the stacked ensemble Super Learner as described in Mark J. van der Laan et al, (2007). The Super Learner is a heterogeneous stacked ensemble classifier. This is a classification model that uses a set of base classifiers of different types, the outputs of which are then combined in another classifier at the stacked layer.

<p align="center">
<img src="https://github.com/alanjeffares/super_learner/blob/master/fig1.png"  width="700">
</p>

## File Descriptions
* `superlearner.py` contains my implementation of the classifier.
* `something.ipynb` is a jupyter notebook demonstrating an example usage on the fashion MNIST dataset.

## Usage
Navigate to the repository folder and simply run `from superlearner import SuperLearnerClassifier` in python.

For example on the Iris dataset:
```
from superlearner import SuperLearnerClassifier
from sklearn.datasets import load_iris

iris = load_iris()
sl_model = SuperLearnerClassifier(use_stacked_prob=False)
sl_model.fit(pd.DataFrame(iris.data), iris.target)
sl_model.predict(pd.DataFrame(iris.data))
```
