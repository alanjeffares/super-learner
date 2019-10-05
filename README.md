<div align='center'>
# Super Learner
</div>

My implementation of the stacked ensemble super learner as described in Mark J. van der Laan (2007).

<p align="center">
<img src="https://github.com/alanjeffares/super_learner/blob/master/fig1.png"  width="500">
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
