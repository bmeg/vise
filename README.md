

Find all classifiers in SciKit Learn
```
from sklearn.base import ClassifierMixin
from sklearn.utils.testing import all_estimators
classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
print(classifiers)
```
