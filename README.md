# Closely :triangular_ruler:
 [![PyPI version](https://badge.fury.io/py/closely.svg)](https://badge.fury.io/py/closely) [![Build Status](https://travis-ci.com/justinshenk/closely.svg?branch=master)](https://travis-ci.com/justinshenk/closely)


Find the closest pairs in an array.

### Getting Started

```bash
pip install closely
```

or install from source:
```bash
git clone https://github.com/justinshenk/closely
cd closely
pip install .
```

### How to use

```python
import closely

# X is an n x m numpy array
pairs, distances = closely.solve(X, n=1)

```

You can specify how many pairs you want to identify with `n`.
 

### Example
```python
import closely
import numpy as np
import matplotlib.pyplot as plt

# Create dataset
X = np.random.random((100,2))
pairs, distance = closely.solve(X, n=1)

# Plot points
z, y = np.split(X, 2, axis=1)
fig, ax = plt.subplots()
ax.scatter(z, y) 

for i, txt in enumerate(X): 
    if i in pairs: 
        ax.annotate(i, (z[i], y[i]), color='red') 
    else: 
        ax.annotate(i, (z[i], y[i])) 
```

Check pairs:
```ipython
In [10]: pairs                                                                                                                                
Out[10]: 
array([[[ 7],
        [16]],

       [[96],
        [50]]])

```

Output:
![example_plot](example_plot.png)


### Caveats
`closely` will reduce the dimensionality with PCA of your data to two-dimensions for faster processing.

It also removes the first point in a pair if `n`>1. In rare cases this leads to false negatives if the data is highly overlapping.


### Credit and Explanation

Python code modified from [Andriy Lazorenko](https://medium.com/@andriylazorenko/closest-pair-of-points-in-python-79e2409fc0b2), packaged and made useful for >2 features by Justin Shenk.
