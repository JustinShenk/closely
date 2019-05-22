# Closest Pairs

Find the closest pair in a dataset.

## Getting Started

```shell
pip install closest_pairs
```

## How to use

```python
import closest_pairs

pairs, distances = closest_pairs.solve(X, n=1)

```

You can specify how many pairs you want to identify with `n`.

## Example
```python
import closest_pairs
import numpy as np
import matplotlib.pyplot as plt

X = np.random.random((100,2))
pairs, distance = closest_pairs.solve(X, n=1)

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

## Credit

Python code modified from [Andriy Lazorenko](https://medium.com/@andriylazorenko/closest-pair-of-points-in-python-79e2409fc0b2), packaged by Justin Shenk.