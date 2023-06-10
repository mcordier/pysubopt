# Opt-Submodular

This package contains submodular greedy algorithms for optimization on sets for knapstack problems. To work correctly, the specified functions to maximize needs to be submodular.


## Install

See the full setup process on the mono-repo instructions.

## Quickstart

A first example to illustrate the package:
```python
from opt_submodular import GreedySubmodularOptimizer

def submod_func_obj(S: List[int]):
	res = ...
	return res

def func_cost(S: List[int]):
	res = ...
	return res

optimizer = GreedySubmodularOptimizer(
    fun=func_obj,
    cost_fun=func_cost,
    budget=10,
    r=1,
    is_lazy=True
)

resultat = optimizer.run()

print(resultat)
```
