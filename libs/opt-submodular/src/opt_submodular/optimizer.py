import abc
import time
from typing import Callable, Generic, List, TypeVar

import numpy as np

from .types import Result

T = TypeVar("T")


class SubmodularOptimizer(Generic[T]):
    def __init__(
        self,
        fun: Callable[List[T], float],
        cost_fun: Callable[List[T], float],
        budget: float,
    ):
        self.fun = fun
        self.cost_fun = cost_fun
        self.budget = budget

    def run(self, full_set: T) -> Result:
        t_start = time.time()
        opt_subset = self._get_optimal_subset(full_set)
        return Result(
            opt_subset=opt_subset,
            fun_value=self.fun(opt_subset),
            cost_fun_value=self.cost_fun(opt_subset),
            delta_time=time.time() - t_start,
        )

    @abc.abstractmethod
    def _get_optimal_subset(self, full_set: List[T]):
        return None


class GreedySubmodularOptimizer(SubmodularOptimizer[T]):
    def __init__(
        self,
        fun: Callable[List[T], float],
        cost_fun: Callable[List[T], float],
        budget: float,
        r: float = 1.0,
        is_lazy: bool = False,
    ):
        self.fun = fun
        self.cost_fun = cost_fun
        self.budget = budget
        self.r = r
        self.is_lazy = is_lazy

    def _get_optimal_subset(self, full_set: List[T]) -> Result:
        G = []
        np.min([self.cost_fun([u]) for u in full_set])

        cost = 0

        if self.is_lazy:
            deltas = [
                self.fun([u]) / (self.cost_fun([u])) ** self.r
                for u in full_set
            ]

        # stop when the cost is 90% of budget
        while len(full_set) != 0 and self.budget - cost >= 0.1 * self.budget:
            if self.is_lazy:
                max_index = np.argmax(deltas)
                delta = (
                    self.fun(G + [full_set[max_index]]) - self.fun(G)
                ) / self.cost_fun([full_set[max_index]]) ** self.r
                deltas[max_index] = delta
                idx = []
                while (max_index not in idx) and (delta < np.amax(deltas)):
                    idx.append(max_index)
                    max_index = np.argmax(deltas)
                    delta = (
                        self.fun(G + [full_set[max_index]]) - self.fun(G)
                    ) / (self.cost_fun([full_set[max_index]]) ** self.r)
                    deltas[max_index] = delta
                k = full_set[max_index]
                deltas = deltas[:max_index] + deltas[max_index + 1 :]

            else:
                L = [
                    (self.fun(G + [u]) - self.fun(G))
                    / (self.cost_fun([u])) ** self.r
                    for u in full_set
                ]
                k = full_set[np.array(L).argmax()]

            cur_cost = self.cost_fun(G + [k])

            if cur_cost <= self.budget:  # and f(G + [k]) - f(G) >= 0:
                G += [k]
                cost = cur_cost

            full_set.remove(k)

        L = [
            self.fun([u]) for u in full_set if self.cost_fun([u]) < self.budget
        ]
        v = np.array(L).argmax()

        if self.fun(G) > self.fun([v]):
            res = G
        else:
            res = v

        return res


class DoubleGreedySubmodularOptimizer(SubmodularOptimizer[T]):
    def _get_optimal_subset(self, full_set: List[T]):
        X0 = []
        X1 = full_set[:]
        for e in full_set:
            improve1 = self.fun(X0 + [e]) - self.fun(X0)
            X1_without_e = X1[:]
            X1_without_e.remove(e)
            improve2 = self.fun(X1_without_e) - self.fun(X1)
            if improve1 >= improve2 and self.cost_fun(X0 + [e]) <= self.budget:
                X0 += [e]
            else:
                X1.remove(e)
        return X0


class RandomSubset(SubmodularOptimizer[T]):
    def _get_optimal_subset(self, full_set: List[T]):
        U = full_set.copy()
        G = []
        cost = 0
        while len(U) != 0 and self.budget - cost >= 0.1 * self.budget:
            k = np.random.choice(U)
            U.remove(k)
            cur_cost = self.cost_fun(G + [k])
            if cur_cost <= self.budget:  # and fun(G + [k])- fun(G) >= 0:
                G += [k]
                cost = cur_cost
        return G
