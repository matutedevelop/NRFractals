import numpy as np  # ty: ignore
import cupy as cp  # ty: ignore
from typing import Self  # ty: ignore
import matplotlib.pyplot as plt  # ty : ignore
import seaborn as sns  # ty : ignore


class ComplexPolynomial:
    def __init__(self, coefs: list[float]) -> None:
        self.deg = len(coefs) - 1
        self.coefs = coefs

    def __repr__(self) -> str:
        string_representation = " + "
        gen = (f"{c}x^{i}" for i, c in enumerate(self.coefs) if c != 0)

        return string_representation.join(gen)

    def eval(self, x: np.complex64 | complex) -> np.complex64:
        result = np.complex64(0 + 0j)

        for i, c in enumerate(self.coefs):
            result += c * np.emath.power(x, i)

        return result

    def evalv(self, x: list[cp.complex64]) -> list[cp.complex64]:
        """This is method is gpu dependant"""
        polinomial_terms = cp.array(
            [c * cp.power(x, i) for i, c in enumerate(self.coefs)]
        )
        result = cp.sum(polinomial_terms)

        return result



    def diff(self) -> Self:
        new_coefs = [i * c for i, c in enumerate(self.coefs)][1:]

        return ComplexPolynomial(new_coefs)

    def graph(self) -> None:
        x = np.linspace(-1, 1, 1_000)
        y = self.evalv(x)
        sns.lineplot(x=x, y=y)
        plt.title(f"{repr(self)}")
        plt.grid()
        plt.axhline(y=0)
        plt.show()


