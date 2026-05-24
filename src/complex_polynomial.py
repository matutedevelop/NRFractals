import numpy as np  # ty: ignore
from typing import  Self  # ty: ignore
import matplotlib.pyplot as plt  #ty : ignore
import seaborn as sns #ty : ignore
from numpy import typing as ntp


class ComplexPolynomial:
    def __init__(self, coefs: list[float]) -> None:
        self.deg = len(coefs) - 1
        self.coefs = coefs

    def __repr__(self) -> str:
        string_representation = " + "
        gen = (f"{c}x^{i}" for i, c in enumerate(self.coefs) if c != 0)

        return string_representation.join(gen)

    def eval(self, x: np.complex128 | complex) -> np.complex128:
        result = np.complex128(0 + 0j)

        for i, c in enumerate(self.coefs):
            result += c * np.emath.power(x, i)

        return result

    def evalv(self, x: ntp.NDArray[np.complex64]) -> list[np.complex64]:
        # vectorized_eval = np.vectorize(self.eval)
        # return vectorized_eval(x)


        result = np.zeros(x.shape[0],dtype="complex64")

        for i, c in enumerate(self.coefs):
            result += c * np.emath.power(x, i)

        return result




    def diff(self) -> Self:
        new_coefs = [i * c for i, c in enumerate(self.coefs)][1:]

        return ComplexPolynomial(new_coefs)

    def graph(self) -> None:
        x = np.linspace(-1, 1, 1_000)
        y = self.evalv(x)
        sns.lineplot(x=x,y=y)
        plt.title(f"{repr(self)}")
        plt.grid()
        plt.axhline(y=0)
        plt.show()




