import numpy as np  # ty: ignore
import numba  # ty: ignore
from complex_polynomial import ComplexPolynomial  # ty: ignore
import copy

# temporary
import seaborn as sns
import matplotlib.pyplot as plt


def polynomial_to_solve(poly: ComplexPolynomial, b: float) -> ComplexPolynomial:
    """
    constructs the polynomial we want to find the roots of. i.e. if we are triying to solve f(x) = b this function just returns the polynomial f(x) - b
    """
    new_poly = copy.deepcopy(poly)
    new_poly.coefs[0] -= b
    return new_poly


def nr(
    poly: ComplexPolynomial,
    b: float,
    *,
    x0: np.complex128 = None,
    epsilon: float = 1e-8,
) -> None:
    if x0 is None:
        re = float(np.random.uniform(-500, 500))
        im = float(np.random.uniform(-500, 500))
        complex_number = complex(real=re, imag=im)
        x0 = np.complex128(complex_number)

    f = polynomial_to_solve(poly, b)

    iter_count = 0
    while abs(f.eval(x0)).real > epsilon:

        f_diff = f.diff().eval(x0)



        if iter_count >= 1_000:
            print("se llego al limite de iteraciones")
            break


        if f_diff == 0:
            error_message = f"{x0} is an Escape point"
            raise NewthonRaphsonEscapePoint(error_message)

        x0 = x0 - f.eval(x0) / f.diff().eval(x0)

        iter_count += 1
    print(iter_count)
    return x0


class NewthonRaphsonEscapePoint(Exception):
    """
    This exception is thrown by nr() when it encounters falls in an escape point within the max iterations provided. i.e. f'(xi) = 0
    """

    pass


# Temporary
if __name__ == '__main__':

    pol = ComplexPolynomial([-1, 0, 0, 1])

    points = []

    for _ in range(100):
        r = np.round(nr(pol, 0), 8)
        print(r)
        points.append(r)

        x = map(lambda x: x.real,points)
        y = map(lambda x: x.imag,points)

    sns.scatterplot(x=x,y=y)
    plt.show()
    pol.graph()

    # pol = ComplexPolynomial([-1, 0, 0, 1])
    # print(nr(pol, 0, x0=-(0.5)**(1/3)))
