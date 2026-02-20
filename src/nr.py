import numpy as np  # ty: ignore
import cupy as cp  # ty: ignore
from complex_polynomial import ComplexPolynomial  # ty: ignore
import copy  # ty: ignore
import numpy.typing as ntp  # ty: ignore
import cupy.typing as ctp  # ty: ignore


def polynomial_to_solve(poly: ComplexPolynomial, b: float) -> ComplexPolynomial:
    """
    constructs the polynomial we want to find the roots of. i.e. if we are triying to solve f(x) = b this function just returns the polynomial f(x) - b
    """
    new_poly = copy.deepcopy(poly)
    new_poly.coefs[0] -= b
    return new_poly


def nr_vectorized(
    poly: ComplexPolynomial,
    x0: ctp.NDArray[np.complex64],
    epsilon: float = 1e-5,
) -> None:
    f_diff = poly.diff()

    iter_count = 0
    while cp.max(cp.abs(poly.evalv(x0))) > epsilon:
        if iter_count >= 20:
            print("se llego al limite de iteraciones")
            return x0

        if cp.min(cp.abs(f_diff.evalv(x0))) == 0:
            error_message = f"{x0} is an Escape point"
            raise NewthonRaphsonEscapePoint(error_message)

        x0 = x0 - poly.evalv(x0) / f_diff.evalv(x0)
        iter_count += 1
        print("======")
        print(cp.max(cp.abs(poly.evalv(x0))))
        print(cp.max(poly.evalv(x0)))
        print("======")
        print(iter_count)

    return x0




class NewthonRaphsonEscapePoint(Exception):
    """
    This exception is thrown by nr_vectorized() when it encounters falls in an escape point within the max iterations provided. i.e. f'(xi) = 0
    """

    pass


# if __name__ == "__main__":
#     # x = cp.linspace(-1, 1, 4)
#     # y = (cp.linspace(-1, 1, 4) * 1j).astype("complex64")
#     # z = (x + y).astype("complex64")
#
#     z_vec = cp.array([12 + 33j]).astype("complex64")
#     z = np.complex64(12 + 3j)
#
#     pol = ComplexPolynomial([21, 121, 441, 21])
#
#     result = nr_vectorized(pol, z_vec)
#     #result_normal = nr(pol, z)
#
#     print(result)
#     # print(result_normal)
#
#     print("----------")
#     # print(pol.eval(result_normal))
#     print(cp.abs(pol.evalv(result)))
