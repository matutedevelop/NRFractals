import numpy as np  # ty: ignore
import numba  # ty: ignore
from complex_polynomial import ComplexPolynomial  # ty: ignore
from numpy import complex64, typing as ntp
import copy


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


def nrv(
    poly: ComplexPolynomial,
    b: float,
    *,
    x0: ntp.NDArray[np.complex128],
    epsilon: float = 1e-6,
):

    f = polynomial_to_solve(poly, b)
    
    f_image = f.evalv(x0)
    
    iter_count = 0
    while np.percentile(np.abs(f_image.real),95) > epsilon:

        
        f_diff = f.diff().evalv(x0)

        if iter_count >= 20:
            print("se llego al limite de iteraciones")
            break

        # if np.min(np.abs(f_diff.real)) == 0:
        #     error_message = f"{x0} is an Escape point"
        #     raise NewthonRaphsonEscapePoint(error_message)

        x0 = x0 - f_image / f_diff

        iter_count += 1
    
        print(f"iteration: {iter_count}")
    print(iter_count)
    return x0


class NewthonRaphsonEscapePoint(Exception):
    """
    This exception is thrown by nr() when it  falls in an escape point within the max iterations provided. i.e. f'(xi) = 0
    """

    pass


# Temporary
if __name__ == "__main__":
    
    import seaborn as sns
    from matplotlib import pyplot as plt

    pol = ComplexPolynomial([-1, 0, 0, 1])


    x = np.linspace(-100,100,201)
    px, py = np.meshgrid(x,x)

    # points = []
    points = px + py * 1j
    points = points.flatten()


    r = nrv(pol,0,x0=points)
    

    sns.scatterplot(x=r.real,y=r.imag)
    plt.show()

    sns.displot(x=r.real)
    plt.show()

    sns.displot(x=r.imag)
    plt.show()
