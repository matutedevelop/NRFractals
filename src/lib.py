import numpy as np  # ty: ignore
from typing import Self  # ty: ignore
import matplotlib.pyplot as plt  # ty : ignore
import seaborn as sns  # ty : ignore
from numpy import typing as ntp
import pandas as pd


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

        result = np.zeros(x.shape[0], dtype="complex64")

        for i, c in enumerate(self.coefs):
            result += c * np.emath.power(x, i)

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


# =====================
# =====================
# =====================
# =====================
# =====================


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

    f = poly

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
    *,
    x0: ntp.NDArray[np.complex128],
    epsilon: float = 1e-6,
):

    f_image = poly.evalv(x0)

    total_escape_point_mask = np.array([False] * x0.shape[0])

    iter_count = 0
    while np.percentile(np.abs(f_image), 99) > epsilon:
        f_diff = poly.diff().evalv(x0)
        escape_ponint_mask = (f_diff == 0) | (np.abs(f_diff) < epsilon)

        if iter_count >= 200:
            print("se llego al limite de iteraciones")
            break

        if np.any(escape_ponint_mask):
            print("this hitup")
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            total_escape_point_mask = np.logical_or(
                total_escape_point_mask, escape_ponint_mask
            )
            f_diff[escape_ponint_mask] = 1e-5
            # error_message = f"{x0} is an Escape point"
            # raise NewthonRaphsonEscapePoint(error_message)

        x0 = x0 - f_image / f_diff

        f_image = poly.evalv(x0)

        iter_count += 1

        print(f"iteration: {iter_count}")

    f_image = poly.evalv(x0)
    converged_mask = np.abs(f_image) < epsilon
    # x0 = x0[converged_mask]
    # total_escape_point_mask = total_escape_point_mask[converged_mask]

    print(f"ended with {iter_count} iteration")

    return {
        "idxs": converged_mask,
        "result": x0,
        "escape_point_mask": total_escape_point_mask,
    }


class NewthonRaphsonEscapePoint(Exception):
    """
    This exception is thrown by nr() when it  falls in an escape point within the max iterations provided. i.e. f'(xi) = 0
    """

    pass


# =====================
# =====================
# =====================
# =====================
# =====================
# =====================


def nr_data(polynomial: ComplexPolynomial, **kwargs) -> pd.DataFrame:

    x_max = kwargs.get("x_max", 100)
    x_min = kwargs.get("x_min", -100)
    Δx = int(5 * abs(x_max - x_min))
    y_max = kwargs.get("y_max", 100)
    y_min = kwargs.get("y_min", -100)
    Δy = int(5 * abs(y_max - y_min))

    x = np.linspace(x_min, x_max, Δx)
    y = np.linspace(y_min, y_max, Δy)

    px, py = np.meshgrid(x, y)
    points = px + py * 1j
    points = points.flatten()

    print("grid initialized")

    newthon_raphson_result = nrv(polynomial, x0=points)
    z_result = newthon_raphson_result["result"]
    escape_point_mask = newthon_raphson_result["escape_point_mask"]

    df = pd.DataFrame(
        {
            "z": points,
            "zx": points.real,
            "zy": points.imag,
        }
    )
    df["nr_result_x"] = 0
    df["nr_result_y"] = 0
    df["is_escape"] = escape_point_mask
    df["is_escape"] = 0

    df["nr_result_x"] = z_result.real
    df["nr_result_y"] = z_result.imag
    df["f(z)"] = polynomial.evalv(z_result)

    return df


def mutate_df(df: pd.DataFrame, n_roots: int) -> pd.DataFrame:

    rounder = lambda x: round(x, 1)  # noqa: E731
    colors = sns.color_palette("bone", n_roots)
    colors = [tuple(int(c * 255) for c in color) for color in colors]

    # load data

    df["f(z)"] = df["f(z)"].astype("complex64")

    # round to cluster points
    df["nr_result_y"] = df["nr_result_y"].map(rounder)
    df["nr_result_x"] = df["nr_result_x"].map(rounder)

    roots_set = set(zip(df["nr_result_x"], df["nr_result_y"]))
    mappings = {k: i for i, k in enumerate(roots_set)}
    color_mappings = {i: c for i, c in enumerate(colors)}

    # image stuff
    continius_discrete_point_maping_x = {
        k: i for i, k in enumerate(sorted(df["zx"].unique()))
    }
    continius_discrete_point_maping_y = {
        k: i for i, k in enumerate(sorted(df["zy"].unique()))
    }

    df["tag"] = [mappings[p] for p in zip(df["nr_result_x"], df["nr_result_y"])]
    df["color"] = df["tag"].map(color_mappings)

    df["x"] = df["zx"].map(continius_discrete_point_maping_x)
    df["y"] = df["zy"].map(continius_discrete_point_maping_y)

    print(color_mappings)
    # print(df["color"].value_counts())
    # print(df["tag"].value_counts())
    print(len(roots_set))

    return df


def plot_image(df: pd.DataFrame) -> None:
    # === Imagen
    max_x = df["x"].max()
    max_y = df["y"].max()

    img = np.zeros((max_y + 1, max_x + 1, 3), dtype=np.uint8)

    for x, y, c in zip(df["x"], df["y"], df["color"]):
        #print(x, y, c)
        img[x - 1, y - 1] = c

    plt.imshow(img)
    plt.gca().invert_yaxis()
    plt.show()
