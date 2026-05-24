from numba import complex128
import numpy as np
import pandas as pd
from complex_polynomial import ComplexPolynomial
from nr import nrv

pol = ComplexPolynomial(
    [
        3211,
        1,
        13,
        1,
        7,
    ]
)


x = np.linspace(-100, 100, 401)
px, py = np.meshgrid(x, x)

# points = []
points = px + py * 1j


points = points.flatten()

print("grid inicializada")

newthon_raphson_result = nrv(pol, 8, x0=points)
z_result = newthon_raphson_result["result"]
escape_point_mask = newthon_raphson_result["escape_point_mask"]
idxs = newthon_raphson_result["idxs"]


df = pd.DataFrame(
    {
        "z": points[idxs],
        "zx": points[idxs].real,
        "zy": points[idxs].imag,
    }
)
df["nr_result_x"] = 0
df["nr_result_y"] = 0
df["is_escape"] = False


df["nr_result_x"] = z_result.real
df["nr_result_y"] = z_result.imag
df["f(z)"] = pol.evalv(z_result)

df.to_csv("o.csv")
