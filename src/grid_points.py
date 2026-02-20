import numpy as np  # ty: ignore
import cupy as cp  # ty: ignore
import pandas as pd  # ty: ignore
from complex_polynomial import ComplexPolynomial  # ty: ignore
from nr import nr_vectorized  # ty: ignore
import time  # ty: ignore

import seaborn as sns
import matplotlib.pyplot as plt

start = time.perf_counter()

pol = ComplexPolynomial(
    [
        3211,
        1,
        13,
        1,
        7,
    ]
)



xs = cp.linspace(-10, 10, 10_004, dtype=cp.float32)
ys = cp.linspace(-10, 10, 10_001, dtype=cp.float32)

z = (xs[None, :] + 1j * ys[:, None]).astype(cp.complex64).ravel()
#print(z.shape)



df = pd.DataFrame()

nr_result = nr_vectorized(pol,z)
image_result = pol.evalv(nr_result)



