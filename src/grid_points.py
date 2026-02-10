from numba import complex128
import numpy as np
import pandas as pd
from complex_polynomial import ComplexPolynomial
from nr import nr

pol = ComplexPolynomial([3211,1,13,1,7,])

grid = [[x, y] for x in np.linspace(-3, 3, 100) for y in np.linspace(-3,3,100)]
print('grid inicializada')
#grid = np.zeros(10_000,10_000)

# for i,x in enumerate(np.linspace(-3,3,10_000)):
#     for j,y in enumerate(np.linspace(-3,3,10_000)):
#         grid[i,j] = [x,y]

df = pd.DataFrame(grid, columns=["real", "img"])
df['nr_result_x'] = 0
df['nr_result_y'] = 0

for i,r in enumerate(df.iterrows()):
    
    z = complex128(complex(real=r[1][0],imag=r[1][1]))
    z_result = nr(pol,0,x0=z)
    df.iloc[i,2] = z_result.real
    df.iloc[i,3] = z_result.imag
    print("===========",i)

df.to_csv('o.csv')
