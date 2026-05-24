import pandas as pd  #ty: ignore
import seaborn as sns   #ty: ignore
import matplotlib.pyplot as plt   #ty: ignore
import numpy as np   #ty: ignore

sns.set_theme(style="white", palette="bright")

df = pd.read_csv("o.csv")

rounder = lambda x: round(x, 1)  # noqa: E731

colors = sns.color_palette("RdBu", 4)
colors = [tuple(int(c * 255) for c in color) for color in colors]





# load data
df["f(z)"] = df["f(z)"].astype("complex64")

df = df.loc[df["f(z)"].abs() < 10]

# round to cluster points
df["nr_result_y"] = df["nr_result_y"].map(rounder)
df["nr_result_x"] = df["nr_result_x"].map(rounder)

roots_set = set(zip(df["nr_result_x"], df["nr_result_y"]))
mappings = {k: i for i, k in enumerate(roots_set)}
color_mappings = {i: c for i, c in enumerate(colors)}


# image stuff
continius_discrete_point_maping_x = {k: i for i, k in enumerate(sorted(df["zx"].unique()))}
continius_discrete_point_maping_y = {k: i for i, k in enumerate(sorted(df["zy"].unique()))}

df["tag"] = [mappings[p] for p in zip(df["nr_result_x"], df["nr_result_y"])]
df["color"] = df["tag"].map(color_mappings)

df["x"] = df["zx"].map(continius_discrete_point_maping_x)
df["y"] = df["zy"].map(continius_discrete_point_maping_y)



# === Imagen
max_x = df["x"].max()
max_y = df["y"].max()

img = np.zeros((max_y + 1, max_x + 1, 3), dtype=np.uint8)

#print(f"{df['color']=}")
#print(f"{color_mappings=}")

for x, y, c in zip(df["x"], df["y"], df["color"]):
    print(x,y,c)
    img[x - 1,y - 1] = c

plt.imshow(img)
plt.gca().invert_yaxis()
plt.show()



print(df["x"].nunique())
