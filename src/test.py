import pandas as pd  #ty: ignore
import seaborn as sns   #ty: ignore
import matplotlib.pyplot as plt   #ty: ignore
import numpy as np   #ty: ignore

sns.set_theme(style="white", palette="bright")

df = pd.read_csv("o.csv")

rounder = lambda x: round(x, 4)  # noqa: E731

colors = sns.color_palette("RdBu", 4)
colors = [tuple(int(c * 255) for c in color) for color in colors]




df["nr_result_y"] = df["nr_result_y"].map(rounder)
df["nr_result_x"] = df["nr_result_x"].map(rounder)

roots_set = set(zip(df["nr_result_x"], df["nr_result_y"]))
mappings = {k: i for i, k in enumerate(roots_set)}
color_mappings = {i: c for i, c in enumerate(colors)}
continius_discrete_point_maping_x = {k: i for i, k in enumerate(sorted(df["real"].unique()))}
continius_discrete_point_maping_y = {k: i for i, k in enumerate(sorted(df["img"].unique()))}

df["tag"] = [mappings[p] for p in zip(df["nr_result_x"], df["nr_result_y"])]
df["color"] = df["tag"].map(color_mappings)

df["x"] = df["real"].map(continius_discrete_point_maping_x)
df["y"] = df["img"].map(continius_discrete_point_maping_y)

# === Imagen
max_x = df["x"].max()
max_y = df["y"].max()

img = np.zeros((max_y + 1, max_x + 1, 3), dtype=np.uint8)


for x, y, c in zip(df["x"], df["y"], df["color"]):
    img[x,y] = c

plt.imshow(img)
plt.gca().invert_yaxis()
plt.show()
# sns.scatterplot(
#     x=df["real"].values,
#     y=df["img"].values,
#     hue=df["tag"],
#     palette=["red", "blue", "green", "orange"],
# )
# plt.show()

print(df['x'])

df.to_csv("o1.csv")

print(df["x"].nunique())
