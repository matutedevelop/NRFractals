import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white", palette="bright")

df = pd.read_csv("o.csv")

rounder = lambda x: round(x, 4)  # noqa: E731

real_x = df["nr_result_x"]
im_y = df["nr_result_y"]

x = map(rounder, df["nr_result_x"])
y = map(rounder, df["nr_result_y"])

df["nr_result_y"] = df["nr_result_y"].map(rounder)
df["nr_result_x"] = df["nr_result_x"].map(rounder)
mappings = {k: i for i, k in enumerate(zip(df["nr_result_x"], df["nr_result_y"]))}
df["tag"] = [mappings[p] for p in zip(df["nr_result_x"], df["nr_result_y"])]
print(df.columns)


sns.scatterplot(x=df["real"].values, y=df["img"].values, hue=df["tag"],palette=['red','blue','green','orange'])
plt.show()
