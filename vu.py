import pandas as pd

# Charger ton dataset
df = pd.read_csv("./dataset/channel_5.csv")

print(df["Éclairage"].unique())

# Compter combien de fois chaque valeur apparaît
print(df["Éclairage"].value_counts())
# ,