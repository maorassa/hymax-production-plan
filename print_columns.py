import pandas as pd

df = pd.read_excel("inventory_bom.xlsx", sheet_name="Dec 8 Inventory", header=0)
print(df.columns.tolist())
