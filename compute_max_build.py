import pandas as pd

print("Reading headers from 'Dec 8 Inventory'...")
inv_df = pd.read_excel("inventory_bom.xlsx", sheet_name="Dec 8 Inventory", header=0)
print(inv_df.columns.tolist())

print("Reading headers from 'Bill of Material'...")
bom_df = pd.read_excel("inventory_bom.xlsx", sheet_name="Bill of Material", header=0)
print(bom_df.columns.tolist())
