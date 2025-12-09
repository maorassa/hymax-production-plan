import pandas as pd

EXCEL_PATH = "inventory_bom.xlsx"
INVENTORY_SHEET = "Dec 8 Inventory"
BOM_SHEET = "Bill of Material"
OUTPUT_PATH = "FG_max_build_plan_greedy.xlsx"

print("Loading data...")
inv_df = pd.read_excel(EXCEL_PATH, sheet_name=INVENTORY_SHEET, header=0)
bom_df = pd.read_excel(EXCEL_PATH, sheet_name=BOM_SHEET, header=0)

# Inventory: SKU + Qty on Stock
inv = inv_df[['SKU', 'Qty on Stock']].dropna()
inv = inv.groupby('SKU', as_index=True)['Qty on Stock'].sum()

# BOM: FG + SKU + Units in FG
bom = bom_df[['FG', 'SKU', 'Units in FG']].dropna()

fgs = bom['FG'].unique()
components = bom['SKU'].unique()

inv_map = inv.to_dict()
remaining = {c: float(inv_map.get(c, 0.0)) for c in components}

# FG -> {component: units}
usage = {}
for fg in fgs:
    sub = bom[bom['FG'] == fg]
    usage[fg] = dict(zip(sub['SKU'], sub['Units in FG']))

x = {fg: 0 for fg in fgs}

print("Running greedy allocation...")
while True:
    feasible = []
    for fg in fgs:
        req = usage[fg]
        if all(remaining.get(c, 0.0) >= float(u) for c, u in req.items()):
            feasible.append(fg)

    if not feasible:
        break

    def total_req(fg_):
        return sum(float(u) for u in usage[fg_].values())

    choice = min(feasible, key=total_req)

    for c, u in usage[choice].items():
        remaining[c] -= float(u)
    x[choice] += 1

res = pd.DataFrame({
    'FG': list(x.keys()),
    'MaxQty_greedy': list(x.values())
}).sort_values('MaxQty_greedy', ascending=False)

print("Saving result...")
res.to_excel(OUTPUT_PATH, sheet_name="MaxBuild", index=False)
print("Done.")
