import pandas as pd
import numpy as np

EXCEL_PATH = "inventory_bom.xlsx"
INVENTORY_SHEET = "Updated Inventory"   # <- new sheet name
BOM_SHEET = "Bill of Material"
OUTPUT_PATH = "FG_max_build_plan_greedy.xlsx"

print("Loading data...")
inv_df = pd.read_excel(EXCEL_PATH, sheet_name=INVENTORY_SHEET, header=0)
bom_df = pd.read_excel(EXCEL_PATH, sheet_name=BOM_SHEET, header=0)

# --- FG description map ---
fg_desc_df = bom_df[['FG', 'FG Description']].dropna().drop_duplicates(subset='FG')
fg_desc_map = dict(zip(fg_desc_df['FG'], fg_desc_df['FG Description']))

# --- Optional weights sheet ---
try:
    weights_df = pd.read_excel(EXCEL_PATH, sheet_name="FG Weights", header=0)
    weights_map = dict(zip(weights_df['FG'], weights_df['Weight']))
except Exception:
    weights_map = {}

def get_weight(fg):
    w = weights_map.get(fg, 1.0)
    try:
        return float(w)
    except Exception:
        return 1.0

# --- Inventory: SKU + Qty on Stock ---
# (Item family column exists but is not used yet)
inv = inv_df[['SKU', 'Qty on Stock']].dropna()
inv = inv.groupby('SKU', as_index=True)['Qty on Stock'].sum()
inv_map = inv.to_dict()

# --- BOM: FG + SKU + Units in FG ---
bom = bom_df[['FG', 'SKU', 'Units in FG']].dropna()
fgs = bom['FG'].unique()
components = bom['SKU'].unique()

remaining = {c: float(inv_map.get(c, 0.0)) for c in components}

# FG -> {component: units}
usage = {}
for fg in fgs:
    sub = bom[bom['FG'] == fg]
    usage[fg] = dict(zip(sub['SKU'], sub['Units in FG']))

# --- Greedy allocation with weights ---
x = {fg: 0 for fg in fgs}

print("Running greedy weighted allocation...")
while True:
    best_fg = None
    best_score = -1.0

    for fg in fgs:
        req = usage[fg]
        if not req:
            continue

        # Check if we can build 1 more unit of this FG
        feasible = all(remaining.get(c, 0.0) >= float(u) for c, u in req.items())
        if not feasible:
            continue

        total_u = sum(float(u) for u in req.values())
        if total_u <= 0:
            continue

        w = get_weight(fg)
        score = w / total_u  # priority per total component usage

        if score > best_score:
            best_score = score
            best_fg = fg

    if best_fg is None:
        break  # nothing more can be built

    # Consume components for chosen FG
    for c, u in usage[best_fg].items():
        remaining[c] -= float(u)
    x[best_fg] += 1

# --- Build result table ---
res = pd.DataFrame({
    'FG': list(x.keys()),
    'MaxQty_greedy': list(x.values())
})

res['FG Description'] = res['FG'].map(fg_desc_map)
res['Weight'] = res['FG'].apply(get_weight)

# Reorder columns and sort
res = res[['FG', 'FG Description', 'Weight', 'MaxQty_greedy']] \
       .sort_values('MaxQty_greedy', ascending=False) \
       .reset_index(drop=True)

print("Saving result...")
res.to_excel(OUTPUT_PATH, sheet_name="MaxBuild", index=False)
print("Done.")
