import pandas as pd
import numpy as np

EXCEL_PATH = "inventory_bom.xlsx"
INVENTORY_SHEET = "Updated Inventory"
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

# --- Inventory ---
inv = inv_df[['SKU', 'Qty on Stock']].dropna()
inv = inv.groupby('SKU', as_index=True)['Qty on Stock'].sum()
inv_map = inv.to_dict()

# --- BOM ---
bom = bom_df[['FG', 'SKU', 'Units in FG']].dropna()
fgs = bom['FG'].unique()
components = bom['SKU'].unique()

# remaining inventory initialized ONLY for components that appear in BOM
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

        feasible = all(remaining.get(c, 0.0) >= float(u) for c, u in req.items())
        if not feasible:
            continue

        total_u = sum(float(u) for u in req.values())
        if total_u <= 0:
            continue

        w = get_weight(fg)
        score = w / total_u

        if score > best_score:
            best_score = score
            best_fg = fg

    if best_fg is None:
        break

    for c, u in usage[best_fg].items():
        remaining[c] -= float(u)
    x[best_fg] += 1

# --- MaxBuild table ---
maxbuild = pd.DataFrame({
    'FG': list(x.keys()),
    'FG Description': [fg_desc_map.get(fg, "") for fg in x.keys()],
    'Weight': [get_weight(fg) for fg in x.keys()],
    'MaxQty_greedy': list(x.values())
}).sort_values('MaxQty_greedy', ascending=False).reset_index(drop=True)

# --- QA calculations ---
# 1) Compute total used per component from the plan
used_map = {c: 0.0 for c in components}
for fg, qty in x.items():
    if qty <= 0:
        continue
    for c, u in usage[fg].items():
        used_map[c] += float(qty) * float(u)

qa = pd.DataFrame({
    "SKU": list(components),
    "InventoryAvailable": [float(inv_map.get(c, 0.0)) for c in components],
    "InventoryUsed": [float(used_map.get(c, 0.0)) for c in components],
})
qa["InventoryRemaining"] = qa["InventoryAvailable"] - qa["InventoryUsed"]
qa["OverConsumed"] = qa["InventoryRemaining"] < -1e-9  # tolerance for float noise

# 2) “Can build one more unit?” check for every FG
build_one_more = []
for fg in fgs:
    req = usage.get(fg, {})
    if not req:
        build_one_more.append(False)
        continue
    # remaining after plan is in dict "remaining"
    ok = all(remaining.get(c, 0.0) >= float(u) for c, u in req.items())
    build_one_more.append(bool(ok))

fg_check = pd.DataFrame({
    "FG": fgs,
    "FG Description": [fg_desc_map.get(fg, "") for fg in fgs],
    "CanBuild_OneMoreUnit": build_one_more
}).sort_values("CanBuild_OneMoreUnit", ascending=False)

# 3) Bottleneck list: lowest remaining SKUs (where remaining is small)
bottlenecks = qa.sort_values("InventoryRemaining", ascending=True).head(30)
# --- Limiting items analysis (why each FG can't build +1) ---
limit_rows = []

for fg in fgs:
    req = usage.get(fg, {})
    if not req:
        continue

    blockers = []
    for c, u in req.items():
        u = float(u)
        if u <= 0:
            continue
        rem = float(remaining.get(c, 0.0))
        shortage = u - rem
        if shortage > 1e-9:
            blockers.append((c, u, rem, shortage))

    # keep only if FG is actually blocked
    blockers.sort(key=lambda x: x[3], reverse=True)  # biggest shortage first
    for rank, (c, u, rem, shortage) in enumerate(blockers[:10], start=1):  # top 10 per FG
        limit_rows.append({
            "FG": fg,
            "FG Description": fg_desc_map.get(fg, ""),
            "Rank": rank,
            "LimitingSKU": c,
            "UnitsNeeded_For+1": u,
            "Remaining": rem,
            "Shortage": shortage
        })

fg_limits = pd.DataFrame(limit_rows)

# --- Aggregate limiting SKUs (across all FGs) ---
if not fg_limits.empty:
    agg_limits = (
        fg_limits.groupby("LimitingSKU", as_index=False)
        .agg(
            FGsBlocked=("FG", "nunique"),
            TotalShortage=("Shortage", "sum"),
            AvgShortage=("Shortage", "mean")
        )
        .merge(
            qa[["SKU", "InventoryRemaining"]],
            left_on="LimitingSKU", right_on="SKU", how="left"
        )
        .drop(columns=["SKU"])
        .sort_values(["FGsBlocked", "TotalShortage"], ascending=[False, False])
        .reset_index(drop=True)
    )
else:
    agg_limits = pd.DataFrame(columns=["LimitingSKU", "FGsBlocked", "TotalShortage", "AvgShortage", "InventoryRemaining"])

# --- Write output workbook with two tabs (+ extra QA sections) ---
print("Saving result with QA tab...")
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    maxbuild.to_excel(writer, sheet_name="MaxBuild", index=False)

    # QA sheet: write multiple sections
    qa.to_excel(writer, sheet_name="QA", index=False, startrow=0)

    # Write summary lines under the table
    summary_row = len(qa) + 3
    pd.DataFrame({
        "QA_Summary": [
            f"Overconsumed SKUs (should be 0): {int(qa['OverConsumed'].sum())}",
            f"FGs that can still build +1 unit (should be 0): {int(fg_check['CanBuild_OneMoreUnit'].sum())}"
        ]
    }).to_excel(writer, sheet_name="QA", index=False, startrow=summary_row)

    # Write FG check section
    fg_check.to_excel(writer, sheet_name="QA", index=False, startrow=summary_row + 5)

    # Write bottleneck section
    # Write bottleneck section
start_bott = summary_row + 5 + len(fg_check) + 3
bottlenecks.to_excel(writer, sheet_name="QA", index=False, startrow=start_bott)

# Write aggregate limiting SKUs section
start_agg = start_bott + len(bottlenecks) + 3
pd.DataFrame({"Section": ["Aggregate limiting SKUs (block most FGs)"]}).to_excel(
    writer, sheet_name="QA", index=False, startrow=start_agg, header=False
)
agg_limits.to_excel(writer, sheet_name="QA", index=False, startrow=start_agg + 1)

# Write per-FG limiting details section
start_fg = start_agg + 1 + len(agg_limits) + 3
pd.DataFrame({"Section": ["Per-FG limiting SKUs (top blockers for +1 unit)"]}).to_excel(
    writer, sheet_name="QA", index=False, startrow=start_fg, header=False
)
fg_limits.to_excel(writer, sheet_name="QA", index=False, startrow=start_fg + 1)

print("Done.")
