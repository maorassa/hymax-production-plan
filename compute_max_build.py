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

# --- Inventory (Item family can exist; not used here) ---
inv = inv_df[['SKU', 'Qty on Stock']].dropna()
inv = inv.groupby('SKU', as_index=True)['Qty on Stock'].sum()
inv_map = inv.to_dict()

# --- BOM ---
bom = bom_df[['FG', 'SKU', 'Units in FG']].dropna()
fgs = bom['FG'].unique()
components = bom['SKU'].unique()

# Remaining inventory only for components appearing in BOM
remaining = {c: float(inv_map.get(c, 0.0)) for c in components}

# Usage map: FG -> {component: units}
usage = {}
for fg in fgs:
    sub = bom[bom['FG'] == fg]
    usage[fg] = dict(zip(sub['SKU'], sub['Units in FG']))

# --- Greedy allocation with weights ---
# Chooses the feasible FG with highest (Weight / total_units_in_BOM) each iteration.
x = {fg: 0 for fg in fgs}

print("Running greedy weighted allocation...")
while True:
    best_fg = None
    best_score = -1.0

    for fg in fgs:
        req = usage.get(fg, {})
        if not req:
            continue

        # Feasible to build +1?
        feasible = all(remaining.get(c, 0.0) >= float(u) for c, u in req.items())
        if not feasible:
            continue

        total_u = sum(float(u) for u in req.values())
        if total_u <= 0:
            continue

        score = get_weight(fg) / total_u
        if score > best_score:
            best_score = score
            best_fg = fg

    if best_fg is None:
        break

    # Consume inventory for chosen FG
    for c, u in usage[best_fg].items():
        remaining[c] -= float(u)
    x[best_fg] += 1

# --- MaxBuild output ---
maxbuild = pd.DataFrame({
    'FG': list(x.keys()),
    'FG Description': [fg_desc_map.get(fg, "") for fg in x.keys()],
    'Weight': [get_weight(fg) for fg in x.keys()],
    'MaxQty_greedy': list(x.values())
}).sort_values('MaxQty_greedy', ascending=False).reset_index(drop=True)

# --- QA: component usage ---
used_map = {c: 0.0 for c in components}
for fg, qty in x.items():
    if qty <= 0:
        continue
    for c, u in usage[fg].items():
        used_map[c] += float(qty) * float(u)

qa_components = pd.DataFrame({
    "SKU": list(components),
    "InventoryAvailable": [float(inv_map.get(c, 0.0)) for c in components],
    "InventoryUsed": [float(used_map.get(c, 0.0)) for c in components],
})
qa_components["InventoryRemaining"] = qa_components["InventoryAvailable"] - qa_components["InventoryUsed"]
qa_components["OverConsumed"] = qa_components["InventoryRemaining"] < -1e-9

# --- QA: Can build +1 check per FG ---
can_build_one_more = []
for fg in fgs:
    req = usage.get(fg, {})
    if not req:
        can_build_one_more.append(False)
        continue
    ok = all(remaining.get(c, 0.0) >= float(u) for c, u in req.items())
    can_build_one_more.append(bool(ok))

qa_fg_check = pd.DataFrame({
    "FG": fgs,
    "FG Description": [fg_desc_map.get(fg, "") for fg in fgs],
    "Weight": [get_weight(fg) for fg in fgs],
    "CanBuild_OneMoreUnit": can_build_one_more
}).sort_values(["CanBuild_OneMoreUnit", "Weight"], ascending=[False, False]).reset_index(drop=True)

# --- QA: bottlenecks (lowest remaining SKUs) ---
qa_bottlenecks = qa_components.sort_values("InventoryRemaining", ascending=True).head(30).reset_index(drop=True)

# --- QA: limiting items (per FG blockers for +1) ---
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

    blockers.sort(key=lambda x: x[3], reverse=True)  # biggest shortage first
    for rank, (c, u, rem, shortage) in enumerate(blockers[:10], start=1):
        limit_rows.append({
            "FG": fg,
            "FG Description": fg_desc_map.get(fg, ""),
            "Weight": get_weight(fg),
            "Rank": rank,
            "LimitingSKU": c,
            "UnitsNeeded_For+1": u,
            "Remaining": rem,
            "Shortage": shortage
        })

qa_fg_limits = pd.DataFrame(limit_rows)

# Aggregate limiting SKUs
if not qa_fg_limits.empty:
    qa_agg_limits = (
        qa_fg_limits.groupby("LimitingSKU", as_index=False)
        .agg(
            FGsBlocked=("FG", "nunique"),
            TotalShortage=("Shortage", "sum"),
            AvgShortage=("Shortage", "mean")
        )
        .merge(
            qa_components[["SKU", "InventoryRemaining"]],
            left_on="LimitingSKU", right_on="SKU", how="left"
        )
        .drop(columns=["SKU"])
        .sort_values(["FGsBlocked", "TotalShortage"], ascending=[False, False])
        .reset_index(drop=True)
    )
else:
    qa_agg_limits = pd.DataFrame(columns=["LimitingSKU", "FGsBlocked", "TotalShortage", "AvgShortage", "InventoryRemaining"])

# --- NEW: Procurement list = missing inventory to build +1 unit per FG ---
missing_rows = []
for fg in fgs:
    req = usage.get(fg, {})
    if not req:
        continue

    for c, u in req.items():
        u = float(u)
        if u <= 0:
            continue
        rem = float(remaining.get(c, 0.0))
        shortage = max(0.0, u - rem)
        if shortage > 1e-9:
            missing_rows.append({
                "FG": fg,
                "FG Description": fg_desc_map.get(fg, ""),
                "Weight": get_weight(fg),
                "ComponentSKU": c,
                "UnitsNeeded_For+1": u,
                "Remaining": rem,
                "ShortageToBuild1": shortage
            })

qa_missing_per_fg = pd.DataFrame(missing_rows)

# Aggregate missing components across all FGs
if not qa_missing_per_fg.empty:
    qa_missing_agg = (
        qa_missing_per_fg.groupby("ComponentSKU", as_index=False)
        .agg(
            FGsBlocked=("FG", "nunique"),
            TotalShortageToBuild1=("ShortageToBuild1", "sum"),
            AvgShortage=("ShortageToBuild1", "mean")
        )
        .merge(
            qa_components[["SKU", "InventoryRemaining"]],
            left_on="ComponentSKU", right_on="SKU", how="left"
        )
        .drop(columns=["SKU"])
        .sort_values(["FGsBlocked", "TotalShortageToBuild1"], ascending=[False, False])
        .reset_index(drop=True)
    )
else:
    qa_missing_agg = pd.DataFrame(columns=["ComponentSKU", "FGsBlocked", "TotalShortageToBuild1", "AvgShortage", "InventoryRemaining"])

# Sort per-FG missing list for procurement: high weight first, then biggest shortage
if not qa_missing_per_fg.empty:
    qa_missing_per_fg = qa_missing_per_fg.sort_values(
        ["Weight", "FG", "ShortageToBuild1"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

# --- Write output workbook ---
print("Saving result with QA tab...")
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    maxbuild.to_excel(writer, sheet_name="MaxBuild", index=False)

    row = 0

    # Section 1: component usage
    qa_components.to_excel(writer, sheet_name="QA", index=False, startrow=row)
    row += len(qa_components) + 2

    # Section 2: summary
    summary = pd.DataFrame({
        "QA_Summary": [
            f"Overconsumed SKUs (should be 0): {int(qa_components['OverConsumed'].sum())}",
            f"FGs that can still build +1 unit (should be 0): {int(qa_fg_check['CanBuild_OneMoreUnit'].sum())}",
        ]
    })
    summary.to_excel(writer, sheet_name="QA", index=False, startrow=row)
    row += len(summary) + 2

    # Section 3: can-build +1 check
    qa_fg_check.to_excel(writer, sheet_name="QA", index=False, startrow=row)
    row += len(qa_fg_check) + 2

    # Section 4: bottlenecks
    pd.DataFrame({"Section": ["Bottlenecks (lowest remaining SKUs)"]}).to_excel(
        writer, sheet_name="QA", index=False, startrow=row, header=False
    )
    row += 1
    qa_bottlenecks.to_excel(writer, sheet_name="QA", index=False, startrow=row)
    row += len(qa_bottlenecks) + 2

    # Section 5: aggregate limiting SKUs
    pd.DataFrame({"Section": ["Aggregate limiting SKUs (block most FGs for +1)"]}).to_excel(
        writer, sheet_name="QA", index=False, startrow=row, header=False
    )
    row += 1
    qa_agg_limits.to_excel(writer, sheet_name="QA", index=False, startrow=row)
    row += len(qa_agg_limits) + 2

    # Section 6: per-FG limiting SKUs
    pd.DataFrame({"Section": ["Per-FG limiting SKUs (top blockers for +1 unit)"]}).to_excel(
        writer, sheet_name="QA", index=False, startrow=row, header=False
    )
    row += 1
    qa_fg_limits.to_excel(writer, sheet_name="QA", index=False, startrow=row)
    row += len(qa_fg_limits) + 2

    # Section 7: aggregate meaningfully-missing SKUs (procurement)
    pd.DataFrame({"Section": ["Aggregate missing components (to build +1 across FGs)"]}).to_excel(
        writer, sheet_name="QA", index=False, startrow=row, header=False
    )
    row += 1
    qa_missing_agg.to_excel(writer, sheet_name="QA", index=False, startrow=row)
    row += len(qa_missing_agg) + 2

    # Section 8: per-FG missing components (procurement list)
    pd.DataFrame({"Section": ["Per-FG missing components (to build +1 unit)"]}).to_excel(
        writer, sheet_name="QA", index=False, startrow=row, header=False
    )
    row += 1
    qa_missing_per_fg.to_excel(writer, sheet_name="QA", index=False, startrow=row)

print("Done.")
