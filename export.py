import runpy
import csv
import os

# Folder where all files will go
OUTPUT_DIR = "outputs"

# Create folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ----------------------------
# CONFIG
# ----------------------------
INPUT_SCRIPT = "TEX_Hydrostatics.py"      # <-- change if your file name differs
EXCEL_OUT = os.path.join(OUTPUT_DIR, "TEX_output.xlsx")
WRITE_CSVS   = True


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    # IMPORTANT: run_name="__main__" makes the target script run its
    # if __name__ == "__main__": block
    g = runpy.run_path(INPUT_SCRIPT, run_name="__main__")

    required = ["items", "cg", "out", "KG", "GMt_ft", "GMl_ft", "W_table"]
    missing = [k for k in required if k not in g]
    if missing:
        raise RuntimeError(
            "Couldn't find expected variables in your input script.\n"
            f"Missing: {missing}\n\n"
            "Your input file ran, but those variables still weren't created.\n"
            "Make sure your script sets them with EXACT names inside the main block:\n"
            "items, cg, out, KG, GMt_ft, GMl_ft, W_table"
        )

    items   = g["items"]
    cg      = g["cg"]
    out     = g["out"]
    KG      = float(g["KG"])
    GMt_ft  = float(g["GMt_ft"])
    GMl_ft  = float(g["GMl_ft"])
    W_table = float(g["W_table"])

    # ----------------------------
    # Build tables
    # ----------------------------
    weights_rows = []
    for it in items:
        weights_rows.append({
            "Item": it.name,
            "Note": getattr(it, "note", ""),
            "W_lb": float(it.w),
            "KG_ft": float(it.kg),
            "VM_ft_lb": float(it.w * it.kg),
            "LCG_ft": float(it.lcg),
            "LM_ft_lb": float(it.w * it.lcg),
            "TCG_ft": float(getattr(it, "tcg", 0.0)),
            "TM_ft_lb": float(it.w * getattr(it, "tcg", 0.0)),
        })

    cg_summary_rows = [{
        "W_total_lb": float(cg["W_total"]),
        "Mz_total_ft_lb": float(cg["Mz_total"]),
        "Mx_total_ft_lb": float(cg["Mx_total"]),
        "My_total_ft_lb": float(cg["My_total"]),
        "KG_ft": float(cg["KG"]),
        "LCG_ft": float(cg["LCG"]),
        "TCG_ft": float(cg["TCG"]),
    }]

    hydro_row = dict(out)
    hydro_row.update({
        "KG_ft": KG,
        "GMt_ft": GMt_ft,
        "GMl_ft": GMl_ft,
    })
    hydro_rows = [hydro_row]

    stab_rows = [{
        "W_total_table_lb": W_table,
        "Wdisp_hydro_lb": float(out["W_disp_lb"]),
        "Delta_hydro_minus_table_lb": float(out["W_disp_lb"] - W_table),
    }]

    # ----------------------------
    # Write Excel
    # ----------------------------
    if HAS_PANDAS:
        with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
            pd.DataFrame(weights_rows).to_excel(writer, index=False, sheet_name="Weights")
            pd.DataFrame(cg_summary_rows).to_excel(writer, index=False, sheet_name="CG_Summary")
            pd.DataFrame(hydro_rows).to_excel(writer, index=False, sheet_name="Hydro")
            pd.DataFrame(stab_rows).to_excel(writer, index=False, sheet_name="Stability_Check")
        print(f"Wrote Excel: {os.path.abspath(EXCEL_OUT)}")
    else:
        print("pandas not installed -> skipping Excel output (CSV still works).")

    # ----------------------------
    # Optional CSVs
    # ----------------------------
    if WRITE_CSVS:
        write_csv(os.path.join(OUTPUT_DIR, "weights.csv"), weights_rows)
        write_csv(os.path.join(OUTPUT_DIR, "cg_summary.csv"), cg_summary_rows)
        write_csv(os.path.join(OUTPUT_DIR, "hydro.csv"), hydro_rows)
        write_csv(os.path.join(OUTPUT_DIR, "stability_check.csv"), stab_rows)

        print("Wrote CSVs:")
        print(f"  {os.path.abspath('weights.csv')}")
        print(f"  {os.path.abspath('cg_summary.csv')}")
        print(f"  {os.path.abspath('hydro.csv')}")
        print(f"  {os.path.abspath('stability_check.csv')}")


if __name__ == "__main__":
    main()