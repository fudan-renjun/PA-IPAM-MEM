#!/usr/bin/env python3
"""Export per-figure Excel data folders for web/manual figure drawing."""

from __future__ import annotations

from pathlib import Path
import re
import shutil

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATE = "2026-06-05"

PLOTTING_PACKAGE = PROJECT / "results" / "12_plotting_data_package"
MANIFEST = PLOTTING_PACKAGE / f"plotting_manifest_{DATE}.csv"
OUT = PROJECT / "results" / "13_figure_excel_data_package"


def safe_name(value: str, max_len: int = 80) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return clean[:max_len] or "data"


def safe_sheet_name(value: str, used: set[str]) -> str:
    base = safe_name(value, max_len=28)
    name = base[:31]
    i = 2
    while name in used:
        suffix = f"_{i}"
        name = f"{base[:31 - len(suffix)]}{suffix}"
        i += 1
    used.add(name)
    return name


def folder_for_row(row: pd.Series) -> Path:
    figure = str(row["figure"])
    destination = str(row["destination"])
    if destination == "main":
        folder = f"Main_{safe_name(figure)}"
    elif destination == "supplementary":
        folder = safe_name(figure).replace("Supplementary_Figure", "Supplementary_Figure")
    else:
        folder = safe_name(figure)
    return OUT / folder


def read_panel_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def write_single_panel_excel(frame: pd.DataFrame, path: Path, meta: dict[str, str]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="data")
        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name="metadata")


def write_workbook(folder: Path, figure: str, rows: pd.DataFrame) -> None:
    workbook = folder / f"{safe_name(figure)}_all_panel_data.xlsx"
    used: set[str] = set()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        manifest_rows = []
        for _, row in rows.iterrows():
            source = PLOTTING_PACKAGE / str(row["data_file"])
            frame = read_panel_data(source)
            sheet = safe_sheet_name(str(row["panel"]), used)
            frame.to_excel(writer, index=False, sheet_name=sheet)
            manifest_rows.append(
                {
                    "sheet": sheet,
                    "panel": row["panel"],
                    "panel_title": row["panel_title"],
                    "recommended_plot_type": row["recommended_plot_type"],
                    "key_message": row["key_message"],
                    "source_file": row["source_file"],
                    "n_rows": len(frame),
                    "n_columns": len(frame.columns),
                }
            )
        manifest_sheet = safe_sheet_name("panel_manifest", used)
        pd.DataFrame(manifest_rows).to_excel(writer, index=False, sheet_name=manifest_sheet)


def build_package() -> pd.DataFrame:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest["destination"].isin(["main", "supplementary"])].copy()

    package_rows = []
    for _, row in manifest.iterrows():
        folder = folder_for_row(row)
        folder.mkdir(parents=True, exist_ok=True)
        source = PLOTTING_PACKAGE / str(row["data_file"])
        frame = read_panel_data(source)
        panel_file = f"{safe_name(row['panel'])}_{safe_name(row['panel_title'])}.xlsx"
        panel_path = folder / panel_file
        meta = {
            "figure": str(row["figure"]),
            "panel": str(row["panel"]),
            "panel_title": str(row["panel_title"]),
            "recommended_plot_type": str(row["recommended_plot_type"]),
            "key_message": str(row["key_message"]),
            "source_file": str(row["source_file"]),
            "package_source_file": str(row["data_file"]),
            "n_rows": str(len(frame)),
            "n_columns": str(len(frame.columns)),
        }
        write_single_panel_excel(frame, panel_path, meta)
        package_rows.append(
            {
                "figure": row["figure"],
                "panel": row["panel"],
                "panel_title": row["panel_title"],
                "recommended_plot_type": row["recommended_plot_type"],
                "excel_file": str(panel_path.relative_to(OUT)).replace("\\", "/"),
                "n_rows": len(frame),
                "n_columns": len(frame.columns),
                "key_message": row["key_message"],
            }
        )

    for figure, rows in manifest.groupby("figure", sort=False):
        folder = folder_for_row(rows.iloc[0])
        write_workbook(folder, str(figure), rows)

    package_manifest = pd.DataFrame(package_rows)
    package_manifest.to_csv(OUT / f"figure_excel_package_manifest_{DATE}.csv", index=False)
    write_readme(package_manifest)
    return package_manifest


def write_readme(package_manifest: pd.DataFrame) -> None:
    counts = (
        package_manifest.groupby("figure")
        .agg(panel_files=("excel_file", "size"))
        .reset_index()
    )
    lines = [
        "# IPM-GPT Figure Excel Data Package",
        "",
        f"Date: {DATE}",
        "",
        "每个正文图和补充图均单独建了文件夹。",
        "每个文件夹内包含：",
        "",
        "- 每个小图 panel 的独立 `.xlsx` 数据文件。",
        "- 一个 `*_all_panel_data.xlsx` 工作簿，包含该图所有 panel 的 sheet。",
        "",
        "## Figure Folder Counts",
        "",
        "| figure | panel_files |",
        "| --- | ---: |",
    ]
    for _, row in counts.iterrows():
        lines.append(f"| {row['figure']} | {int(row['panel_files'])} |")
    lines.extend(
        [
            "",
            "## 使用方法",
            "",
            "1. 打开对应 Figure 文件夹。",
            "2. 如果画单个小图，使用对应 panel 的 `.xlsx`。",
            "3. 如果想一次查看整张图所有数据，打开 `*_all_panel_data.xlsx`。",
            "4. 每个单独 panel Excel 里有 `data` 和 `metadata` 两个 sheet。",
            "5. `metadata` sheet 写了推荐图形类型和 key message。",
            "",
            "## 注意",
            "",
            "- `Figure 5F` 是正文可选 panel，如果版面紧可以放补充。",
            "- AmpC-axis disruptive 等小样本结果画图时建议标注 exploratory/small n。",
            "- No high-confidence driver 建议标注为 mechanism-unresolved。",
        ]
    )
    (OUT / f"README_figure_excel_data_package_{DATE}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    package_manifest = build_package()
    print(f"Wrote {len(package_manifest)} panel Excel files to {OUT}")


if __name__ == "__main__":
    main()
