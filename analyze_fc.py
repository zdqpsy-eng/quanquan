"""Analyze functional connectivity matrices for group differences without NumPy.

This script loads `.npy` connectivity matrices, computes summary statistics for
each group, and reports per-edge differences between groups for each rest
session. It avoids third-party dependencies so it can run in restricted
environments.
"""

from __future__ import annotations

import ast
import csv
import math
import struct
import zipfile
from xml.etree import ElementTree as ET
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SUPPORTED_DTYPES = {
    "<f4": "f",  # float32
    "<f8": "d",  # float64
}


class NpyLoadError(RuntimeError):
    """Raised when an .npy file cannot be interpreted."""


def _read_header(path: Path) -> Tuple[dict, int]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise NpyLoadError(f"{path} is not a valid .npy file (missing magic)")

        version = handle.read(2)
        if len(version) != 2:
            raise NpyLoadError(f"{path} header is truncated")
        major, minor = version

        if major == 1:
            header_len_bytes = handle.read(2)
            if len(header_len_bytes) != 2:
                raise NpyLoadError(f"{path} header length is truncated")
            header_len = struct.unpack("<H", header_len_bytes)[0]
        elif major in (2, 3):
            header_len_bytes = handle.read(4)
            if len(header_len_bytes) != 4:
                raise NpyLoadError(f"{path} header length is truncated")
            header_len = struct.unpack("<I", header_len_bytes)[0]
        else:
            raise NpyLoadError(f"{path} has unsupported .npy version {major}.{minor}")

        header_text = handle.read(header_len).decode("latin1")
        header_dict = ast.literal_eval(header_text)
        data_start = handle.tell()
    return header_dict, data_start


def load_npy_matrix(path: Path) -> List[List[float]]:
    """Load a 2D .npy array using only the standard library."""

    header, data_start = _read_header(path)
    descr = header.get("descr")
    shape = header.get("shape")
    fortran_order = header.get("fortran_order", False)

    if fortran_order:
        raise NpyLoadError(f"{path} uses Fortran order, which is not supported")
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise NpyLoadError(f"{path} is expected to store a 2D array, got {shape}")
    if descr not in SUPPORTED_DTYPES:
        raise NpyLoadError(f"{path} has unsupported dtype {descr}")

    rows, cols = shape
    expected = rows * cols
    typecode = SUPPORTED_DTYPES[descr]

    with path.open("rb") as handle:
        handle.seek(data_start)
        raw = handle.read()
    values = array(typecode)
    values.frombytes(raw)
    if len(values) != expected:
        raise NpyLoadError(
            f"{path} contains {len(values)} values, expected {expected}"
        )

    matrix: List[List[float]] = []
    index = 0
    for _ in range(rows):
        row = values[index : index + cols]
        matrix.append([float(v) for v in row])
        index += cols
    return matrix


def read_subject_sequence(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_region_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


@dataclass
class GroupStats:
    count: int
    means: List[float]
    variances: List[float]
    overall_mean: float
    overall_sd: float


def compute_group_stats(values: List[List[float]]) -> GroupStats:
    if not values:
        raise ValueError("Cannot compute statistics for an empty group")

    count = len(values)
    cols = len(values[0])
    sums = [0.0] * cols
    sumsq = [0.0] * cols
    for row in values:
        if len(row) != cols:
            raise ValueError("Rows have inconsistent lengths")
        for i, val in enumerate(row):
            sums[i] += val
            sumsq[i] += val * val

    means = [s / count for s in sums]
    variances = []
    for mean, s2 in zip(means, sumsq):
        if count > 1:
            variances.append(max((s2 / count - mean * mean) * count / (count - 1), 0.0))
        else:
            variances.append(0.0)

    flat_values = [v for row in values for v in row]
    overall_mean = sum(flat_values) / len(flat_values)
    overall_variance = (
        sum((v - overall_mean) ** 2 for v in flat_values) / (len(flat_values) - 1)
        if len(flat_values) > 1
        else 0.0
    )

    return GroupStats(
        count=count,
        means=means,
        variances=variances,
        overall_mean=overall_mean,
        overall_sd=math.sqrt(overall_variance),
    )


@dataclass
class EdgeComparison:
    index: int
    pair: Tuple[str, str]
    mean_a: float
    mean_b: float
    difference: float
    cohen_d: float
    t_value: float


def compare_groups(
    stats_a: GroupStats, stats_b: GroupStats, pairs: Sequence[Tuple[str, str]]
) -> List[EdgeComparison]:
    if len(stats_a.means) != len(stats_b.means):
        raise ValueError("Groups have different feature lengths")
    if len(pairs) != len(stats_a.means):
        raise ValueError("Region pair list does not match connectivity length")

    comparisons: List[EdgeComparison] = []
    n_a, n_b = stats_a.count, stats_b.count
    for idx, (pair, mean_a, mean_b, var_a, var_b) in enumerate(
        zip(pairs, stats_a.means, stats_b.means, stats_a.variances, stats_b.variances)
    ):
        diff = mean_a - mean_b
        pooled_var = None
        if n_a > 1 and n_b > 1:
            pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        if pooled_var is not None and pooled_var > 0:
            cohen_d = diff / math.sqrt(pooled_var)
        else:
            cohen_d = 0.0

        denom = var_a / n_a + var_b / n_b if n_a > 0 and n_b > 0 else 0.0
        t_value = diff / math.sqrt(denom) if denom > 0 else 0.0

        comparisons.append(
            EdgeComparison(
                index=idx,
                pair=pair,
                mean_a=mean_a,
                mean_b=mean_b,
                difference=diff,
                cohen_d=cohen_d,
                t_value=t_value,
            )
        )
    return comparisons


def top_edges(
    comparisons: Iterable[EdgeComparison], key: str, limit: int = 10
) -> List[EdgeComparison]:
    sorted_edges = sorted(
        comparisons, key=lambda item: abs(getattr(item, key)), reverse=True
    )
    return sorted_edges[:limit]


def format_edge_table(edges: Sequence[EdgeComparison]) -> str:
    lines = [
        "| Rank | Edge | Mean AH | Mean AP | Difference | Cohen's d | t-value |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, edge in enumerate(edges, start=1):
        lines.append(
            f"| {rank} | {edge.pair[0]} — {edge.pair[1]} | "
            f"{edge.mean_a:.4f} | {edge.mean_b:.4f} | "
            f"{edge.difference:.4f} | {edge.cohen_d:.3f} | {edge.t_value:.3f} |"
        )
    return "\n".join(lines)


def analyze_session(
    session_name: str,
    ah_path: Path,
    ap_path: Path,
    ah_seq: Path,
    ap_seq: Path,
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[str, dict]:
    ah_data = load_npy_matrix(ah_path)
    ap_data = load_npy_matrix(ap_path)
    ah_ids = read_subject_sequence(ah_seq)
    ap_ids = read_subject_sequence(ap_seq)

    ah_stats = compute_group_stats(ah_data)
    ap_stats = compute_group_stats(ap_data)
    comparisons = compare_groups(ah_stats, ap_stats, pairs)
    strongest = top_edges(comparisons, key="cohen_d", limit=10)

    lines = [
        f"## {session_name}",
        f"- AH subjects: {ah_stats.count} (IDs: {', '.join(ah_ids)})",
        f"- AP subjects: {ap_stats.count} (IDs: {', '.join(ap_ids)})",
        f"- Edges per subject: {len(pairs)}",
        "",
        f"Overall mean connectivity (AH): {ah_stats.overall_mean:.4f} ± {ah_stats.overall_sd:.4f}",
        f"Overall mean connectivity (AP): {ap_stats.overall_mean:.4f} ± {ap_stats.overall_sd:.4f}",
        "",
        "Top 10 edges by absolute Cohen's d:",
        format_edge_table(strongest),
        "",
    ]
    session_data = {
        "name": session_name,
        "ah_ids": ah_ids,
        "ap_ids": ap_ids,
        "ah_stats": ah_stats,
        "ap_stats": ap_stats,
        "top_edges": strongest,
        "pairs": pairs,
    }
    return "\n".join(lines), session_data


def col_to_number(col: str) -> int:
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - 64)
    return idx


def load_xlsx_sheet(path: Path, sheet_name: str = "xl/worksheets/sheet1.xml"):
    """Read a worksheet from an .xlsx file using only the standard library."""

    with zipfile.ZipFile(path) as archive:
        sst = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            si.find(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t").text
            or ""
            for si in sst.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si")
        ]
        sheet = ET.fromstring(archive.read(sheet_name))

    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: Dict[int, Dict[str, str]] = {}
    max_col = 0
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        rnum = int(row.attrib["r"])
        row_cells: Dict[str, str] = {}
        for cell in row.findall("a:c", ns):
            ref = cell.attrib["r"]
            col = "".join(ch for ch in ref if ch.isalpha())
            value_node = cell.find("a:v", ns)
            value = ""
            if value_node is not None:
                if cell.attrib.get("t") == "s":
                    value = strings[int(value_node.text)]
                else:
                    value = value_node.text or ""
            row_cells[col] = value
            max_col = max(max_col, col_to_number(col))
        rows[rnum] = row_cells

    # Headers live on the first row.
    headers: Dict[str, str] = {}
    for col, value in rows.get(1, {}).items():
        headers[col] = value

    return headers, rows


def load_id_mapping(mapping_path: Path) -> Dict[str, str]:
    """Map questionnaire 编号 to MRI subject IDs using the mapping workbook."""

    _, rows = load_xlsx_sheet(mapping_path)
    mapping: Dict[str, str] = {}
    for rnum, cells in rows.items():
        if rnum == 1:
            continue
        # File layout: A=编号, B=基线问卷编号, C=fMRI, D=fMRI(核磁编号)
        base_id = cells.get("A", "")
        if base_id:
            mapping[base_id] = cells.get("D", "")
    return mapping


def find_depression_flags(
    baseline_path: Path, mapping_path: Path, ap_ids: Sequence[str]
) -> List[Tuple[str, str, str]]:
    """Identify AP subjects whose baseline row mentions 抑郁.

    Returns tuples of (subject_id, evidence_header, evidence_value).
    """

    ap_set = set(ap_ids)
    id_map = load_id_mapping(mapping_path)
    headers, rows = load_xlsx_sheet(baseline_path)

    flagged: List[Tuple[str, str, str]] = []
    for rnum, cells in rows.items():
        if rnum == 1:
            continue
        base_id = cells.get("A", "")
        subject_id = id_map.get(base_id, "")
        if subject_id not in ap_set:
            continue

        for col, value in cells.items():
            if "抑郁" in value:
                flagged.append((subject_id, headers.get(col, col), value))
    return flagged


MINI_DIAG_FIELDS = [
    "抑郁发作-当前_MINI",
    "抑郁发作-既往_MINI",
    "抑郁发作-复发_MINI",
    "自杀倾向_MINI",
    "躁狂病史_MINI",
    "躁狂发作-当前_MINI",
    "躁狂发作-既往_MINI",
    "终生惊恐障碍_MINI",
    "当前的惊恐障碍_MINI",
    "广场恐怖_MINI",
    "当前社交恐怖_MINI",
    "强迫症_MINI",
    "创伤后应激障碍_MINI",
    "酒精依赖_MINI",
    "酒精滥用_MINI",
    "物质依赖_MINI",
    "物质滥用_MINI",
    "精神病特征的心境障碍终身_MINI",
    "精神病特征的心境障碍当前_MINI",
    "当前的精神病性障碍_MINI",
    "终生的精神病性障碍_MINI",
    "神经性厌食症_MINI",
    "神经性暴食症_MINI",
    "广泛性焦虑障碍_MINI",
    "反社会人格障碍_MINI",
]

USER_FLAGGED_BASE_IDS = [
    "11251049",
    "12101050",
    "04031005",
    "240320CYW",
    "07311028",
    "08211030",
    "09101034",
    "04111099",
    "03281004",
    "05291011",
    "06171017",
    "06061015",
    "07171023",
    "01161067",
    "062610440",
    "07011132",
    "07071133",
    "07191134",
    "02031007",
    "04181009",
    "07231129",
    "02031039",
    "13191059",
    "08061031",
    "07260131",
    "21031003",
    "02111074",
    "07250025",
    "12131052",
    "12271058",
    "01041062",
    "122501640",
    "02181078",
    "02071073",
    "03111088",
    "02281085",
    "04011094",
    "10291041",
    "10311042",
    "03200002",
    "07240024",
    "06030124",
]


def extract_mini_diagnoses(
    mini_csv: Path, mapping_path: Path, target_ids: Sequence[str]
) -> Tuple[List[Tuple[str, str, List[str]]], List[str]]:
    """Return MINI diagnoses for the requested MRI subject IDs.

    Output is a tuple of (found, missing):
    - found: list of (subject_id, interview_date, diagnoses)
    - missing: MRI subject IDs that had no matching MINI row
    """

    target_set = set(target_ids)
    id_map = load_id_mapping(mapping_path)

    found: Dict[str, Tuple[str, List[str]]] = {}
    with mini_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            base_id = (row.get("被试编号") or "").strip()
            subject_id = id_map.get(base_id, "")
            if subject_id not in target_set:
                continue

            diagnoses: List[str] = []
            for field in MINI_DIAG_FIELDS:
                value = (row.get(field) or "").strip()
                if value and value not in {"0", "-999", "-99", "-9"}:
                    diagnoses.append(f"{field}={value}")

            interview_date = (row.get("访谈时间_MINI") or "").strip()
            found[subject_id] = (interview_date, diagnoses)

    missing = sorted(target_set - found.keys())
    ordered_found = sorted(
        [(sid, found[sid][0], found[sid][1]) for sid in found],
        key=lambda item: item[0],
    )
    return ordered_found, missing


def evaluate_flagged_subjects(
    mini_csv: Path, mapping_path: Path
) -> Tuple[List[Tuple[str, str, str, List[str]]], List[str], List[str]]:
    """Check user-flagged base IDs for NSSI+depression and return status.

    Returns:
    - confirmed: list of (base_id, mri_id, interview_date, diag_fields)
    - missing_mini: base IDs not present in MINI CSV
    - missing_mapping: base IDs with MINI rows but no MRI mapping
    """

    id_map = load_id_mapping(mapping_path)
    confirmed: List[Tuple[str, str, str, List[str]]] = []
    missing_mini: List[str] = []
    missing_mapping: List[str] = []

    with mini_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = {row.get("被试编号", "").strip(): row for row in csv.DictReader(handle)}

    for base_id in USER_FLAGGED_BASE_IDS:
        row = reader.get(base_id)
        if not row:
            missing_mini.append(base_id)
            continue

        dep_positive = any(
            (row.get(field, "").strip() not in {"", "0", "-9", "-99", "-999"})
            for field in ("抑郁发作-当前_MINI", "抑郁发作-既往_MINI", "抑郁发作-复发_MINI")
        )
        nssi_group = row.get("入组", "").strip() == "NSSI组"
        if not (dep_positive and nssi_group):
            continue

        mri_id = id_map.get(base_id, "")
        if not mri_id:
            missing_mapping.append(base_id)
            continue

        diag_fields: List[str] = []
        for field in MINI_DIAG_FIELDS:
            value = (row.get(field) or "").strip()
            if value and value not in {"0", "-9", "-99", "-999"}:
                diag_fields.append(f"{field}={value}")
        confirmed.append((base_id, mri_id, (row.get("访谈时间_MINI") or "").strip(), diag_fields))

    return confirmed, missing_mini, missing_mapping


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_html(
    sessions: List[dict],
    depression_flags: List[Tuple[str, str, str]],
    mini_found: List[Tuple[str, str, List[str]]],
    mini_missing: List[str],
    flagged_confirmed: List[Tuple[str, str, str, List[str]]],
    flagged_missing_mini: List[str],
    flagged_missing_mapping: List[str],
) -> str:
    def render_edge_table(edges: Sequence[EdgeComparison]) -> str:
        rows = []
        for idx, e in enumerate(edges, start=1):
            rows.append(
                f"<tr><td>{idx}</td><td>{escape_html(e.pair[0])} — {escape_html(e.pair[1])}</td>"
                f"<td>{e.mean_a:.4f}</td><td>{e.mean_b:.4f}</td>"
                f"<td>{e.difference:.4f}</td><td>{e.cohen_d:.3f}</td><td>{e.t_value:.3f}</td></tr>"
            )
        rows_html = "\n".join(rows)
        return (
            "<div class='table-wrapper'>"
            "<input class='filter' placeholder='过滤边名称或数值' oninput='filterTable(this)'/>"
            "<table><thead><tr><th>#</th><th>Edge</th><th>Mean AH</th><th>Mean AP</th>"
            "<th>Difference</th><th>Cohen&#39;s d</th><th>t-value</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table></div>"
        )

    session_blocks = []
    for sess in sessions:
        session_blocks.append(
            f"""
        <section>
          <h2>{escape_html(sess['name'])}</h2>
          <p><strong>AH (n={sess['ah_stats'].count}):</strong> {', '.join(sess['ah_ids'])}</p>
          <p><strong>AP (n={sess['ap_stats'].count}):</strong> {', '.join(sess['ap_ids'])}</p>
          <p>Overall mean connectivity — AH: {sess['ah_stats'].overall_mean:.4f} ± {sess['ah_stats'].overall_sd:.4f}; 
          AP: {sess['ap_stats'].overall_mean:.4f} ± {sess['ap_stats'].overall_sd:.4f}</p>
          <details open>
            <summary>Top 10 edges by |Cohen&#39;s d|</summary>
            {render_edge_table(sess['top_edges'])}
          </details>
        </section>
        """
        )

    def render_list(title: str, items: List[str]) -> str:
        if not items:
            return ""
        return "<details open><summary>" + escape_html(title) + "</summary><ul>" + "".join(
            f"<li>{escape_html(it)}</li>" for it in items
        ) + "</ul></details>"

    dep_section = ""
    if depression_flags:
        dep_items = "".join(
            f"<li>AP {escape_html(subj)}: {escape_html(header)} → {escape_html(value)}</li>"
            for subj, header, value in depression_flags
        )
        dep_section = (
            "<section><h2>AP baseline entries containing “抑郁”</h2><ul>"
            f"{dep_items}</ul></section>"
        )

    mini_rows = ""
    for sid, date, diags in mini_found:
        diag_text = "; ".join(diags) if diags else "无阳性MINI诊断字段"
        date_text = f" (访谈时间: {escape_html(date)})" if date else ""
        mini_rows += f"<li>{escape_html(sid)}{date_text}: {escape_html(diag_text)}</li>"
    mini_section = (
        "<section><h2>MINI baseline diagnoses mapped to MRI IDs</h2>"
        "<h3>有访谈记录</h3><ul>"
        f"{mini_rows or '<li>未找到匹配记录</li>'}</ul>"
        f"{render_list('无匹配MINI访谈的MRI编号', mini_missing)}"
        "</section>"
    )

    flagged_rows = ""
    for base_id, mri_id, date, diags in flagged_confirmed:
        diag_text = "; ".join(diags) if diags else "无阳性MINI字段"
        date_text = f" (访谈时间: {escape_html(date)})" if date else ""
        flagged_rows += (
            f"<li>MRI {escape_html(mri_id)} / 基线编号 {escape_html(base_id)}{date_text}: "
            f"{escape_html(diag_text)}</li>"
        )
    flagged_section = (
        "<section><h2>用户标注的 NSSI + 抑郁受试者核查</h2>"
        "<h3>确认名单</h3><ul>"
        f"{flagged_rows or '<li>无符合条件的受试者</li>'}</ul>"
        f"{render_list('提供但未在MINI CSV中找到的编号', flagged_missing_mini)}"
        f"{render_list('有MINI行但缺少MRI映射的编号', flagged_missing_mapping)}"
        "</section>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>脑功能连接组别比较（互动版）</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.6; }}
    section {{ margin-bottom: 32px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 14px; }}
    th {{ background: #f2f2f2; position: sticky; top: 0; }}
    .table-wrapper {{ max-height: 420px; overflow: auto; border: 1px solid #ddd; }}
    .filter {{ margin: 8px 0; padding: 6px; width: 260px; }}
    details summary {{ cursor: pointer; font-weight: 600; }}
  </style>
  <script>
    function filterTable(input) {{
      const query = input.value.toLowerCase();
      const tbody = input.closest('.table-wrapper').querySelector('tbody');
      for (const row of tbody.rows) {{
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
      }}
    }}
  </script>
</head>
<body>
  <h1>脑功能连接组别比较（互动版）</h1>
  <p>提供 Rest 1（6 分钟）与 Rest 2（8 分钟）的组间比较。表格支持本地过滤，便于快速定位差异最大的边。</p>
  {''.join(session_blocks)}
  {dep_section}
  {mini_section}
  {flagged_section}
</body>
</html>"""


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_html(
    sessions: List[dict],
    depression_flags: List[Tuple[str, str, str]],
    mini_found: List[Tuple[str, str, List[str]]],
    mini_missing: List[str],
    flagged_confirmed: List[Tuple[str, str, str, List[str]]],
    flagged_missing_mini: List[str],
    flagged_missing_mapping: List[str],
) -> str:
    def render_edge_table(edges: Sequence[EdgeComparison]) -> str:
        rows = []
        for idx, e in enumerate(edges, start=1):
            rows.append(
                f"<tr><td>{idx}</td><td>{escape_html(e.pair[0])} — {escape_html(e.pair[1])}</td>"
                f"<td>{e.mean_a:.4f}</td><td>{e.mean_b:.4f}</td>"
                f"<td>{e.difference:.4f}</td><td>{e.cohen_d:.3f}</td><td>{e.t_value:.3f}</td></tr>"
            )
        rows_html = "\n".join(rows)
        return (
            "<div class='table-wrapper'>"
            "<input class='filter' placeholder='过滤边名称或数值' oninput='filterTable(this)'/>"
            "<table><thead><tr><th>#</th><th>Edge</th><th>Mean AH</th><th>Mean AP</th>"
            "<th>Difference</th><th>Cohen&#39;s d</th><th>t-value</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table></div>"
        )

    session_blocks = []
    for sess in sessions:
        session_blocks.append(
            f"""
        <section>
          <h2>{escape_html(sess['name'])}</h2>
          <p><strong>AH (n={sess['ah_stats'].count}):</strong> {', '.join(sess['ah_ids'])}</p>
          <p><strong>AP (n={sess['ap_stats'].count}):</strong> {', '.join(sess['ap_ids'])}</p>
          <p>Overall mean connectivity — AH: {sess['ah_stats'].overall_mean:.4f} ± {sess['ah_stats'].overall_sd:.4f}; 
          AP: {sess['ap_stats'].overall_mean:.4f} ± {sess['ap_stats'].overall_sd:.4f}</p>
          <details open>
            <summary>Top 10 edges by |Cohen&#39;s d|</summary>
            {render_edge_table(sess['top_edges'])}
          </details>
        </section>
        """
        )

    def render_list(title: str, items: List[str]) -> str:
        if not items:
            return ""
        return "<details open><summary>" + escape_html(title) + "</summary><ul>" + "".join(
            f"<li>{escape_html(it)}</li>" for it in items
        ) + "</ul></details>"

    dep_section = ""
    if depression_flags:
        dep_items = "".join(
            f"<li>AP {escape_html(subj)}: {escape_html(header)} → {escape_html(value)}</li>"
            for subj, header, value in depression_flags
        )
        dep_section = (
            "<section><h2>AP baseline entries containing “抑郁”</h2><ul>"
            f"{dep_items}</ul></section>"
        )

    mini_rows = ""
    for sid, date, diags in mini_found:
        diag_text = "; ".join(diags) if diags else "无阳性MINI诊断字段"
        date_text = f" (访谈时间: {escape_html(date)})" if date else ""
        mini_rows += f"<li>{escape_html(sid)}{date_text}: {escape_html(diag_text)}</li>"
    mini_section = (
        "<section><h2>MINI baseline diagnoses mapped to MRI IDs</h2>"
        "<h3>有访谈记录</h3><ul>"
        f"{mini_rows or '<li>未找到匹配记录</li>'}</ul>"
        f"{render_list('无匹配MINI访谈的MRI编号', mini_missing)}"
        "</section>"
    )

    flagged_rows = ""
    for base_id, mri_id, date, diags in flagged_confirmed:
        diag_text = "; ".join(diags) if diags else "无阳性MINI字段"
        date_text = f" (访谈时间: {escape_html(date)})" if date else ""
        flagged_rows += (
            f"<li>MRI {escape_html(mri_id)} / 基线编号 {escape_html(base_id)}{date_text}: "
            f"{escape_html(diag_text)}</li>"
        )
    flagged_section = (
        "<section><h2>用户标注的 NSSI + 抑郁受试者核查</h2>"
        "<h3>确认名单</h3><ul>"
        f"{flagged_rows or '<li>无符合条件的受试者</li>'}</ul>"
        f"{render_list('提供但未在MINI CSV中找到的编号', flagged_missing_mini)}"
        f"{render_list('有MINI行但缺少MRI映射的编号', flagged_missing_mapping)}"
        "</section>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>脑功能连接组别比较（互动版）</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.6; }}
    section {{ margin-bottom: 32px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 14px; }}
    th {{ background: #f2f2f2; position: sticky; top: 0; }}
    .table-wrapper {{ max-height: 420px; overflow: auto; border: 1px solid #ddd; }}
    .filter {{ margin: 8px 0; padding: 6px; width: 260px; }}
    details summary {{ cursor: pointer; font-weight: 600; }}
  </style>
  <script>
    function filterTable(input) {{
      const query = input.value.toLowerCase();
      const tbody = input.closest('.table-wrapper').querySelector('tbody');
      for (const row of tbody.rows) {{
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
      }}
    }}
  </script>
</head>
<body>
  <h1>脑功能连接组别比较（互动版）</h1>
  <p>提供 Rest 1（6 分钟）与 Rest 2（8 分钟）的组间比较。表格支持本地过滤，便于快速定位差异最大的边。</p>
  {''.join(session_blocks)}
  {dep_section}
  {mini_section}
  {flagged_section}
</body>
</html>"""


def main() -> None:
    root = Path(".")
    pairs = read_region_pairs(root / "Region_Coup_Sequence.txt")
    ap_ids = read_subject_sequence(root / "AP_Rest_1_SubsSeq.txt") + read_subject_sequence(
        root / "AP_Rest_2_SubsSeq.txt"
    )
    ah_ids = read_subject_sequence(root / "AH_Rest_1_SubsSeq.txt") + read_subject_sequence(
        root / "AH_Rest_2_SubsSeq.txt"
    )

    depression_flags = find_depression_flags(
        baseline_path=root / "北六基线数据-内感受-Selfharm-Suicide-SITBI-截止0401.xlsx",
        mapping_path=root / "北六入组编号-问卷编号-核磁编号对应表.xlsx",
        ap_ids=ap_ids,
    )

    mini_found, mini_missing = extract_mini_diagnoses(
        mini_csv=root / "TP追踪_基线随访访谈_基线问卷csv.csv",
        mapping_path=root / "北六入组编号-问卷编号-核磁编号对应表.xlsx",
        target_ids=sorted(set(ap_ids + ah_ids)),
    )
    flagged_confirmed, flagged_missing_mini, flagged_missing_mapping = evaluate_flagged_subjects(
        mini_csv=root / "TP追踪_基线随访访谈_基线问卷csv.csv",
        mapping_path=root / "北六入组编号-问卷编号-核磁编号对应表.xlsx",
    )

    # Build session data for both markdown and interactive HTML.
    session_sections = []
    session_data = []
    for name, ah_path, ap_path, ah_seq, ap_seq in [
        (
            "Rest 1",
            root / "AH_Rest_1_FCMat.npy",
            root / "AP_Rest_1_FCMat.npy",
            root / "AH_Rest_1_SubsSeq.txt",
            root / "AP_Rest_1_SubsSeq.txt",
        ),
        (
            "Rest 2",
            root / "AH_Rest_2_FCMat.npy",
            root / "AP_Rest_2_FCMat.npy",
            root / "AH_Rest_2_SubsSeq.txt",
            root / "AP_Rest_2_SubsSeq.txt",
        ),
    ]:
        section_text, data = analyze_session(
            name, ah_path=ah_path, ap_path=ap_path, ah_seq=ah_seq, ap_seq=ap_seq, pairs=pairs
        )
        session_sections.append(section_text)
        session_data.append(data)

    report_sections = [
        "# Functional connectivity group comparison",
        "This report summarizes differences between the AH and AP groups for each rest session.",
        "",
        "- Rest 1: 6-minute acquisition",
        "- Rest 2: 8-minute acquisition",
        "",
        session_sections[0],
        session_sections[1],
    ]

    if depression_flags:
        report_sections.extend(
            [
                "## AP participants with depression-related entries in baseline questionnaire",
                "The following AP subjects had baseline responses containing the string “抑郁”. "
                "Evidence is pulled directly from the questionnaire fields for review.",
                "",
            ]
        )
        for subject_id, header, value in depression_flags:
            report_sections.append(f"- AP {subject_id}: {header} → {value}")
        report_sections.append("")

    if mini_found or mini_missing:
        report_sections.extend(
            [
                "## MINI baseline diagnoses mapped to MRI subject IDs",
                "Values are reported as raw MINI field outputs; any non-zero/non-missing entries are listed per subject.",
                "",
                "### Subjects with MINI interview rows",
            ]
        )
        if mini_found:
            for subject_id, date, diagnoses in mini_found:
                diag_text = "; ".join(diagnoses) if diagnoses else "无阳性MINI诊断字段"
                date_text = f" (访谈时间: {date})" if date else ""
                report_sections.append(f"- {subject_id}{date_text}: {diag_text}")
        else:
            report_sections.append("- 未找到任何匹配的 MINI 访谈行")

        if mini_missing:
            report_sections.extend(
                [
                    "",
                    "### MRI subjects without a matching MINI interview row",
                    f"{', '.join(mini_missing)}",
                    "",
                ]
            )

    if flagged_confirmed or flagged_missing_mini or flagged_missing_mapping:
        report_sections.extend(
            [
                "## User-flagged NSSI + depression cases (MINI cross-check)",
                "Below are the provided base IDs verified against MINI and mapped to MRI IDs.",
                "",
                "### Confirmed (NSSI组 + 抑郁阳性 + 有MRI编号)",
            ]
        )
        if flagged_confirmed:
            for base_id, mri_id, date, diags in sorted(flagged_confirmed, key=lambda x: x[1]):
                diag_text = "; ".join(diags) if diags else "无阳性MINI字段"
                date_text = f" (访谈时间: {date})" if date else ""
                report_sections.append(f"- MRI {mri_id} / 基线编号 {base_id}{date_text}: {diag_text}")
        else:
            report_sections.append("- 无符合条件的受试者")

        if flagged_missing_mini:
            report_sections.extend(
                [
                    "",
                    "### Provided IDs missing in MINI CSV",
                    f"{', '.join(flagged_missing_mini)}",
                ]
            )

        if flagged_missing_mapping:
            report_sections.extend(
                [
                    "",
                    "### Provided IDs with MINI rows but no MRI mapping",
                    f"{', '.join(flagged_missing_mapping)}",
                ]
            )
        report_sections.append("")

    Path("analysis_report.md").write_text("\n".join(report_sections), encoding="utf-8")

    html = render_html(
        sessions=session_data,
        depression_flags=depression_flags,
        mini_found=mini_found,
        mini_missing=mini_missing,
        flagged_confirmed=flagged_confirmed,
        flagged_missing_mini=flagged_missing_mini,
        flagged_missing_mapping=flagged_missing_mapping,
    )
    Path("analysis_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
