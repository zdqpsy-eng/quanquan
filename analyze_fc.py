"""Analyze functional connectivity matrices for group differences without NumPy.

This script loads `.npy` connectivity matrices, computes summary statistics for
each group, and reports per-edge differences between groups for each rest
session. It avoids third-party dependencies so it can run in restricted
environments.
"""

from __future__ import annotations

import ast
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
) -> str:
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
    return "\n".join(lines)


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


def main() -> None:
    root = Path(".")
    pairs = read_region_pairs(root / "Region_Coup_Sequence.txt")
    ap_ids = read_subject_sequence(root / "AP_Rest_1_SubsSeq.txt") + read_subject_sequence(
        root / "AP_Rest_2_SubsSeq.txt"
    )

    depression_flags = find_depression_flags(
        baseline_path=root / "北六基线数据-内感受-Selfharm-Suicide-SITBI-截止0401.xlsx",
        mapping_path=root / "北六入组编号-问卷编号-核磁编号对应表.xlsx",
        ap_ids=ap_ids,
    )

    report_sections = [
        "# Functional connectivity group comparison",
        "This report summarizes differences between the AH and AP groups for each rest session.",
        "",
        "- Rest 1: 6-minute acquisition",
        "- Rest 2: 8-minute acquisition",
        "",
        analyze_session(
            "Rest 1",
            ah_path=root / "AH_Rest_1_FCMat.npy",
            ap_path=root / "AP_Rest_1_FCMat.npy",
            ah_seq=root / "AH_Rest_1_SubsSeq.txt",
            ap_seq=root / "AP_Rest_1_SubsSeq.txt",
            pairs=pairs,
        ),
        analyze_session(
            "Rest 2",
            ah_path=root / "AH_Rest_2_FCMat.npy",
            ap_path=root / "AP_Rest_2_FCMat.npy",
            ah_seq=root / "AH_Rest_2_SubsSeq.txt",
            ap_seq=root / "AP_Rest_2_SubsSeq.txt",
            pairs=pairs,
        ),
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

    Path("analysis_report.md").write_text("\n".join(report_sections), encoding="utf-8")


if __name__ == "__main__":
    main()
