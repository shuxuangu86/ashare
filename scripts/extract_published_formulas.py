from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract formula-only metadata from the Alpha101 and GTJA191 source PDFs"
    )
    parser.add_argument("--alpha101-pdf", type=Path, required=True)
    parser.add_argument("--gtja191-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/factors"))
    return parser


def main() -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("formula extraction requires pypdf") from exc

    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alpha_pages = tuple(page.extract_text() or "" for page in PdfReader(args.alpha101_pdf).pages)
    gtja_pages = tuple(page.extract_text() or "" for page in PdfReader(args.gtja191_pdf).pages)
    alpha_formulas = _extract_alpha101(alpha_pages)
    gtja_formulas = _extract_gtja191(gtja_pages)
    if len(alpha_formulas) != 101 or len(gtja_formulas) != 191:
        raise ValueError(
            f"unexpected extraction counts: Alpha101={len(alpha_formulas)}, "
            f"GTJA191={len(gtja_formulas)}"
        )
    _write(
        args.output_dir / "alpha101_formulas.json",
        "SRC_ALPHA101",
        args.alpha101_pdf,
        alpha_formulas,
    )
    _write(
        args.output_dir / "gtja191_formulas.json",
        "SRC_GTJA_ALPHA191",
        args.gtja191_pdf,
        gtja_formulas,
    )
    print(f"extracted Alpha101={len(alpha_formulas)} GTJA191={len(gtja_formulas)}")
    return 0


def _extract_alpha101(pages: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    selected = pages[7:15]
    page_map = _page_map(selected, re.compile(r"Alpha#(\d+)\s*:"), first_page=8)
    text = "\n".join(_clean_page(page, alpha101=True) for page in selected)
    text = text[text.index("Alpha#1:") : text.index("A.1. Functions and Operators")]
    matches = tuple(re.finditer(r"Alpha#(\d+)\s*:", text))
    return tuple(
        {
            "formula_id": int(match.group(1)),
            "source_page": page_map[int(match.group(1))],
            "raw_formula": _clean_formula(
                text[
                    match.end() : matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(text)
                ]
            ),
        }
        for index, match in enumerate(matches)
    )


def _extract_gtja191(pages: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    selected = pages[10:17]
    page_map = _page_map(selected, re.compile(r"Alpha(\d+)\s+"), first_page=11)
    text = "\n".join(_clean_page(page, alpha101=False) for page in selected)
    text = text[text.index("Alpha1 ") :]
    matches = tuple(re.finditer(r"Alpha(\d+)\s+", text))
    formulas: list[dict[str, object]] = []
    for index, match in enumerate(matches):
        raw = text[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(text)
        ]
        if int(match.group(1)) == 191:
            raw = re.split(r"数据来源|3\.4\.", raw, maxsplit=1)[0]
        formulas.append(
            {
                "formula_id": int(match.group(1)),
                "source_page": page_map[int(match.group(1))],
                "raw_formula": _clean_formula(raw),
            }
        )
    return tuple(formulas)


def _page_map(
    pages: tuple[str, ...],
    pattern: re.Pattern[str],
    *,
    first_page: int,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for offset, page in enumerate(pages):
        for match in pattern.finditer(page):
            result[int(match.group(1))] = first_page + offset
    return result


def _clean_page(page: str, *, alpha101: bool) -> str:
    lines = []
    for line in page.splitlines():
        stripped = line.strip()
        if not stripped or re.fullmatch(r"\d+", stripped):
            continue
        if not alpha101 and (
            "数量化专题报告" in stripped
            or "请务必阅读正文之后" in stripped
            or re.search(r"\d+\s+of\s+32", stripped)
        ):
            continue
        lines.append(line)
    return "\n".join(lines)


def _clean_formula(value: str) -> str:
    value = value.replace("\u2013", "-").replace("\uff0c", ",")
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace("D ELAY", "DELAY")


def _write(
    path: Path,
    source_id: str,
    source_pdf: Path,
    formulas: tuple[dict[str, object], ...],
) -> None:
    payload = {
        "schema_version": "aquant.published-formulas.v1",
        "source_id": source_id,
        "source_pdf_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        "formula_count": len(formulas),
        "formulas": formulas,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
