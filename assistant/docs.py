"""Load the three MGC documents and split them into citable sections (one per '## ' heading)."""
import re
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


@dataclass(frozen=True)
class Section:
    doc: str        # file name, e.g. '02_price_list_payment_plan.md'
    doc_title: str  # H1 of the file
    heading: str    # H2 of the section
    text: str       # body of the section (markdown)

    @property
    def citation(self) -> str:
        return f"{self.doc} · {self.heading}"


def load_sections() -> list[Section]:
    sections = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        title = re.search(r"^# (.+)$", raw, re.M).group(1).strip()
        # re.split with a capture group keeps the headings: [preamble, h1, body1, h2, body2, ...]
        parts = re.split(r"^## (.+)$", raw, flags=re.M)
        for heading, body in zip(parts[1::2], parts[2::2]):
            sections.append(Section(path.name, title, heading.strip(), body.strip()))
    return sections


def get(sections: list[Section], doc_prefix: str, heading: str) -> Section:
    """Look a section up by file prefix ('02') and exact heading. Raises if the docs changed shape."""
    for s in sections:
        if s.doc.startswith(doc_prefix) and s.heading == heading:
            return s
    raise KeyError(f"section not found: {doc_prefix} / {heading}")
