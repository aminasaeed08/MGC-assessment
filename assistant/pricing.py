"""All price arithmetic lives here, in plain Python Decimal - never in a language model.

Every number is parsed out of the price-list markdown at start-up (not typed in from memory),
and the parser fails loudly if the document's layout changes, so a silent wrong price is
impossible.  Rules implemented are exactly the ones in '02_price_list_payment_plan.md'.
"""
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from assistant.docs import Section, get, load_sections

PRICE_DOC = "02"
BASE_SECTIONS = {"A": "Base Prices (Block A)", "B": "Base Prices (Block B)"}


def pkr(x: Decimal | int) -> str:
    return f"PKR {int(Decimal(x).quantize(Decimal(1), ROUND_HALF_UP)):,}"


def _find(pattern: str, text: str, what: str) -> Decimal:
    m = re.search(pattern, text, re.I)
    if not m:
        raise ValueError(f"Could not parse '{what}' from the price list - has the document layout changed?")
    return Decimal(m.group(1))


@dataclass
class Rules:
    base: dict                  # (block, unit_type) -> (Decimal price, area sq ft, citation)
    floor_premiums: list        # [(lo, hi, pct)]
    corner_pct: Decimal
    margalla_pct: Decimal
    cash_pct: Decimal
    upfront_pct: Decimal
    overseas_pct: Decimal
    second_unit_pct: Decimal
    token_pct: Decimal
    premiums_cite: str
    discounts_cite: str
    plan_cite: str


def load_rules() -> Rules:
    secs = load_sections()
    base = {}
    row = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([\d,]+) sq ft\s*\|\s*([\d,]+)\s*\|", re.M)
    for block, heading in BASE_SECTIONS.items():
        sec = get(secs, PRICE_DOC, heading)
        for unit, area, price in row.findall(sec.text):
            base[(block, unit)] = (Decimal(price.replace(",", "")), int(area.replace(",", "")), sec.citation)
    if not base:
        raise ValueError("No base prices parsed")

    prem = get(secs, PRICE_DOC, "Location Premiums")
    floors = [(int(a), int(b), Decimal(p)) for a, b, p in
              re.findall(r"Floors (\d+)[–-](\d+):\s*\*\*\+(\d+(?:\.\d+)?)%\*\*", prem.text)]
    disc = get(secs, PRICE_DOC, "Discounts")
    plan = get(secs, PRICE_DOC, "Standard Payment Plan (4 years)")
    return Rules(
        base=base,
        floor_premiums=floors,
        corner_pct=_find(r"Corner unit:\s*\*\*\+(\d+(?:\.\d+)?)%", prem.text, "corner premium"),
        margalla_pct=_find(r"Margalla-facing:\s*\*\*\+(\d+(?:\.\d+)?)%", prem.text, "Margalla premium"),
        cash_pct=_find(r"Full cash payment:\*\*\s*(\d+(?:\.\d+)?)%", disc.text, "cash discount"),
        upfront_pct=_find(r"50% upfront:\*\*\s*(\d+(?:\.\d+)?)%", disc.text, "50% upfront discount"),
        overseas_pct=_find(r"additional (\d+(?:\.\d+)?)%, stackable", disc.text, "overseas discount"),
        second_unit_pct=_find(r"Second unit by the same buyer:\*\*\s*(\d+(?:\.\d+)?)%", disc.text, "second-unit discount"),
        token_pct=_find(r"\|\s*Booking / token\s*\|\s*(\d+(?:\.\d+)?)%", plan.text, "booking token %"),
        premiums_cite=prem.citation, discounts_cite=disc.citation, plan_cite=plan.citation,
    )


RULES = load_rules()


@dataclass
class Quote:
    unit: str
    block: str
    base: Decimal
    steps: list = field(default_factory=list)   # human-readable calculation, shown to the user
    sources: list = field(default_factory=list)
    total: Decimal = Decimal(0)
    token: Decimal | None = None


def floor_premium_pct(floor: int | None) -> Decimal:
    if floor is None:
        return Decimal(0)
    for lo, hi, pct in RULES.floor_premiums:
        if lo <= floor <= hi:
            return pct
    return Decimal(0)


def quote(unit: str, block: str, *, floor: int | None = None, corner: bool = False, margalla: bool = False,
          full_cash: bool = False, upfront_50: bool = False, overseas: bool = False,
          second_unit: bool = False, want_token: bool = False) -> Quote:
    """Price = base - discounts(base) + premiums(base).
    Both discounts and premiums are percentages of the BASE price: premiums are cumulative
    (added together, not compounded) and the price list says location premiums are not discounted."""
    base, area, cite = RULES.base[(block, unit)]
    q = Quote(unit=unit, block=block, base=base, sources=[cite])
    q.steps.append(f"Base price, {unit} Block {block}: {pkr(base)}")

    # --- premiums (additive, on base) -------------------------------------------------
    prem_parts = []
    fp = floor_premium_pct(floor)
    if fp:
        prem_parts.append((f"floor {floor}", fp))
    if corner:
        prem_parts.append(("corner", RULES.corner_pct))
    if margalla:
        prem_parts.append(("Margalla-facing", RULES.margalla_pct))
    prem_pct = sum((p for _, p in prem_parts), Decimal(0))
    if prem_parts:
        label = " + ".join(f"{n} (+{p}%)" for n, p in prem_parts)
        q.steps.append(f"Location premiums, cumulative: {label} = +{prem_pct}% of base")
        q.sources.append(RULES.premiums_cite)

    # --- discounts (additive, on base) ------------------------------------------------
    disc_parts = []
    if full_cash:
        disc_parts.append(("full cash", RULES.cash_pct))
    if upfront_50:
        disc_parts.append(("50% upfront", RULES.upfront_pct))
    if overseas:
        disc_parts.append(("overseas / Roshan Digital Account", RULES.overseas_pct))
    if second_unit:
        disc_parts.append(("second unit", RULES.second_unit_pct))
    disc_pct = sum((p for _, p in disc_parts), Decimal(0))
    if disc_parts:
        label = " + ".join(f"{n} ({p}%)" for n, p in disc_parts)
        q.steps.append(f"Discounts on base only: {label} = {disc_pct}% of base")
        q.sources.append(RULES.discounts_cite)

    discount_amt = base * disc_pct / 100
    premium_amt = base * prem_pct / 100
    q.total = (base - discount_amt + premium_amt).quantize(Decimal(1), ROUND_HALF_UP)
    if disc_parts:
        q.steps.append(f"Base after discount: {pkr(base)} × (1 − {disc_pct}%) = {pkr(base - discount_amt)}")
    if prem_parts:
        q.steps.append(f"Add premiums: {pkr(base)} × {prem_pct}% = {pkr(premium_amt)}")
    q.steps.append(f"Price: {pkr(q.total)}")

    if want_token:
        q.token = (q.total * RULES.token_pct / 100).quantize(Decimal(1), ROUND_HALF_UP)
        q.steps.append(f"Due at booking (token {RULES.token_pct}% of that price): {pkr(q.token)}")
        q.sources.append(RULES.plan_cite)
    q.sources = list(dict.fromkeys(q.sources))   # de-duplicate, keep order
    return q
