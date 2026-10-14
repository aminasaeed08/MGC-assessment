"""Part 1: grounded document assistant.   ask(question) -> Answer

Design (why it is built this way):
  * Every answer is assembled from the three documents and carries its citations
    (file · section). An answer with no source is never returned.
  * Arithmetic is done by pricing.py (Decimal), never by a language model.
  * Hard cases are handled by explicit rules, checked BEFORE any retrieval:
        - not in the documents / must not be answered  -> refuse + say who to ask
        - documents contradict each other              -> show both values, do not pick one
  * Anything else goes to retrieval: the best-matching document lines are quoted back
    with their section, or we say 'not in the documents' if nothing matches well enough.
  * No API key / network is needed, so it runs anywhere from the README.
"""
import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from assistant import pricing
from assistant.docs import Section, get, load_sections
from assistant.pricing import pkr

SECTIONS = load_sections()


@dataclass
class Answer:
    status: str                      # answered | conflict | refused | unconfirmed | needs_info | not_found
    headline: str
    details: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    sources: list = field(default_factory=list)

    def text(self) -> str:
        parts = [self.headline]
        if self.details:
            parts += [""] + self.details
        if self.notes:
            parts += [""] + [f"• {n}" for n in self.notes]
        parts += ["", "Sources: " + "; ".join(self.sources)]
        return "\n".join(parts)


def cite(prefix: str, heading: str) -> str:
    return get(SECTIONS, prefix, heading).citation


# --------------------------------------------------------------------------------------
# 1. Questions the documents deliberately do NOT answer
# --------------------------------------------------------------------------------------
def _refusal(q: str) -> Answer | None:
    if re.search(r"rental|rent\b|yield|roi\b|return on investment|capital gain|appreciation", q):
        return Answer(
            "refused",
            "I can't give that. MGC does not publish rental-yield (or return) projections, and sales staff must not give projections verbally.",
            ["Please direct this question to the marketing manager."],
            sources=[cite("03", "Frequently Asked")],
        )
    if re.search(r"anchor|major tenant|main tenant|flagship tenant", q):
        return Answer(
            "unconfirmed",
            "There is no confirmed anchor tenant. Anchor-tenancy discussions are ongoing and none has been confirmed as of the brochure's issue date (March 2025).",
            ["Don't name or hint at a tenant. Check with management for any update since the brochure was issued."],
            sources=[cite("01", "Commercial Podium")],
        )
    if re.search(r"mark-?up|interest rate|loan rate|bank rate|home loan|mortgage|financing rate", q):
        return Answer(
            "refused",
            "MGC has arrangements in principle with two banks, but terms, eligibility and mark-up rates are set by the bank. Sales staff must not quote mark-up rates.",
            ["Refer the buyer to the bank for rates and eligibility."],
            sources=[cite("03", "Frequently Asked")],
        )
    if re.search(r"\bgas\b|sui", q):
        return Answer(
            "unconfirmed",
            "A Sui gas connection has been applied for. There is no confirmed timeline, so don't promise one.",
            sources=[cite("03", "Frequently Asked")],
        )
    return None


# --------------------------------------------------------------------------------------
# 2. Known contradictions between documents (checked for the question AND any answer text)
# --------------------------------------------------------------------------------------
def _transfer_fee_values() -> tuple[str, str]:
    """Read the two figures out of the documents themselves (not typed in from memory)."""
    price_list = get(SECTIONS, "02", "Other Charges").text
    faq = get(SECTIONS, "03", "Transfers").text
    a = re.search(r"Transfer fee[^:]*:\s*(\d+(?:\.\d+)?)%", price_list)
    b = re.search(r"Transfer fee is \*\*(\d+(?:\.\d+)?)%", faq)
    if not (a and b):
        raise ValueError("Could not parse the transfer fee from the documents - has their layout changed?")
    return a.group(1), b.group(1)


def _transfer_fee_conflict() -> Answer | None:
    pl, faq = _transfer_fee_values()
    if pl == faq:
        return None                       # the documents now agree; no conflict to flag
    return Answer(
        "conflict",
        "The documents disagree on the transfer fee, so I can't give one number.",
        [
            f"• Price List says {pl}% of the current list price (transfer before possession).",
            f"• Booking FAQ says {faq}% of the current list price (after the confirmation payment and at least 4 quarterly instalments).",
        ],
        [
            "Please confirm the correct figure with management before quoting it to a buyer.",
            "For context only: the FAQ (v2.1, May 2025) is dated after the price list (effective 1 April 2025). That may mean "
            f"{faq}% is newer, but the documents don't say so, so treat it as unverified.",
        ],
        [cite("02", "Other Charges"), cite("03", "Transfers")],
    )


def _is_transfer_fee_question(q: str) -> bool:
    return "transfer" in q and bool(re.search(r"fee|charge|cost|%|percent|how much|pay", q))


# --------------------------------------------------------------------------------------
# 3. Price questions -> parse the unit, call the Decimal calculator
# --------------------------------------------------------------------------------------
UNIT_PATTERNS = [   # order matters: most specific first
    ("4-Bed Penthouse", r"penthouse|4[- ]?bed|four[- ]?bed"),
    ("3-Bed Executive", r"3[- ]?bed|three[- ]?bed|executive"),
    ("2-Bed", r"2[- ]?bed|two[- ]?bed"),
    ("1-Bed Standard", r"1[- ]?bed|one[- ]?bed"),
    ("Studio", r"studio"),
]
PRICE_WORDS = r"price|cost|total|how much|pay|payable|discount|booking|token|due|worth|base|rate|quote"


def _parse_unit(q: str) -> str | None:
    for unit, pat in UNIT_PATTERNS:
        if re.search(pat, q):
            return unit
    return None


def _price_answer(q: str) -> Answer | None:
    unit = _parse_unit(q)
    if unit is None or not re.search(PRICE_WORDS, q):
        return None

    m = re.search(r"block\s*([ab])\b", q)
    block = m.group(1).upper() if m else None
    m = re.search(r"floor\s*(?:no\.?\s*)?(\d{1,2})|(\d{1,2})(?:st|nd|rd|th)\s*floor", q)
    floor = int(m.group(1) or m.group(2)) if m else None
    corner = "corner" in q
    margalla = "margalla" in q
    notes = []
    if floor is not None and floor > 22:
        return Answer("refused", f"There is no floor {floor}: the tower has 22 storeys, so there is no price to quote.", sources=[cite("01", "Overview")])
    if floor is not None and 1 <= floor <= 3:
        return Answer("refused", f"Floors 1–3 are the commercial podium, not residential, so the residential price list doesn't apply "
                      "(and discounts don't apply to podium units). Commercial payment plans are issued separately on request.",
                      sources=[cite("01", "Commercial Podium"), cite("02", "Discounts")])

    # payment options -------------------------------------------------------------
    full_cash = bool(re.search(r"full cash|all cash|cash payment|lump ?sum|full payment", q))
    upfront_50 = bool(re.search(r"50\s*%\s*(up ?front|advance|down)|up ?front.{0,20}50\s*%|half.{0,10}up ?front", q))
    overseas = bool(re.search(r"overseas|roshan|\brda\b|abroad|expat|\bnri\b", q))
    second = bool(re.search(r"second unit|2nd unit|another unit|second apartment", q))
    want_token = bool(re.search(r"booking|token|due now|down ?payment", q))
    if full_cash and upfront_50:
        return Answer("needs_info", "You mention both full cash and 50% upfront; these are separate discounts. Which payment option applies?",
                      sources=[cite("02", "Discounts")])

    # resolve the unit type (2-bed is ambiguous: Standard vs Corner) --------------------
    if unit == "2-Bed":
        unit = "2-Bed Corner" if corner else "2-Bed Standard"
    blocks_for_unit = sorted(b for (b, u) in pricing.RULES.base if u == unit)

    # impossible combinations straight from the brochure ---------------------------------
    if margalla and unit in ("Studio", "1-Bed Standard"):
        return Answer("refused", f"{unit} units are not available with a Margalla-facing orientation, so there is no such price.",
                      sources=[cite("01", "Unit Configuration")])
    if corner and unit not in ("2-Bed Corner",):
        notes.append(f"The brochure only lists a corner configuration for the 2-Bed (Block B). Check a corner {unit} exists before quoting.")

    if block is None:
        if len(blocks_for_unit) == 1:
            block = blocks_for_unit[0]
            notes.append(f"{unit} is sold in Block {block} only, so I used Block {block}.")
        else:
            return _both_blocks(unit, blocks_for_unit, floor, corner, margalla, full_cash, upfront_50, overseas, second, want_token)
    if block not in blocks_for_unit:
        avail = ", ".join(f"Block {b}" for b in blocks_for_unit) or "no block"
        return Answer("not_found", f"{unit} is not listed in Block {block}. It is available in: {avail}.",
                      sources=[cite("02", pricing.BASE_SECTIONS[blocks_for_unit[0]]), cite("01", "Unit Configuration")])

    # alternative reading of "corner 2-bed in Block B": Standard base + corner premium --------
    alt = None
    if unit == "2-Bed Corner" and block == "B":
        alt = pricing.quote("2-Bed Standard", "B", floor=floor, corner=True, margalla=margalla, full_cash=full_cash,
                            upfront_50=upfront_50, overseas=overseas, second_unit=second, want_token=want_token)

    qt = pricing.quote(unit, block, floor=floor, corner=corner or unit == "2-Bed Corner", margalla=margalla, full_cash=full_cash,
                       upfront_50=upfront_50, overseas=overseas, second_unit=second, want_token=want_token)
    if unit == "2-Bed Corner" and not corner:
        qt = pricing.quote(unit, block, floor=floor, corner=False, margalla=margalla, full_cash=full_cash,
                           upfront_50=upfront_50, overseas=overseas, second_unit=second, want_token=want_token)

    asks_total = bool(re.search(r"total|premium|floor|margalla|corner|discount|after|upfront|cash|overseas|booking|due", q)) or floor is not None
    if qt.token is not None:
        headline = f"Price after discounts/premiums: {pkr(qt.total)}. Due at booking: {pkr(qt.token)}. ({unit}, Block {block})"
    elif asks_total:
        headline = f"Total: {pkr(qt.total)} ({unit}, Block {block})"
    else:
        headline = f"{pkr(qt.total)} — base price, {unit}, Block {block}"
    details = (["Calculation (done in code, not by a language model):"] + [f"  {s}" for s in qt.steps]) if asks_total else []

    if alt is not None:     # ambiguous corner 2-bed: show BOTH readings up front, with both calculations
        area_c, area_s = pricing.RULES.base[("B", "2-Bed Corner")][1], pricing.RULES.base[("B", "2-Bed Standard")][1]
        headline = (f"Two readings of the price list apply, please confirm which: "
                    f"{pkr(qt.total)} as a 2-Bed Corner unit ({area_c:,} sq ft), or {pkr(alt.total)} if priced as a 2-Bed Standard "
                    f"({area_s:,} sq ft) plus the corner premium.")
        details = ([f"A) 2-Bed Corner unit (own base price)"] + [f"    {s}" for s in qt.steps]
                   + ["B) 2-Bed Standard base + corner premium"] + [f"    {s}" for s in alt.steps])
        notes.append("The price list has BOTH a separate 2-Bed Corner base price and a +3% corner premium, and doesn't say whether they combine "
                     "for a corner 2-bed. Confirm with the sales manager before quoting.")

    # extra base-price context -------------------------------------------------------
    if not asks_total and unit == "2-Bed Standard" and ("B", "2-Bed Corner") in pricing.RULES.base:
        c, a_c, _ = pricing.RULES.base[("B", "2-Bed Corner")]
        notes.append(f"The 2-Bed Corner in Block B is a separate, larger unit ({a_c:,} sq ft) with its own base price of {pkr(c)}.")
    _unit_notes(notes, block, floor, margalla, unit, want_token, full_cash or upfront_50 or overseas)
    sources = qt.sources + ([cite("01", "Unit Configuration")] if (margalla or floor or corner) else [])
    return Answer("answered", headline, details, notes, list(dict.fromkeys(sources)))


def _both_blocks(unit, blocks, floor, corner, margalla, full_cash, upfront_50, overseas, second, want_token) -> Answer:
    lines, sources = [], []
    for b in blocks:
        qt = pricing.quote(unit, b, floor=floor, corner=corner, margalla=margalla, full_cash=full_cash,
                           upfront_50=upfront_50, overseas=overseas, second_unit=second, want_token=want_token)
        lines.append(f"Block {b}: {pkr(qt.total)}" + (f"  (due at booking {pkr(qt.token)})" if qt.token is not None else ""))
        sources += qt.sources
    return Answer("needs_info", f"You didn't say which block. {unit} is sold in both:", lines,
                  ["Tell me the block and I'll give the full calculation."], list(dict.fromkeys(sources)))


def _unit_notes(notes, block, floor, margalla, unit, want_token, has_payment_option):
    brochure = cite("01", "Unit Configuration")
    if margalla and block == "A":
        notes.append("The brochure says units above floor 12 in Block B face the Margalla range; it gives no Margalla orientation for Block A.")
    if margalla and block == "B" and floor is not None and floor <= 12:
        notes.append("The brochure says units ABOVE floor 12 in Block B face Margalla; a Margalla-facing unit on floor ≤12 isn't covered. Verify availability.")
    if (not margalla) and block == "B" and floor is not None and floor > 12 and unit not in ("Studio", "1-Bed Standard"):
        notes.append(f"Block B units above floor 12 face Margalla (see {brochure}); the +6% Margalla premium was NOT added because you didn't ask for it. Add it if the unit is sold as Margalla-facing.")
    if want_token:
        notes.append("Token = 10% per the Standard Payment Plan, taken here on the price after discounts. The price list doesn't spell out the payment schedule for the cash / 50%-upfront options, so confirm the booking amount with management.")
    notes.append("Excludes Other Charges (utility connection, mandatory parking slot, maintenance advance). Price list is effective 1 April 2025 and 'subject to revision'; check it is still current.")


# --------------------------------------------------------------------------------------
# 4. Retrieval fallback: quote the best-matching sentences / rows, or admit it isn't there
# --------------------------------------------------------------------------------------
_STOP = {"what", "whats", "is", "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "do", "does", "can", "how", "much", "many",
         "are", "there", "any", "i", "we", "you", "it", "me", "my", "about", "tell", "with", "at", "be", "if", "when", "who", "which",
         "have", "has", "will", "was", "this", "that", "your", "our", "get", "got", "available", "happen", "happens", "please", "give", "someone", "somebody", "else", "another", "involve", "want", "need"}
# Everyday sales words that the documents phrase differently.
_ALIAS = {"gym": "gymnasium", "swimming": "pool", "lift": "elevator", "wifi": "internet", "mosque": "prayer", "cctv": "security",
          "timings": "open", "timing": "open", "hours": "open", "schedule": "open"}


def _tokens(s: str) -> set:
    """Lower-cased content words with a crude plural strip (units -> unit, charges -> charge)."""
    out = set()
    for w in re.findall(r"[a-z0-9]+", s.lower()):
        w = _ALIAS.get(w, w)
        if w in _STOP or len(w) <= 2:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


@dataclass(frozen=True)
class Item:
    section: Section
    text: str            # what we show
    header: str = ""     # table header row, shown above matching rows


def _split_items(sec: Section) -> list[Item]:
    """Break a section into the smallest self-contained facts: table rows, bullets, sentences,
    and FAQ question+answer blocks. (The markdown is hard-wrapped, so lines are NOT sentences.)"""
    items = []
    for block in re.split(r"\n\s*\n", sec.text):
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        if lines[0].startswith("|"):                                   # table
            header = lines[0]
            items += [Item(sec, l, header) for l in lines[2:]]
        elif all(re.match(r"^(\s*[-*]|\s*\d+\.)\s", l) or l.startswith("  ") for l in lines):   # bullet / numbered list
            cur = []
            for l in lines:
                if re.match(r"^\s*([-*]|\d+\.)\s", l) and cur:
                    items.append(Item(sec, " ".join(cur)))
                    cur = []
                cur.append(l.strip())
            items.append(Item(sec, " ".join(cur)))
        elif lines[0].startswith("**"):                                # FAQ: bold question + answer
            items.append(Item(sec, " ".join(l.strip() for l in lines)))
        else:                                                          # prose paragraph -> sentences
            items += [Item(sec, x) for x in re.split(r"(?<=[.!?])\s+", " ".join(l.strip() for l in lines)) if x]
    return items


ITEMS = [it for sec in SECTIONS for it in _split_items(sec)]
_item_text = [f"{it.section.heading} {it.header} {it.text}" for it in ITEMS]
_vec = TfidfVectorizer(tokenizer=lambda t: list(_tokens(t)), lowercase=False, token_pattern=None, sublinear_tf=True).fit(_item_text)
_mat = _vec.transform(_item_text)
MIN_COVERAGE = 0.50    # see _coverage(). Measured gap: off-topic questions score <= 0.45, real ones >= 0.55

_SECTION_TOKENS = {sec: _tokens(f"{sec.heading} {sec.text}") for sec in SECTIONS}
_DF = {}
for _toks in _SECTION_TOKENS.values():
    for _t in _toks:
        _DF[_t] = _DF.get(_t, 0) + 1


def _idf(t: str) -> float:
    import math
    return math.log((len(SECTIONS) + 1) / (_DF.get(t, 0) + 1)) + 1     # unknown word -> highest weight


def _coverage(q_tokens: set, sec: Section) -> float:
    """Share of the question's *meaning* (IDF-weighted) found in this section.
    A question about 'weather in Islamabad' covers only the cheap word 'islamabad', while the rare word
    'weather' is nowhere in the documents -> low coverage -> we refuse instead of returning an unrelated passage.
    Pure numbers are ignored ('after 40 days' shouldn't fail because 40 isn't in the text)."""
    words = {t for t in q_tokens if not t.isdigit()}
    if not words:
        return 0.0
    got = sum(_idf(t) for t in words & _SECTION_TOKENS[sec])
    return got / sum(_idf(t) for t in words)


def _tag(sec: Section) -> str:
    return f"[{sec.doc[:2]} · {sec.heading}]"


def _retrieve(q: str) -> Answer:
    q_tokens = _tokens(q)
    cov = {sec: _coverage(q_tokens, sec) for sec in SECTIONS}
    ok = {sec for sec, c in cov.items() if c >= MIN_COVERAGE}
    sims = cosine_similarity(_vec.transform([q]), _mat)[0]
    cand = [i for i, it in enumerate(ITEMS) if it.section in ok and sims[i] > 0]
    if not cand:
        return Answer("not_found", "I couldn't find that in the three MGC documents (brochure, price list, booking FAQ), so I won't guess.",
                      ["Check with the relevant manager."], sources=["no matching section"])

    # If the question simply names a section ("booking process", "payment plan"), show that whole section.
    named = [sec for sec in ok if q_tokens and q_tokens <= _tokens(sec.heading)]
    if named:
        sec = named[0]
        return Answer("answered", f"{sec.heading} ({sec.doc_title}):", [f"  {l}" for l in sec.text.splitlines() if l.strip()],
                      sources=[sec.citation])

    # Otherwise: the best few lines overall, each tagged with the section it came from.
    top = max(sims[i] for i in cand)
    idx = [i for i in sorted(cand, key=lambda i: -sims[i])[:3] if sims[i] >= 0.5 * top]
    lines, sources = [], []
    for i in idx:
        it = ITEMS[i]
        lines.append(f"  {it.text}   {_tag(it.section)}")
        sources.append(it.section.citation)
    return Answer("answered", "Closest passages in the documents (quoted as written):", lines,
                  ["These are the documents' own words, not a summary. If this isn't what you asked, rephrase or ask the manager."],
                  list(dict.fromkeys(sources)))


# --------------------------------------------------------------------------------------
def ask(question: str) -> Answer:
    q = re.sub(r"\s+", " ", question.lower()).strip()
    if not q:
        return Answer("needs_info", "Please type a question.", sources=["n/a"])
    if _is_transfer_fee_question(q):
        conflict = _transfer_fee_conflict()
        if conflict:
            return conflict
    for rule in (_refusal, _price_answer):
        ans = rule(q)
        if ans is not None:
            return ans
    ans = _retrieve(q)
    # Safety net: if ANY retrieved text mentions the transfer fee, keep the answer but attach the conflict warning.
    if ans.status == "answered" and re.search(r"transfer fee", " ".join(ans.details), re.I):
        conflict = _transfer_fee_conflict()
        if conflict:
            ans.notes.insert(0, "⚠ DOCUMENT CONFLICT on the transfer fee: " + conflict.details[0].lstrip("• ") + " " + conflict.details[1].lstrip("• ")
                             + " Confirm with management before quoting.")
            ans.sources = list(dict.fromkeys(ans.sources + conflict.sources))
    return ans
