# MGC Developments — AI Developer & Engineer build task

One small repo, four parts, one story: help the sales team (1) answer document questions correctly and
(2) call the right leads first.

| Part | Where | Status |
|---|---|---|
| 1. Grounded document assistant | `assistant/` (`qa.py`, `pricing.py`, `docs.py`), tests in `tests/` | Done, 12 tests pass |
| 2. Database | `schema.sql`, `queries.sql`, `run_sql.py` | Done, run against the real CSV |
| 3. ML lead scoring | `ml/` (`train.py`, `features.py`, `score.py`) | Done |
| 4. Web interface | `app.py` | Done (Flask, bare HTML) |

## How to run (Python 3.10+)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m unittest discover -s tests -v   # Part 1 tests (uses unittest, nothing extra to install)
python run_sql.py                         # Part 2: builds mgc.db from leads.csv, prints every query's result
python -m ml.train                        # Part 3: trains, prints the metrics below, writes ml/metrics.json
python app.py                             # Part 4: open http://127.0.0.1:5000
```
`app.py` trains the model automatically on first start if `ml/model.joblib` is missing (about 2 seconds), so you can skip `ml.train`.
No API keys or internet access needed by default (runs offline).

### Optional: LLM Configuration (`.env`)

To enable natural language LLM generation, create a `.env` file in the project root:

```env
# Free Groq key from https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile

# Or use OpenAI:
# OPENAI_API_KEY=your_openai_api_key_here
# LLM_MODEL=gpt-4o-mini
```

If `.env` is omitted, the assistant operates in deterministic offline mode (rule-based & Decimal calculator).

---

## Part 1 — Grounded document assistant

**Approach.** Answers are assembled from the three documents by code, not generated freely, and every answer carries
`file · section` citations (the page shows them under "Sources"). There are three layers, checked in this order:

1. **Rules for the hard cases** (run before any search): rental yield / returns, anchor tenant, bank mark-up rates and gas are
   *refused or flagged unconfirmed*, quoting the document's own instruction (e.g. "direct to the marketing manager").
   The **transfer fee** is handled as a *document conflict*: both values are shown, none is chosen, staff are told to confirm with
   management. The two numbers are parsed out of the documents at runtime, so if MGC fixes the documents the warning disappears by itself.
2. **Price questions → a Decimal calculator** (`pricing.py`). Unit type, block, floor, corner, Margalla and payment option are parsed from
   the question; **all arithmetic is plain Python, never a language model**. Base prices and every percentage are parsed from the price-list
   markdown at start-up (the code raises if the layout changes), so no number is typed in from memory. Every answer shows its calculation steps.
3. **Retrieval fallback** for everything else: the best-matching sentences / table rows are quoted back verbatim with their section.
   If the question's *rare* words aren't covered by the documents ("weather", "CEO", "architect") it says "not in the documents"
   instead of returning a loosely related passage. The threshold was set from measured scores (off-topic ≤ 0.45, real questions ≥ 0.55).

**Results on the six required questions** (all asserted in `tests/test_assistant.py`):

| Question | Answer |
|---|---|
| Base price, 2-bed Block B | PKR 22,425,000 |
| Margalla corner, floor 15, 2-bed Block B | **Two readings, flagged** (see caveat 1): PKR 30,346,150 or PKR 25,340,250; both with +13% (4 + 3 + 6) |
| Overseas + 50% upfront, 3-Bed Executive | 8% off base → PKR 36,321,600; token 10% → PKR 3,632,160 |
| Transfer fee | Conflict: 2% (price list) vs 2.5% (FAQ); confirm with management |
| Rental yield, 1-bed | Refused; not published, ask the marketing manager |
| Anchor tenant | Unconfirmed; discussions ongoing |

**Caveats / judgement calls (please read):**
1. **Corner 2-bed is ambiguous in the price list.** It has *both* a separate "2-Bed Corner" base (26,855,000, 1,310 sq ft) *and* a "+3% corner
   premium", and doesn't say whether they combine. I show both readings rather than pick one. A reading that uses the 2-Bed Standard base
   (22,425,000) plus the corner premium gives 25,340,250.
2. **Booking amount under a discount plan.** The price list defines the 10% token only for the standard plan. I take 10% of the *discounted* price and
   say that management should confirm. Taking 10% of the undiscounted base would give 3,948,000 instead.
3. Discounts and premiums are both applied to the **base** price and added (not compounded), because the price list says premiums are cumulative and not discounted.
4. The price list is dated 1 April 2025 and "subject to revision"; every price answer says so.
5. It is **rule-based plus lexical retrieval, not an LLM.** That makes it deterministic and runnable offline, but phrasing it doesn't anticipate can
   fall through to "not found" (safe, but unhelpful). Retrieval can also pick a tangential passage on multi-topic questions; it shows up to 3 passages, each tagged with its section.
   Next step: use an LLM only for *parsing the question* and *phrasing*, keeping the same calculator and refusal rules underneath.

---

## Part 2 — Database

`schema.sql`: **one table**, `leads`, because each row is one lead and every column is a single-valued attribute (no repeating groups). Typed columns with
`CHECK` constraints (flags in {0,1}, non-negative counts, positive budget), `lead_id` primary key, indexes on source / created_at / hash.
`city` is stored **raw** on purpose so nothing is lost; cleaning happens in the query. In production I'd add `source` and `city` lookup tables with FKs to stop messy spellings at write time.
(`crm_record_hash` is `BIGINT` in PostgreSQL, since values reach ~1e10.)

`queries.sql` (SQLite syntax, runnable via `run_sql.py`; reporting queries read the de-duplicated view):
1. **Conversion by source**, `HAVING COUNT(*) >= 200`, best first. Referral 13.2%, Walk-in 10.3% … Expo Stall 3.0%.
2. **Duplicates:** same `crm_record_hash`. The dump has **160 exact pairs**, identical except the copy's `lead_id` ends in `-B`. Rule: `ROW_NUMBER() OVER (PARTITION BY hash ORDER BY created_at, lead_id)`, winner = row 1
   (earliest; ties → smallest id, so the original beats its `-B` copy; deterministic). Includes a sanity query (160 groups → 160 kept → 160 dropped), the `DELETE`, and the prevention: **`UNIQUE` index on the hash + `INSERT … ON CONFLICT DO NOTHING`**
   (commented in `schema.sql`; it can only be created *after* cleaning the dump).
3. **Conversion by city** with `CASE` normalisation (`ISB→Islamabad`, `RWP→Rawalpindi`, `KHI→Karachi`, everything else Title-Cased after `UPPER(TRIM())`): 21 raw spellings collapse to 9 cities, and the counts sum to 9,000.

Note: the CSV has no agent column, so "entered by different agents" can't be verified; I identify duplicates by the hash.

---

## Part 3 — ML lead scoring

### Data decisions
| Decision | Why |
|---|---|
| **De-duplicate first** (160 pairs), before splitting | Otherwise one copy can land in train and its twin in test, which flatters the score. |
| **Drop `token_amount_received_pkr`** | **Target leak.** It is > 0 for 100% of converters, because paying the token *is* the conversion. Keeping it gives AUC 0.996 (shown below). It is useless for ranking leads that haven't paid yet. |
| Drop `lead_id`, `crm_record_hash` | Identifiers, not behaviour. |
| Drop `area` | Ten near-equal categories (5.8%–9.5% conversion, ~850 leads each: within noise). More columns, no signal. |
| `created_at` used **only to order the split** | A raw timestamp doesn't generalise to future dates. |
| Normalise city (same rules as the SQL) | 21 spellings → 9 cities. |
| Missing values: median-impute **and add a "was missing" flag** | `bedrooms` is empty for 39% of rows (plots/shops: structurally missing, itself informative; they convert more). |
| `log1p` on budget, first-response time, call seconds | Heavy right tails (response time reaches 3,107 min). |
| Keep calls / WhatsApp / site visits | Legitimate *engagement so far* signals, but they accrue over time; see the ablation below. |

### Split: by time, not randomly
Sorted by `created_at`; **oldest 80% train (7,200 leads, 1 Jan 2024 → 27 May 2025), newest 20% test (1,800 leads, 28 May → 1 Oct 2025)**.
In real life the model scores *tomorrow's* leads using *yesterday's* history. A random split lets the model see the future (and neighbours of each test row)
and flatters the result, whereas a time split mimics deployment and would expose drift. (Honest note: I did not measure a gap between random and time splits on this data, so I make no claim about its size here.)

### Model and metric
Baseline **logistic regression** (no tuning), one-hot categoricals + scaled numerics. I also tried gradient boosting on the same split and it was no better
(AUC 0.84 vs 0.86), so I kept the simpler, explainable model. Probabilities are left unweighted so they stay roughly calibrated: on the test set predicted and actual conversion rates match closely by band (e.g. 28% predicted vs 28% actual in the 20–40% band), though the model is somewhat *under*-confident at the very top (56% predicted vs 72% actual for scores above 40%).

| | Model | No-skill baseline |
|---|---|---|
| ROC-AUC (test) | **0.858** | 0.500 |
| Conversions in the **top 100 scored leads** | **49** (bootstrap 95% CI 39–59) | **≈ 7.9** (100 × the 7.9% test base rate) |
| Lift | **6.2×** | 1× |

**The sales number:** out of 1,800 recent leads (143 converted), calling only the top 100 would have reached **49 conversions, 34% of all of them**, versus about 8 from calling 100 at random.

Sensitivity checks (printed by `python -m ml.train`):
* **At-creation features only** (no calls/WhatsApp/visits): AUC 0.784, top-100 → 36. Still ~4.5× random, so the model is useful even before anyone has called.
* **With the leaking token column:** AUC 0.996, top-100 → 97. This is the "too good to be true" result that dropping it avoids.

### Caveats
* The CI only reflects sampling noise in the 143 test conversions, not model-training variance or future drift.
* "Top 100" depends on pool size: 100 of 1,800 is the top 5.6%. Against 9,000+ leads, expect to scale the cut accordingly.
* Engagement features are measured *after* the lead arrives. For a brand-new lead (all zeros) use the at-creation numbers above as the honest expectation.
* Test base rate (7.9%) is higher than train (6.7%): mild drift; the data stops on 1 Oct 2025 (only 16 leads in Q4).

---

## Part 4 — Web interface
`python app.py` → one page with (1) a question box (with the six task questions as one-click examples) showing the answer and its sources, and (2) a lead form (budget, city, source, property type, visits, calls, WhatsApp, …)
returning a conversion probability and where it ranks among recent leads. Optional fields left blank are treated as missing. Same two tools are available as JSON at `POST /api/ask` and `POST /api/score`. No styling, by design.

## Known gaps / what I'd do next
* No screen recording included.
* Part 1: LLM for question parsing/phrasing on top of the existing calculator and refusal rules; evaluation set of ~50 questions.
* Part 3: re-score on a rolling window; calibrate by source; monitor drift. Decide with the sales manager whether scoring should happen at creation or after first contact.
* Part 2: lookup tables for city/source; run on PostgreSQL (syntax is portable apart from noted types).
