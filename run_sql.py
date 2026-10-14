"""Load leads.csv into a SQLite file using schema.sql, then run every query in queries.sql.

    python run_sql.py            # builds mgc.db, prints the results of all queries
Only needs the Python standard library + pandas.
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DB = ROOT / "mgc.db"


def build_db() -> sqlite3.Connection:
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript((ROOT / "schema.sql").read_text())
    df = pd.read_csv(ROOT / "leads.csv")
    df.to_sql("leads", con, if_exists="append", index=False)  # columns match schema.sql 1:1
    con.commit()
    return con


def split_queries(sql: str):
    """Yield (title, statement) for each runnable statement; commented-out SQL is ignored."""
    title = ""
    buf = []
    for line in sql.splitlines():
        if line.startswith("-- Query"):
            title = line[3:].strip()
        if line.lstrip().startswith("--"):
            continue
        buf.append(line)
        if line.rstrip().endswith(";"):
            stmt = "\n".join(buf).strip()
            if stmt:
                yield title, stmt
            buf = []


if __name__ == "__main__":
    con = build_db()
    n = con.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    print(f"Loaded {n} rows into {DB.name}\n")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 40)
    for title, stmt in split_queries((ROOT / "queries.sql").read_text()):
        print("=" * 70)
        print(title)
        print(pd.read_sql_query(stmt, con).head(12).to_string(index=False))
        print()
