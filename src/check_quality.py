"""Проверка требований к качеству данных (раздел 2.3) для исходной части и порций."""
import json
import sys
from pathlib import Path

import pandas as pd

from eda import AGE, ORG, REMOTE
from eda_initial import AGE_MAX, SAL_MAX, SAL_MIN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "quality"

REQUIRED = ["ResponseId", "MainBranch", "Employment", "Age", "EdLevel", "WorkExp", "YearsCode",
            "OrgSize", "RemoteWork", "ICorPM", "DevType", "Country", "LanguageHaveWorkedWith",
            "CompTotal", "Currency", "ConvertedCompYearly"]
KNOWN = {
    "Age": set(AGE) | {"Prefer not to say"},
    "OrgSize": set(ORG) | {"I don’t know"},
    "RemoteWork": set(REMOTE),
    "ICorPM": {"Individual contributor", "People manager"},
}
EDU_PREFIX = ("Primary", "Secondary", "Some college", "Associate", "Bachelor", "Master",
              "Professional", "Other")
KEY = ["WorkExp", "YearsCode", "Age", "EdLevel", "OrgSize", "RemoteWork"]
MIN_FILLED = 10
MIN_WITH_SALARY = 1000


def check(df, seen_ids=frozenset()):
    rep = {"rows": int(len(df)), "errors": [], "warnings": [], "rules": {}}
    miss = [c for c in REQUIRED if c not in df.columns]
    rep["rules"]["DQ-1 схема"] = len(miss)
    if miss:
        rep["errors"].append(f"нет столбцов: {miss}")
        return rep
    ids = df["ResponseId"]
    bad_ids = int(ids.isna().sum() + ids.duplicated().sum() + ids.isin(seen_ids).sum())
    rep["rules"]["DQ-2 ResponseId"] = bad_ids
    if bad_ids:
        rep["errors"].append(f"ResponseId пустые или повторяются: {bad_ids}")

    filled = df.drop(columns="ResponseId").notna().sum(axis=1)
    rep["rules"]["DQ-3 пустые анкеты"] = int((filled <= MIN_FILLED).sum())

    pro = df[(df["MainBranch"] == "I am a developer by profession") & (df["Employment"] == "Employed")]
    s = pro["ConvertedCompYearly"]
    out = int((s.notna() & ~s.between(SAL_MIN, SAL_MAX)).sum())
    rep["rules"]["DQ-4 доход вне диапазона"] = out
    if out:
        rep["warnings"].append(f"доход вне [{SAL_MIN}; {SAL_MAX}]: {out} записей исключается")

    rng = int((~pro["WorkExp"].between(0, 60) & pro["WorkExp"].notna()).sum()
              + (~pro["YearsCode"].between(0, 60) & pro["YearsCode"].notna()).sum())
    rep["rules"]["DQ-5 стаж вне [0; 60]"] = rng
    impossible = int((pro["WorkExp"] > pro["Age"].map(AGE_MAX) - 14).sum())
    rep["rules"]["DQ-6 стаж больше возраста"] = impossible
    if rng or impossible:
        rep["warnings"].append(f"стаж заменяется пропуском: {rng + impossible}")

    unknown = {}
    for col, values in KNOWN.items():
        v = pro[col].dropna()
        unknown[col] = float((~v.isin(values)).mean()) if len(v) else 0.0
    v = pro["EdLevel"].dropna()
    unknown["EdLevel"] = float((~v.str.startswith(EDU_PREFIX)).mean()) if len(v) else 0.0
    rep["rules"]["DQ-7 неизвестные категории, макс. доля"] = round(max(unknown.values()), 4)
    if max(unknown.values()) > 0.05:
        rep["errors"].append(f"много неизвестных категорий: {unknown}")

    with_s = pro[s.between(SAL_MIN, SAL_MAX)]
    key_miss = with_s[KEY].isna().mean()
    rep["rules"]["DQ-8 пропуски ключевых признаков, макс. доля"] = round(float(key_miss.max()), 4)
    if key_miss.max() > 0.20:
        rep["errors"].append(f"пропусков больше 20 %: {key_miss[key_miss > 0.2].to_dict()}")
    elif key_miss.max() > 0.05:
        rep["warnings"].append(f"пропусков больше 5 %: {key_miss[key_miss > 0.05].to_dict()}")

    rep["rules"]["DQ-9 анкет с доходом"] = int(len(with_s))
    if len(with_s) < MIN_WITH_SALARY:
        rep["warnings"].append(f"мало анкет с доходом ({len(with_s)}), показатели помечаются как неточные")
    rep["status"] = "отклонена" if rep["errors"] else "принята"
    return rep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [ROOT / "data" / "initial" / "initial.csv"] + sorted((ROOT / "data" / "incoming").glob("batch_*.csv"))
    seen = set()
    report = {}
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        rep = check(df, frozenset(seen))
        seen |= set(df["ResponseId"].dropna())
        report[f.name] = rep
        print(f"{f.name}: {rep['status']}, {rep['rules']}")
    (OUT / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if any(r["errors"] for r in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
