import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "so_survey_2025.csv"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "eda_summary.json"

AGE = {"18-24 years old": 21, "25-34 years old": 29.5, "35-44 years old": 39.5,
       "45-54 years old": 49.5, "55-64 years old": 59.5, "65 years or older": 67}
EDU = {"Primary/elementary school": 0, "Secondary school": 1,
       "Some college/university study without earning a degree": 2,
       "Associate degree": 2, "Bachelor’s degree": 3, "Master’s degree": 4,
       "Professional degree": 5}
ORG = {"Just me - I am a freelancer, sole proprietor, etc.": 1,
       "Less than 20 employees": 10, "20 to 99 employees": 60,
       "100 to 499 employees": 300, "500 to 999 employees": 750,
       "1,000 to 4,999 employees": 3000, "5,000 to 9,999 employees": 7500,
       "10,000 or more employees": 15000}
REMOTE = {"In-person": 0, "Hybrid (some remote, leans heavy to in-person)": 1,
          "Your choice (very flexible, you can come in when you want or just as needed)": 2,
          "Hybrid (some in-person, leans heavy to flexibility)": 2, "Remote": 3}
LANGS = ["Python", "SQL", "JavaScript", "TypeScript", "Java", "C#", "Go", "Rust",
         "C++", "PHP", "Kotlin", "Bash/Shell (all shells)"]


def edu_level(value):
    if not isinstance(value, str):
        return np.nan
    for key, level in EDU.items():
        if value.startswith(key):
            return level
    return np.nan


def load():
    df = pd.read_csv(RAW, low_memory=False)
    pro = df[(df["MainBranch"] == "I am a developer by profession")
             & (df["Employment"] == "Employed")]
    return df, pro


def features(pro):
    x = pd.DataFrame(index=pro.index)
    x["work_exp"] = pro["WorkExp"]
    x["years_code"] = pro["YearsCode"]
    x["age"] = pro["Age"].map(AGE)
    x["edu_level"] = pro["EdLevel"].map(edu_level)
    x["org_size_log"] = np.log10(pro["OrgSize"].map(ORG))
    x["remote"] = pro["RemoteWork"].map(REMOTE)
    x["manager"] = (pro["ICorPM"] == "People manager").astype(float)
    x.loc[pro["ICorPM"].isna(), "manager"] = np.nan
    langs = pro["LanguageHaveWorkedWith"].fillna("")
    x["n_languages"] = langs.str.count(";") + (langs != "")
    x.loc[langs == "", "n_languages"] = np.nan
    for lang in LANGS:
        name = "lang_" + lang.split("/")[0].replace("#", "sharp").replace("+", "p").lower()
        x[name] = langs.str.split(";").apply(lambda items: float(lang in items))
        x.loc[langs == "", name] = np.nan
    x["salary_usd"] = pro["ConvertedCompYearly"]
    return x


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df, pro = load()
    x = features(pro)
    s = x["salary_usd"].dropna()
    q1, q99 = s.quantile([0.01, 0.99])
    clean = x[x["salary_usd"].between(q1, q99)].copy()
    clean["log_salary"] = np.log10(clean["salary_usd"])

    summary = {
        "rows_total": int(len(df)), "columns_total": int(df.shape[1]),
        "professionals_employed": int(len(pro)),
        "with_salary": int(s.size),
        "salary_q01": float(q1), "salary_q99": float(q99),
        "salary_median": float(s.median()), "salary_mean": float(s.mean()),
        "salary_min": float(s.min()), "salary_max": float(s.max()),
        "salary_skew": float(s.skew()), "log_salary_skew": float(clean["log_salary"].skew()),
        "after_outliers": int(len(clean)),
        "duplicates_response_id": int(df["ResponseId"].duplicated().sum()),
        "exp_gt_years_code": int((pro["WorkExp"] > pro["YearsCode"]).sum()),
        "missing_share": {c: float(x.loc[x["salary_usd"].notna(), c].isna().mean())
                          for c in x.columns if c != "salary_usd"},
        "countries_with_salary": int(pro.loc[pro["ConvertedCompYearly"].notna(), "Country"].nunique()),
    }

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(s[s < s.quantile(0.995)] / 1000, bins=60, color="#4c72b0")
    ax[0].set_xlabel("Годовой доход, тыс. USD")
    ax[0].set_ylabel("Число анкет")
    ax[0].set_title("Исходная шкала")
    ax[1].hist(clean["log_salary"], bins=60, color="#dd8452")
    ax[1].set_xlabel("log10(годовой доход, USD)")
    ax[1].set_title("После удаления 1 % хвостов и логарифмирования")
    fig.tight_layout()
    fig.savefig(FIG / "salary_distribution.png", dpi=180)

    miss = pd.Series(summary["missing_share"]).sort_values(ascending=True) * 100
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(miss.index, miss.values, color="#55a868")
    ax.set_xlabel("Доля пропусков среди анкет с зарплатой, %")
    fig.tight_layout()
    fig.savefig(FIG / "missing_share.png", dpi=180)

    bins = [0, 2, 5, 10, 15, 20, 30, 60]
    clean["exp_group"] = pd.cut(clean["work_exp"], bins, right=False,
                                labels=["0–1", "2–4", "5–9", "10–14", "15–19", "20–29", "30+"])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=clean.dropna(subset=["exp_group"]), x="exp_group", y="salary_usd",
                ax=ax, color="#8172b3", fliersize=1)
    ax.set_yscale("log")
    ax.set_xlabel("Опыт работы, лет")
    ax.set_ylabel("Годовой доход, USD (лог. шкала)")
    fig.tight_layout()
    fig.savefig(FIG / "salary_by_experience.png", dpi=180)
    summary["median_by_exp"] = {str(k): float(v) for k, v in
                                clean.groupby("exp_group", observed=True)["salary_usd"].median().items()}

    num = clean.drop(columns=["salary_usd", "exp_group"])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                annot=True, fmt=".2f", annot_kws={"size": 6}, cbar_kws={"shrink": 0.7})
    fig.tight_layout()
    fig.savefig(FIG / "correlation.png", dpi=180)
    summary["corr_with_log_salary"] = {k: float(v) for k, v in
                                       corr["log_salary"].drop("log_salary").sort_values().items()}
    summary["corr_work_exp_years_code"] = float(corr.loc["work_exp", "years_code"])
    summary["corr_work_exp_age"] = float(corr.loc["work_exp", "age"])

    complete = num.drop(columns=["log_salary"]).dropna()
    summary["complete_rows"] = int(len(complete))
    pca = PCA().fit(StandardScaler().fit_transform(complete))
    ratio = pca.explained_variance_ratio_
    summary["pca_ratio"] = [float(v) for v in ratio]
    summary["pca_n_features"] = int(complete.shape[1])
    summary["pca_k80"] = int(np.searchsorted(np.cumsum(ratio), 0.8) + 1)
    summary["pca_kaiser"] = int((pca.explained_variance_ > 1).sum())
    fig, ax = plt.subplots(figsize=(9, 4))
    idx = np.arange(1, len(ratio) + 1)
    ax.bar(idx, ratio * 100, color="#4c72b0", label="доля компоненты")
    ax.plot(idx, np.cumsum(ratio) * 100, "o-", color="#c44e52", label="накопленная доля")
    ax.axhline(80, ls="--", color="gray", lw=1)
    ax.set_xlabel("Номер главной компоненты")
    ax.set_ylabel("Объяснённая дисперсия, %")
    ax.set_xticks(idx)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "pca_scree.png", dpi=180)

    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"анкет всего: {summary['rows_total']}, столбцов: {summary['columns_total']}")
    print(f"профессиональные разработчики (Employed): {summary['professionals_employed']}")
    print(f"из них с указанным доходом: {summary['with_salary']}")
    print(f"доход: медиана {s.median():,.0f} USD, min {s.min():,.0f}, max {s.max():,.0f}")
    print(f"границы 1 % и 99 %: {q1:,.0f} и {q99:,.0f} USD; осталось {len(clean)}")
    print(f"асимметрия: исходная {summary['salary_skew']:.1f}, log10 {summary['log_salary_skew']:.2f}")
    print(f"полных строк для PCA: {len(complete)}, признаков: {complete.shape[1]}")
    print(f"PCA: 80 % дисперсии дают {summary['pca_k80']} компонент, "
          f"по Кайзеру {summary['pca_kaiser']}")
    print("графики:", ", ".join(p.name for p in sorted(FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
