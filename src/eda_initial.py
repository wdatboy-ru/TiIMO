"""Разведочный анализ исходной части (data/initial/initial.csv, 60 % анкет)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from eda import AGE, ORG, REMOTE, features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "initial" / "initial.csv"
OUT = ROOT / "reports" / "eda_initial"
FIG = OUT / "figures"

AGE_MAX = {"18-24 years old": 24, "25-34 years old": 34, "35-44 years old": 44,
           "45-54 years old": 54, "55-64 years old": 64, "65 years or older": 100}
SAL_MIN, SAL_MAX = 1_000, 1_000_000
MIN_COUNTRY = 30
ROLES = {
    "Бэкенд/фулстек": ["Developer, back-end", "Developer, full-stack"],
    "Фронтенд/мобайл": ["Developer, front-end", "Developer, mobile"],
    "Десктоп/встраиваемые/игры": ["Developer, desktop or enterprise applications",
                                  "Developer, embedded applications or devices",
                                  "Developer, game or graphics"],
    "Данные и ИИ": ["Data engineer", "Data scientist", "AI/ML engineer",
                    "Data or business analyst"],
    "DevOps/облака/ИБ": ["DevOps engineer or professional", "Cloud infrastructure engineer",
                         "System administrator", "Cybersecurity or InfoSec professional"],
    "Архитектор": ["Architect, software or solutions"],
    "Руководитель": ["Engineering manager", "Senior executive (C-suite, VP, etc.)",
                     "Project manager", "Product manager"],
    "Тестирование": ["Developer, QA or test"],
}


def role_group(value):
    for name, items in ROLES.items():
        if value in items:
            return name
    return np.nan if pd.isna(value) else "Другое"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=180)
    plt.close(fig)


def slope_ci(x, y, rng, n_boot=500):
    b = np.polyfit(x, y, 1)[0]
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boot.append(np.polyfit(x[i], y[i], 1)[0])
    return float(b), [float(v) for v in np.percentile(boot, [2.5, 97.5])]


def kmo(corr):
    inv = np.linalg.inv(corr)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    partial = -inv / d
    np.fill_diagonal(partial, 0)
    r = corr.copy()
    np.fill_diagonal(r, 0)
    return float((r ** 2).sum() / ((r ** 2).sum() + (partial ** 2).sum()))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    df = pd.read_csv(DATA, low_memory=False)
    res = {"rows": int(len(df)), "columns": int(df.shape[1]),
           "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()}}

    # --- структура и целевая переменная
    pro = df[(df["MainBranch"] == "I am a developer by profession")
             & (df["Employment"] == "Employed")].copy()
    s = pro["ConvertedCompYearly"]
    res["pro"] = int(len(pro))
    res["pro_with_salary"] = int(s.notna().sum())
    res["salary_missing_share"] = float(s.isna().mean())
    res["salary_quantiles"] = {str(q): float(v) for q, v in
                               s.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]).items()}
    res["salary_mean"] = float(s.mean())
    res["salary_skew"] = float(s.skew())
    res["salary_below_min"] = int((s < SAL_MIN).sum())
    res["salary_above_max"] = int((s > SAL_MAX).sum())
    good = pro[s.between(SAL_MIN, SAL_MAX)].copy()
    good["log_salary"] = np.log10(good["ConvertedCompYearly"])
    res["after_bounds"] = int(len(good))
    res["log_salary_skew"] = float(good["log_salary"].skew())
    res["log_salary_kurt"] = float(good["log_salary"].kurt())
    us_share = lambda part: float((part["Country"].dropna() == "United States of America").mean())
    res["us_share_with_salary"] = us_share(pro[s.notna()])
    res["us_share_without_salary"] = us_share(pro[s.isna()])
    res["no_salary_country_missing"] = float(pro.loc[s.isna(), "Country"].isna().mean())

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(s[s < s.quantile(0.995)] / 1000, bins=60, color="#4c72b0")
    ax[0].set_xlabel("Годовой доход, тыс. USD")
    ax[0].set_ylabel("Число анкет")
    ax[0].set_title("Исходная шкала (без 0,5 % верхних)")
    ax[1].hist(good["log_salary"], bins=60, color="#dd8452")
    ax[1].set_xlabel("log10(годовой доход, USD)")
    ax[1].set_title(f"Доход от {SAL_MIN:,} до {SAL_MAX:,} USD".replace(",", " "))
    save(fig, "target_distribution.png")

    # --- качество данных
    q = {}
    q["dup_response_id"] = int(df["ResponseId"].duplicated().sum())
    answers = df.drop(columns="ResponseId")
    dup = answers.duplicated(keep=False)
    q["dup_rows"] = int(answers.duplicated().sum())
    q["dup_rows_pro"] = int(pro.drop(columns="ResponseId").duplicated().sum())
    q["dup_filled_median"] = float(answers[dup].notna().sum(axis=1).median())
    q["filled_median"] = float(answers.notna().sum(axis=1).median())
    q["abandoned_le10"] = int((answers.notna().sum(axis=1) <= 10).sum())
    q["comp_total_le0"] = int((pro["CompTotal"] <= 0).sum())
    q["comp_total_gt_1e9"] = int((pro["CompTotal"] > 1e9).sum())
    q["comp_without_converted"] = int((pro["CompTotal"].notna() & s.isna()).sum())
    age_max = pro["Age"].map(AGE_MAX)
    q["work_exp_gt_age"] = int((pro["WorkExp"] > age_max - 14).sum())
    q["years_code_gt_age"] = int((pro["YearsCode"] > age_max - 5).sum())
    q["work_exp_gt_years_code"] = int((pro["WorkExp"] > pro["YearsCode"]).sum())
    q["work_exp_gt_60"] = int((pro["WorkExp"] > 60).sum())
    q["years_code_gt_60"] = int((pro["YearsCode"] > 60).sum())
    q["age_prefer_not"] = int((pro["Age"] == "Prefer not to say").sum())
    q["edlevel_other"] = int(pro["EdLevel"].str.startswith("Other", na=False).sum())
    q["tool_count_gt_100"] = int((pro["ToolCountWork"] > 100).sum())
    res["quality"] = q

    key = ["WorkExp", "YearsCode", "Age", "EdLevel", "OrgSize", "RemoteWork", "ICorPM",
           "DevType", "Industry", "LanguageHaveWorkedWith", "Country", "JobSat"]
    miss_all = df[key].isna().mean()
    miss_pro = good[key].isna().mean()
    res["missing_all"] = {k: float(v) for k, v in miss_all.items()}
    res["missing_pro_salary"] = {k: float(v) for k, v in miss_pro.items()}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m = pd.DataFrame({"все анкеты": miss_all, "профи с доходом": miss_pro}).sort_values("все анкеты") * 100
    m.plot.barh(ax=ax, color=["#bbbbbb", "#55a868"])
    ax.set_xlabel("Доля пропусков, %")
    save(fig, "missing_share.png")

    # --- признаки и нормировка по стране
    x = features(good)
    x["log_salary"] = good["log_salary"]
    x["country"] = good["Country"]
    x["role"] = good["DevType"].map(role_group)
    x["age_cat"] = good["Age"]
    x["org_cat"] = good["OrgSize"]
    x["remote_cat"] = good["RemoteWork"]
    x["icpm"] = good["ICorPM"]
    counts = x["country"].value_counts()
    big = counts[counts >= MIN_COUNTRY].index
    x["country_grp"] = x["country"].where(x["country"].isin(big), "Другие")
    country_med = x.groupby("country_grp")["log_salary"].median()
    x["rel_salary"] = x["log_salary"] - x["country_grp"].map(country_med)
    res["countries"] = int(x["country"].nunique())
    res["countries_big"] = int(len(big))
    res["share_in_big_countries"] = float(x["country"].isin(big).mean())
    res["top_countries"] = {k: int(v) for k, v in counts.head(10).items()}
    res["country_median_usd"] = {k: float(10 ** v) for k, v in
                                 country_med.sort_values(ascending=False).items()}

    top = counts.head(15).index
    fig, ax = plt.subplots(figsize=(9, 5.5))
    t = x[x["country"].isin(top)].copy()
    t["country"] = t["country"].replace({
        "United States of America": "США",
        "United Kingdom of Great Britain and Northern Ireland": "Великобритания",
        "Germany": "Германия", "India": "Индия", "France": "Франция", "Canada": "Канада",
        "Ukraine": "Украина", "Netherlands": "Нидерланды", "Brazil": "Бразилия",
        "Italy": "Италия", "Australia": "Австралия", "Poland": "Польша", "Spain": "Испания",
        "Sweden": "Швеция", "Switzerland": "Швейцария", "Austria": "Австрия",
        "Russian Federation": "Россия", "Czech Republic": "Чехия"})
    order = t.groupby("country")["log_salary"].median().sort_values().index
    sns.boxplot(data=t, y="country", x="log_salary", order=order, ax=ax, color="#4c72b0", fliersize=1)
    ax.set_xlabel("log10(годовой доход, USD)")
    ax.set_ylabel("")
    save(fig, "salary_by_country.png")

    # распределения категориальных признаков
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    org_ru = {"Just me - I am a freelancer, sole proprietor, etc.": "1 (фриланс)",
              "Less than 20 employees": "<20", "20 to 99 employees": "20–99",
              "100 to 499 employees": "100–499", "500 to 999 employees": "500–999",
              "1,000 to 4,999 employees": "1000–4999", "5,000 to 9,999 employees": "5000–9999",
              "10,000 or more employees": "10 000+"}
    edu_ru = {"Bachelor": "бакалавр", "Master": "магистр", "Some college": "неоконченное высшее",
              "Professional": "PhD и др.", "Secondary": "среднее", "Associate": "associate degree",
              "Other": "другое", "Primary": "начальное"}
    short = {"Age": lambda v: v.replace(" years old", "").replace("65 years or older", "65+").replace("Prefer not to say", "не указан"),
             "OrgSize": lambda v: org_ru.get(v, "не знаю"),
             "RemoteWork": lambda v: {"In-person": "в офисе", "Remote": "удалённо",
                                      "Hybrid (some remote, leans heavy to in-person)": "гибрид, больше офис",
                                      "Hybrid (some in-person, leans heavy to flexibility)": "гибрид, больше дома",
                                      "Your choice (very flexible, you can come in when you want or just as needed)": "на выбор"}.get(v, v),
             "EdLevel": lambda v: next((r for k, r in edu_ru.items() if v.startswith(k)), v)}
    titles = {"Age": "Возраст", "EdLevel": "Образование", "OrgSize": "Размер организации",
              "RemoteWork": "Формат работы"}
    res["category_shares"] = {}
    for a, col in zip(axes.ravel(), ["Age", "EdLevel", "OrgSize", "RemoteWork"]):
        vc = good[col].dropna().map(short[col]).value_counts(normalize=True) * 100
        res["category_shares"][col] = {k: float(v) for k, v in vc.items()}
        vc.sort_values().plot.barh(ax=a, color="#8172b3")
        a.set_title(titles[col])
        a.set_xlabel("% анкет")
        a.set_ylabel("")
    save(fig, "categorical_overview.png")
    res["work_exp_median"] = float(good["WorkExp"].median())
    res["work_exp_skew"] = float(good["WorkExp"].skew())
    res["n_languages_median"] = float(x["n_languages"].median())

    # --- взаимосвязи
    num_cols = ["work_exp", "years_code", "age", "edu_level", "org_size_log", "remote",
                "manager", "n_languages", "lang_python", "lang_javascript", "lang_php",
                "lang_go", "lang_rust", "lang_bash", "log_salary", "rel_salary"]
    corr = x[num_cols].corr(method="spearman")
    res["spearman_log_salary"] = {k: float(v) for k, v in corr["log_salary"].drop(["log_salary", "rel_salary"]).items()}
    res["spearman_rel_salary"] = {k: float(v) for k, v in corr["rel_salary"].drop(["log_salary", "rel_salary"]).items()}
    res["spearman_exp_years_code"] = float(corr.loc["work_exp", "years_code"])
    res["spearman_exp_age"] = float(corr.loc["work_exp", "age"])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.7})
    save(fig, "correlation.png")

    def cramers_v(a, b):
        tab = pd.crosstab(a, b)
        chi2 = stats.chi2_contingency(tab)[0]
        return float(np.sqrt(chi2 / tab.values.sum() / (min(tab.shape) - 1)))
    res["cramers_country_currency"] = cramers_v(good["Country"], good["Currency"])
    res["cramers_country_remote"] = cramers_v(good["Country"], good["RemoteWork"])

    pca_cols = [c for c in features(good).columns if c != "salary_usd"]
    complete = x[pca_cols].dropna()
    r = np.corrcoef(complete.values, rowvar=False)
    n, p = complete.shape
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(r))
    res["bartlett_chi2"] = float(chi2)
    res["bartlett_p"] = float(stats.chi2.sf(chi2, p * (p - 1) / 2))
    res["kmo"] = kmo(r)
    res["kmo_rows"] = int(n)
    core = ["work_exp", "years_code", "age", "edu_level", "org_size_log", "remote", "manager", "n_languages"]
    res["kmo_core"] = kmo(np.corrcoef(x[core].dropna().values, rowvar=False))

    # --- утечка
    usd = good[good["Currency"].str.startswith("USD", na=False)]
    res["leak_usd_equal_share"] = float((usd["CompTotal"] == usd["ConvertedCompYearly"]).mean())
    eur = good[good["Currency"].str.startswith("EUR", na=False) & (good["CompTotal"] > 0)]
    res["leak_eur_corr_log"] = float(np.corrcoef(np.log10(eur["CompTotal"]),
                                                 eur["log_salary"])[0, 1])
    res["leak_eur_ratio_median"] = float((eur["ConvertedCompYearly"] / eur["CompTotal"]).median())
    js = x.join(good["JobSat"])
    res["jobsat_spearman_rel_salary"] = float(js[["JobSat", "rel_salary"]].corr("spearman").iloc[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(eur["CompTotal"], eur["ConvertedCompYearly"], s=3, alpha=0.4, color="#4c72b0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CompTotal (в валюте анкеты, EUR)")
    ax.set_ylabel("ConvertedCompYearly, USD")
    save(fig, "leakage_comptotal.png")

    # --- гипотезы
    h = {}
    # H1: страна объясняет больше разброса дохода, чем опыт
    grand = x["log_salary"].mean()
    grp_means = x.groupby("country_grp")["log_salary"].transform("mean")
    h["h1_eta2_country"] = float(((grp_means - grand) ** 2).sum() / ((x["log_salary"] - grand) ** 2).sum())
    ok = x[["work_exp", "log_salary"]].dropna()
    h["h1_r2_work_exp"] = float(np.corrcoef(ok["work_exp"], ok["log_salary"])[0, 1] ** 2)
    ok = x[["work_exp", "rel_salary"]].dropna()
    h["h1_r2_work_exp_rel"] = float(np.corrcoef(ok["work_exp"], ok["rel_salary"])[0, 1] ** 2)
    h["h1_kruskal_p"] = float(stats.kruskal(*[g["log_salary"].values for _, g in x.groupby("country_grp")]).pvalue)

    # H2: после 10 лет опыта рост дохода замедляется
    ok = x[["work_exp", "rel_salary"]].dropna()
    ok = ok[ok["work_exp"] <= 40]
    lo, hi = ok[ok["work_exp"] < 10], ok[ok["work_exp"] >= 10]
    h["h2_slope_0_10"], h["h2_ci_0_10"] = slope_ci(lo["work_exp"].values, lo["rel_salary"].values, rng)
    h["h2_slope_10_40"], h["h2_ci_10_40"] = slope_ci(hi["work_exp"].values, hi["rel_salary"].values, rng)
    by_year = ok.groupby(ok["work_exp"].clip(upper=35))["rel_salary"].agg(
        ["median", lambda v: v.quantile(0.25), lambda v: v.quantile(0.75), "size"])
    by_year.columns = ["med", "q25", "q75", "n"]
    by_year = by_year[by_year["n"] >= 20]
    h["h2_rel_median_by_exp"] = {str(int(k)): float(v) for k, v in by_year["med"].items()}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(by_year.index, 10 ** by_year["q25"], 10 ** by_year["q75"], color="#8172b3", alpha=0.25,
                    label="межквартильный размах")
    ax.plot(by_year.index, 10 ** by_year["med"], "o-", color="#8172b3", ms=3, label="медиана")
    ax.axvline(10, ls="--", color="gray", lw=1)
    ax.set_xlabel("Опыт работы, лет")
    ax.set_ylabel("Доход / медиана по стране")
    ax.legend()
    save(fig, "salary_by_experience.png")

    # H3: премия за удалённую работу в основном объясняется страной
    rem = x.dropna(subset=["remote"])
    a_raw = rem.loc[rem["remote"] == 3, "log_salary"]
    b_raw = rem.loc[rem["remote"] == 0, "log_salary"]
    a_rel = rem.loc[rem["remote"] == 3, "rel_salary"]
    b_rel = rem.loc[rem["remote"] == 0, "rel_salary"]
    h["h3_ratio_raw"] = float(10 ** (a_raw.median() - b_raw.median()))
    h["h3_ratio_rel"] = float(10 ** (a_rel.median() - b_rel.median()))
    h["h3_p_raw"] = float(stats.mannwhitneyu(a_raw, b_raw).pvalue)
    h["h3_p_rel"] = float(stats.mannwhitneyu(a_rel, b_rel).pvalue)
    h["h3_us_share_remote"] = float((rem.loc[rem["remote"] == 3, "country"] == "United States of America").mean())
    h["h3_us_share_office"] = float((rem.loc[rem["remote"] == 0, "country"] == "United States of America").mean())
    labels = {0: "в офисе", 1: "гибрид,\nбольше офис", 2: "гибрид / на выбор", 3: "удалённо"}
    med = rem.groupby("remote")[["log_salary", "rel_salary"]].median()
    h["h3_by_remote"] = {labels[k].replace("\n", " "): [float(10 ** v["log_salary"]), float(10 ** v["rel_salary"])]
                         for k, v in med.iterrows()}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar([labels[k] for k in med.index], 10 ** med["log_salary"] / 1000, color="#4c72b0")
    ax[0].set_ylabel("Медиана дохода, тыс. USD")
    ax[0].set_title("Без учёта страны")
    ax[1].bar([labels[k] for k in med.index], 10 ** med["rel_salary"], color="#dd8452")
    ax[1].axhline(1, color="gray", lw=1)
    ax[1].set_ylabel("Медиана (доход / медиана по стране)")
    ax[1].set_title("С учётом страны")
    save(fig, "salary_by_remote.png")

    # H4: в крупных компаниях платят больше при той же стране
    org = x.dropna(subset=["org_size_log"])
    org = org[org["org_size_log"] > 0]  # без фрилансеров «Just me»
    rho = stats.spearmanr(org["org_size_log"], org["rel_salary"])
    h["h4_spearman"] = float(rho.statistic)
    h["h4_p"] = float(rho.pvalue)
    med = org.groupby("org_size_log")["rel_salary"].median()
    inv = {np.log10(v): k for k, v in ORG.items()}
    h["h4_by_org"] = {inv[k]: float(10 ** v) for k, v in med.items()}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = [short["OrgSize"](inv[k]) for k in med.index]
    ax.bar(names, 10 ** med.values, color="#55a868")
    ax.axhline(1, color="gray", lw=1)
    ax.set_xlabel("Размер организации, сотрудников")
    ax.set_ylabel("Доход / медиана по стране")
    save(fig, "salary_by_org.png")

    # H5: руководители людей получают больше при равном опыте
    mg = x.dropna(subset=["manager", "work_exp"])
    mg = mg.assign(exp_group=pd.cut(mg["work_exp"], [0, 5, 10, 15, 20, 30, 60], right=False,
                                    labels=["0–4", "5–9", "10–14", "15–19", "20–29", "30+"]))
    rows = {}
    for g, part in mg.groupby("exp_group", observed=True):
        a = part.loc[part["manager"] == 1, "rel_salary"]
        b = part.loc[part["manager"] == 0, "rel_salary"]
        if len(a) >= 20:
            rows[str(g)] = {"n_manager": int(len(a)), "ratio": float(10 ** (a.median() - b.median())),
                            "p": float(stats.mannwhitneyu(a, b).pvalue)}
    h["h5_by_exp"] = rows
    h["h5_manager_share"] = float(mg["manager"].mean())

    # роль (DevType) — для плана признаков
    rg = x.dropna(subset=["role"]).groupby("role")["rel_salary"].agg(["median", "size"])
    h["role_rel_median"] = {k: [float(10 ** v["median"]), int(v["size"])] for k, v in
                            rg.sort_values("median").iterrows()}
    fig, ax = plt.subplots(figsize=(9, 4))
    rg = rg.sort_values("median")
    ax.barh(rg.index, 10 ** rg["median"], color="#c44e52")
    ax.axvline(1, color="gray", lw=1)
    ax.set_xlabel("Медиана (доход / медиана по стране)")
    save(fig, "salary_by_role.png")
    res["hypotheses"] = h

    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"исходная часть: {res['rows']} анкет, профи: {res['pro']}, с доходом: {res['pro_with_salary']}")
    print(f"после границ {SAL_MIN}–{SAL_MAX}: {res['after_bounds']}")
    print(f"H1 eta2 страны {h['h1_eta2_country']:.3f}, r2 опыта {h['h1_r2_work_exp']:.3f}")
    print(f"H2 наклон 0–10: {h['h2_slope_0_10']:.4f}, 10–40: {h['h2_slope_10_40']:.4f}")
    print(f"H3 удалёнка/офис: {h['h3_ratio_raw']:.2f} -> {h['h3_ratio_rel']:.2f}")
    print(f"H4 rho {h['h4_spearman']:.3f} (p={h['h4_p']:.1e})")
    print("H5", {k: round(v['ratio'], 2) for k, v in rows.items()})
    print("графики:", ", ".join(p.name for p in sorted(FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
