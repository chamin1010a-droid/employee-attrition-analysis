"""다른 공선성 그룹 확인: years 계열 변수"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

conn = sqlite3.connect("employee.db")
df = pd.read_sql("SELECT * FROM employees", conn)
conn.close()

# 전처리
df["over_time"] = (df["over_time"] == "Yes").astype(int)
df["gender"] = (df["gender"] == "Male").astype(int)
df["married"] = (df["marital_status"] == "Married").astype(int)
df["divorced"] = (df["marital_status"] == "Divorced").astype(int)
df["travel_rarely"] = (df["business_travel"] == "Travel_Rarely").astype(int)
df["travel_frequently"] = (df["business_travel"] == "Travel_Frequently").astype(int)

y = df["attrition"]

# 경력/보상 계열 제외한 기본 변수
base = [
    "age", "gender", "distance_from_home", "num_companies_worked",
    "over_time", "environment_satisfaction", "job_satisfaction",
    "job_involvement", "relationship_satisfaction", "work_life_balance",
    "training_times_last_year", "married", "divorced",
    "travel_rarely", "travel_frequently", "stock_option_level",
    "total_working_years", "monthly_income",
]

continuous_base = ["age", "distance_from_home", "num_companies_worked",
                   "total_working_years", "monthly_income"]

def run_model(features, label, interest_vars):
    X = df[features].astype(float)
    cont = [c for c in features if c in continuous_base or c.startswith("years")]
    scaler = StandardScaler()
    X_scaled = X.copy()
    if cont:
        X_scaled[cont] = scaler.fit_transform(X[cont])
    X_const = sm.add_constant(X_scaled)
    result = sm.Logit(y, X_const).fit(disp=0, maxiter=200)

    print(f"\n  [{label}]")
    for v in interest_vars:
        if v in result.params.index:
            OR = np.exp(result.params[v])
            p = result.pvalues[v]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            direction = "높임" if OR > 1 else "낮춤"
            print(f"    {v:<30} OR={OR:.2f}  p={p:.4f} {stars:>4}  ({direction})")
        else:
            print(f"    {v:<30} (없음)")

# =================================================
print("=" * 65)
print("의심 그룹: years 계열 (상관계수 0.71~0.77)")
print("=" * 65)

years_vars = ["years_at_company", "years_in_current_role",
              "years_with_curr_manager", "years_since_last_promotion"]

# 상관관계 확인
print("\n  상관 행렬:")
corr = df[years_vars].corr()
for i, c1 in enumerate(years_vars):
    for j, c2 in enumerate(years_vars):
        if j > i:
            print(f"    {c1} <-> {c2}: {corr.iloc[i,j]:.2f}")

# 원래 모델에서 결과 복기
print("\n  원래 모델(23변수) 결과:")
print("    years_at_company        OR=1.70  p=0.021  (오래 있으면 이직 높임?)")
print("    years_in_current_role   OR=0.61  p=0.002  (오래 있으면 이직 낮춤)")
print("    years_with_curr_manager OR=0.62  p=0.003  (오래 있으면 이직 낮춤)")
print("    years_since_last_promotion OR=1.77 p<0.001 (승진 못하면 이직 높임)")
print()
print("  ** years_at_company만 방향이 반대! 오래 다녔는데 이직 확률이 높다? **")

# 실험
print("\n" + "-" * 65)
print("실험: years 변수를 하나씩 넣어보기")
print("-" * 65)

run_model(base + years_vars,
          "A: 4개 전부", years_vars)

run_model(base + ["years_at_company"],
          "B: years_at_company만", ["years_at_company"])

run_model(base + ["years_in_current_role"],
          "C: years_in_current_role만", ["years_in_current_role"])

run_model(base + ["years_with_curr_manager"],
          "D: years_with_curr_manager만", ["years_with_curr_manager"])

run_model(base + ["years_since_last_promotion"],
          "E: years_since_last_promotion만", ["years_since_last_promotion"])

# 2개 조합
run_model(base + ["years_at_company", "years_since_last_promotion"],
          "F: 재직기간 + 승진후경과", 
          ["years_at_company", "years_since_last_promotion"])

# =================================================
print("\n" + "=" * 65)
print("개별 상관분석 (참고)")
print("=" * 65)
from scipy import stats
for col in years_vars:
    corr_val, pval = stats.pointbiserialr(y, df[col])
    print(f"  {col:<30} r={corr_val:+.3f}  p={pval:.2e}")
