"""다중공선성의 실제 영향 확인: 월급/직급을 하나만 넣으면?"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

conn = sqlite3.connect("employee.db")
df = pd.read_sql("SELECT * FROM employees", conn)
conn.close()

# 전처리 (동일)
df["over_time"] = (df["over_time"] == "Yes").astype(int)
df["gender"] = (df["gender"] == "Male").astype(int)
df["married"] = (df["marital_status"] == "Married").astype(int)
df["divorced"] = (df["marital_status"] == "Divorced").astype(int)
df["travel_rarely"] = (df["business_travel"] == "Travel_Rarely").astype(int)
df["travel_frequently"] = (df["business_travel"] == "Travel_Frequently").astype(int)

y = df["attrition"]

base_features = [
    "age", "gender", "distance_from_home", "num_companies_worked",
    "over_time", "environment_satisfaction", "job_satisfaction",
    "job_involvement", "relationship_satisfaction", "work_life_balance",
    "training_times_last_year", "married", "divorced",
    "travel_rarely", "travel_frequently",
    "stock_option_level", "years_at_company", "years_in_current_role",
    "years_since_last_promotion", "years_with_curr_manager",
]

continuous = ["age", "distance_from_home", "num_companies_worked",
              "years_at_company", "years_in_current_role",
              "years_since_last_promotion", "years_with_curr_manager"]

def run_model(features, label):
    X = df[features].astype(float)
    cont = [c for c in continuous if c in features]
    # 추가 연속형
    for c in ["total_working_years", "monthly_income"]:
        if c in features and c not in cont:
            cont.append(c)
    scaler = StandardScaler()
    X_scaled = X.copy()
    if cont:
        X_scaled[cont] = scaler.fit_transform(X[cont])
    X_const = sm.add_constant(X_scaled)
    result = sm.Logit(y, X_const).fit(disp=0, maxiter=200)

    # 관심 변수만 출력
    interest = ["monthly_income", "job_level", "total_working_years"]
    print(f"\n  [{label}]")
    for v in interest:
        if v in result.params.index:
            OR = np.exp(result.params[v])
            p = result.pvalues[v]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"    {v:<25} OR={OR:.2f}  p={p:.4f} {stars}")
        else:
            print(f"    {v:<25} (모델에 없음)")

# 모델 A: 둘 다 넣기 (원래 모델)
print("=" * 60)
print("실험: monthly_income과 job_level을 하나만 넣으면?")
print("=" * 60)

features_both = base_features + ["monthly_income", "job_level", "total_working_years"]
run_model(features_both, "모델 A: 월급 + 직급 + 총경력 - 전부 넣기")

# 모델 B: 월급만
features_income = base_features + ["monthly_income", "total_working_years"]
run_model(features_income, "모델 B: 월급 + 총경력 (직급 제거)")

# 모델 C: 직급만
features_level = base_features + ["job_level", "total_working_years"]
run_model(features_level, "모델 C: 직급 + 총경력 (월급 제거)")

# 모델 D: 월급만 (총경력도 제거)
features_income_only = base_features + ["monthly_income"]
run_model(features_income_only, "모델 D: 월급만 (직급, 총경력 모두 제거)")

# 모델 E: 총경력만
features_years_only = base_features + ["total_working_years"]
run_model(features_years_only, "모델 E: 총경력만 (월급, 직급 제거)")

print("\n" + "=" * 60)
print("개별 상관분석 복기 (참고)")
print("=" * 60)
from scipy import stats
for col in ["monthly_income", "job_level", "total_working_years"]:
    corr, pval = stats.pointbiserialr(y, df[col])
    print(f"  {col:<25} r={corr:+.3f}  p={pval:.2e}")
