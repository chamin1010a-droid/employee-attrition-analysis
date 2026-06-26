"""
logistic_regression.py
로지스틱 회귀: 이직의 독립적 요인 분리

목적:
- 각 변수가 "다른 변수를 통제했을 때" 이직에 얼마나 기여하는지
- 통제 가능 vs 불가 요인 분리
- 오즈비(Odds Ratio)로 해석
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# ===========================================
# 1. 데이터 준비
# ===========================================
conn = sqlite3.connect("employee.db")
df = pd.read_sql("SELECT * FROM employees", conn)
conn.close()

print("=" * 70)
print("1. 데이터 준비")
print("=" * 70)

# 분석 변수 선택 (의미 있는 것만)
# 제외: employee_number(ID), hourly_rate/monthly_rate/daily_rate(무관),
#       performance_rating(변별력 없음), education(무관)
# job_role은 9개 -> 더미변수가 너무 많아짐 -> job_level로 대체 (상관도 높음)
# education_field도 6개 -> 제외 (이직과 약한 관련)
# department도 job_role과 중복 -> 제외

features = {
    # 통제 불가 (직원 특성)
    "age": "나이",
    "gender": "성별",
    "marital_status": "결혼 상태",
    "distance_from_home": "통근 거리",
    "num_companies_worked": "이전 직장 수",
    "total_working_years": "총 경력",
    # 직급/보상 (부분적 통제 가능)
    "job_level": "직급",
    "monthly_income": "월급",
    "stock_option_level": "스톡옵션",
    "years_at_company": "재직 기간",
    "years_in_current_role": "현 직무 기간",
    "years_since_last_promotion": "승진 후 경과",
    "years_with_curr_manager": "현 매니저 기간",
    # 통제 가능 (회사 관리)
    "over_time": "야근 여부",
    "business_travel": "출장 빈도",
    "environment_satisfaction": "환경 만족도",
    "job_satisfaction": "직무 만족도",
    "job_involvement": "업무 몰입도",
    "relationship_satisfaction": "관계 만족도",
    "work_life_balance": "워라밸",
    "training_times_last_year": "교육 횟수",
}

df_model = df[list(features.keys()) + ["attrition"]].copy()

# 범주형 변환
df_model["over_time"] = (df_model["over_time"] == "Yes").astype(int)
df_model["gender"] = (df_model["gender"] == "Male").astype(int)

# 결혼 상태: Single을 기준으로 (이직률 가장 높음)
df_model["married"] = (df_model["marital_status"] == "Married").astype(int)
df_model["divorced"] = (df_model["marital_status"] == "Divorced").astype(int)
df_model = df_model.drop(columns=["marital_status"])

# 출장: Non-Travel 기준
df_model["travel_rarely"] = (df_model["business_travel"] == "Travel_Rarely").astype(int)
df_model["travel_frequently"] = (df_model["business_travel"] == "Travel_Frequently").astype(int)
df_model = df_model.drop(columns=["business_travel"])

# 타겟 분리
y = df_model["attrition"]
X = df_model.drop(columns=["attrition"])

print(f"  변수 수: {X.shape[1]}개")
print(f"  이직자/변수 비율: {y.sum()}/{X.shape[1]} = {y.sum()/X.shape[1]:.1f} (10 이상이면 안정적)")
print(f"  변수 목록: {list(X.columns)}")

# 연속형 변수 표준화
continuous_cols = ["age", "distance_from_home", "num_companies_worked",
                   "total_working_years", "monthly_income",
                   "years_at_company", "years_in_current_role",
                   "years_since_last_promotion", "years_with_curr_manager"]

scaler = StandardScaler()
X_scaled = X.copy().astype(float)
X_scaled[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# 상수항 추가
X_const = sm.add_constant(X_scaled)

# ===========================================
# 2. 전체 모델 적합
# ===========================================
print("\n" + "=" * 70)
print("2. 로지스틱 회귀 결과")
print("=" * 70)

model = sm.Logit(y, X_const)
result = model.fit(disp=0, maxiter=200)

print(f"  Pseudo R-squared: {result.prsquared:.3f}")
print(f"  AIC: {result.aic:.1f}")

# ===========================================
# 3. 오즈비 정리
# ===========================================
print("\n" + "=" * 70)
print("3. 전체 변수 오즈비 (Odds Ratio)")
print("   OR > 1: 이직 확률 높임 / OR < 1: 이직 확률 낮춤")
print("   표준화된 연속변수: OR = 1 표준편차 증가 시 이직 확률 변화")
print("=" * 70)

summary_df = pd.DataFrame({
    "odds_ratio": np.exp(result.params),
    "p_value": result.pvalues,
    "ci_lower": np.exp(result.conf_int()[0]),
    "ci_upper": np.exp(result.conf_int()[1]),
})
summary_df = summary_df.drop("const", errors="ignore")

# 한글 이름 매핑
name_map = {
    "age": "나이", "gender": "성별(남)", "distance_from_home": "통근거리",
    "num_companies_worked": "이전직장수", "total_working_years": "총경력",
    "job_level": "직급", "monthly_income": "월급", "stock_option_level": "스톡옵션",
    "years_at_company": "재직기간", "years_in_current_role": "현직무기간",
    "years_since_last_promotion": "승진후경과", "years_with_curr_manager": "현매니저기간",
    "over_time": "야근", "environment_satisfaction": "환경만족도",
    "job_satisfaction": "직무만족도", "job_involvement": "업무몰입도",
    "relationship_satisfaction": "관계만족도", "work_life_balance": "워라밸",
    "training_times_last_year": "교육횟수",
    "married": "기혼(vs미혼)", "divorced": "이혼(vs미혼)",
    "travel_rarely": "가끔출장(vs안함)", "travel_frequently": "잦은출장(vs안함)",
}

summary_df = summary_df.sort_values("p_value")

print(f"\n  {'변수':<20} {'한글':<15} {'OR':>6} {'95% CI':>16} {'p-value':>10} {'':>4}")
print(f"  {'-'*20} {'-'*15} {'-'*6} {'-'*16} {'-'*10} {'-'*4}")

for idx, row in summary_df.iterrows():
    stars = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
    kr = name_map.get(idx, idx)
    print(f"  {idx:<20} {kr:<15} {row['odds_ratio']:>6.2f} [{row['ci_lower']:>5.2f}, {row['ci_upper']:>5.2f}] "
          f"{row['p_value']:>10.4f} {stars:>4}")

# ===========================================
# 4. 해석 요약
# ===========================================
print("\n" + "=" * 70)
print("4. 핵심 해석 (유의미한 변수, p < 0.05)")
print("=" * 70)

sig = summary_df[summary_df["p_value"] < 0.05].sort_values("odds_ratio", ascending=False)

print("\n  [이직 확률을 높이는 요인] (OR > 1)")
for idx, row in sig[sig["odds_ratio"] > 1].iterrows():
    kr = name_map.get(idx, idx)
    print(f"    {kr}: 다른 조건 동일 시 이직 확률 {row['odds_ratio']:.1f}배")

print("\n  [이직 확률을 낮추는 요인] (OR < 1)")
for idx, row in sig[sig["odds_ratio"] < 1].iterrows():
    kr = name_map.get(idx, idx)
    reduction = (1 - row['odds_ratio']) * 100
    print(f"    {kr}: 1단위(또는 1SD) 증가 시 이직 확률 {reduction:.0f}% 감소")

# ===========================================
# 5. 모델 적합도
# ===========================================
print("\n" + "=" * 70)
print("5. 모델 적합도")
print("=" * 70)

from sklearn.metrics import roc_auc_score
y_pred_prob = result.predict(X_const)
auc = roc_auc_score(y, y_pred_prob)
print(f"  AUC-ROC (학습 데이터): {auc:.3f}")
print(f"  Pseudo R-squared: {result.prsquared:.3f}")
print(f"  유의미 변수: {len(sig)}개 / 전체 {len(summary_df)}개")
