"""로지스틱 회귀 신뢰도 검증"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

conn = sqlite3.connect("employee.db")
df = pd.read_sql("SELECT * FROM employees", conn)
conn.close()

# 동일한 전처리
df_model = df.copy()
df_model["over_time"] = (df_model["over_time"] == "Yes").astype(int)
df_model["gender"] = (df_model["gender"] == "Male").astype(int)
df_model["married"] = (df_model["marital_status"] == "Married").astype(int)
df_model["divorced"] = (df_model["marital_status"] == "Divorced").astype(int)
df_model["travel_rarely"] = (df_model["business_travel"] == "Travel_Rarely").astype(int)
df_model["travel_frequently"] = (df_model["business_travel"] == "Travel_Frequently").astype(int)

feature_cols = [
    "age", "gender", "distance_from_home", "num_companies_worked",
    "total_working_years", "job_level", "monthly_income", "stock_option_level",
    "years_at_company", "years_in_current_role", "years_since_last_promotion",
    "years_with_curr_manager", "over_time", "environment_satisfaction",
    "job_satisfaction", "job_involvement", "relationship_satisfaction",
    "work_life_balance", "training_times_last_year",
    "married", "divorced", "travel_rarely", "travel_frequently",
]

y = df_model["attrition"]
X = df_model[feature_cols].astype(float)

continuous_cols = ["age", "distance_from_home", "num_companies_worked",
                   "total_working_years", "monthly_income",
                   "years_at_company", "years_in_current_role",
                   "years_since_last_promotion", "years_with_curr_manager"]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# =====================================================
# 검증 1: 다중공선성 (VIF)
# =====================================================
print("=" * 70)
print("검증 1: 다중공선성 (VIF)")
print("  VIF > 5: 주의 / VIF > 10: 심각한 공선성")
print("=" * 70)

X_vif = sm.add_constant(X_scaled)
vif_data = []
for i, col in enumerate(X_vif.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_vif.values, i)
    vif_data.append((col, vif))

vif_data.sort(key=lambda x: x[1], reverse=True)
for col, vif in vif_data:
    flag = " *** 주의" if vif > 5 else (" ** 경계" if vif > 3 else "")
    print(f"  {col:<30} VIF = {vif:.2f}{flag}")

# =====================================================
# 검증 2: 교차 검증 (5-Fold)
# =====================================================
print("\n" + "=" * 70)
print("검증 2: 교차 검증 (5-Fold Stratified)")
print("  학습 데이터 AUC vs 교차 검증 AUC 비교 → 과적합 확인")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")
print(f"  각 폴드 AUC: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  교차 검증 평균 AUC: {cv_scores.mean():.3f} (+-{cv_scores.std():.3f})")
print(f"  학습 데이터 AUC:    0.851")
print(f"  차이:               {0.851 - cv_scores.mean():.3f}")

if 0.851 - cv_scores.mean() > 0.05:
    print("  → 과적합 가능성 있음")
else:
    print("  → 과적합 없음 (차이 < 0.05)")

# =====================================================
# 검증 3: 상관된 변수들의 영향 확인
# =====================================================
print("\n" + "=" * 70)
print("검증 3: 경력 관련 변수 상관관계")
print("  이 변수들이 서로 비슷한 정보를 담고 있는지 확인")
print("=" * 70)

career_cols = ["total_working_years", "years_at_company",
               "years_in_current_role", "years_with_curr_manager",
               "age", "monthly_income", "job_level"]

corr = df[career_cols].corr()
print("\n  상관 행렬 (0.7 이상 = 높은 상관):")
print()
# 깔끔하게 출력
header = "                       " + "  ".join([f"{c[:8]:>8}" for c in career_cols])
print(header)
for i, col in enumerate(career_cols):
    vals = "  ".join([f"{corr.iloc[i, j]:>8.2f}" for j in range(len(career_cols))])
    print(f"  {col:<20} {vals}")

# =====================================================
# 검증 4: 경력 변수를 줄여서 재적합
# =====================================================
print("\n" + "=" * 70)
print("검증 4: 경력 변수 축소 후 결과 비교")
print("  total_working_years만 남기고 나머지 경력 변수 제거")
print("=" * 70)

reduced_features = [c for c in feature_cols
                    if c not in ["years_at_company", "years_in_current_role",
                                 "years_with_curr_manager", "job_level"]]

X_reduced = X_scaled[reduced_features]
X_reduced_const = sm.add_constant(X_reduced)

model_reduced = sm.Logit(y, X_reduced_const)
result_reduced = model_reduced.fit(disp=0, maxiter=200)

# 교차 검증
lr2 = LogisticRegression(max_iter=1000, random_state=42)
cv_scores2 = cross_val_score(lr2, X_reduced, y, cv=cv, scoring="roc_auc")

from sklearn.metrics import roc_auc_score
auc_train = roc_auc_score(y, result_reduced.predict(X_reduced_const))

print(f"  변수 수: {len(reduced_features)}개 (기존 23개)")
print(f"  학습 AUC: {auc_train:.3f} (기존 0.851)")
print(f"  교차검증 AUC: {cv_scores2.mean():.3f} (기존 {cv_scores.mean():.3f})")

# 핵심 변수 오즈비 비교
print(f"\n  주요 변수 오즈비 비교 (전체 모델 vs 축소 모델):")
compare_vars = ["over_time", "environment_satisfaction", "job_satisfaction",
                "job_involvement", "monthly_income", "total_working_years"]

full_or = np.exp(sm.Logit(y, sm.add_constant(X_scaled)).fit(disp=0, maxiter=200).params)
reduced_or = np.exp(result_reduced.params)

print(f"  {'변수':<28} {'전체 OR':>8} {'축소 OR':>8} {'변동':>8}")
for v in compare_vars:
    if v in full_or.index and v in reduced_or.index:
        fo = full_or[v]
        ro = reduced_or[v]
        change = abs(fo - ro) / fo * 100
        flag = " ← 변동" if change > 20 else ""
        print(f"  {v:<28} {fo:>8.2f} {ro:>8.2f} {change:>7.1f}%{flag}")

# =====================================================
# 검증 5: 가상 데이터라는 한계
# =====================================================
print("\n" + "=" * 70)
print("검증 5: 데이터 자체의 한계")
print("=" * 70)
print("  이 데이터셋은 IBM이 만든 가상(fictional) 데이터입니다.")
print("  현실의 복잡성(팀 문화, 경기 상황, 스카우트 제안 등)이 반영되지 않음.")
print("  따라서 이 분석의 결론은:")
print("    O: '이런 패턴이 있다면 이런 정책이 유효하다'는 프레임워크 제시")
print("    X: '실제 회사에서 이 수치를 그대로 적용' 은 불가")
