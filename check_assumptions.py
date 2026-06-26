"""정규성 검정 + 분포 확인"""
import sqlite3
import pandas as pd
from scipy import stats

conn = sqlite3.connect("employee.db")
df = pd.read_sql("SELECT * FROM employees", conn)

# point-biserial에서 중요한 숫자형 변수들의 정규성 검정
key_cols = [
    "total_working_years", "monthly_income", "age",
    "years_in_current_role", "years_with_curr_manager",
    "years_at_company", "job_level", "stock_option_level",
    "distance_from_home", "hourly_rate",
]

print("=" * 70)
print("1. 정규성 검정 (Shapiro-Wilk) - 전체 데이터")
print("   H0: 정규분포를 따른다 / p < 0.05면 정규분포 아님")
print("=" * 70)

for col in key_cols:
    # Shapiro-Wilk는 5000개 이하에서 작동, 우리 데이터는 1470개라 OK
    stat, pval = stats.shapiro(df[col])
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    print(f"  {col:<28} p={pval:.2e}  skew={skew:+.2f}  kurtosis={kurt:+.2f}  "
          f"{'정규분포 아님' if pval < 0.05 else '정규분포'}")

print()
print("=" * 70)
print("2. 각 그룹별(이직/잔류) 정규성")
print("=" * 70)

for col in key_cols:
    stayed = df[df["attrition"] == 0][col]
    left = df[df["attrition"] == 1][col]
    _, p_stayed = stats.shapiro(stayed)
    _, p_left = stats.shapiro(left)
    print(f"  {col:<28} 잔류 p={p_stayed:.2e} | 이직 p={p_left:.2e}")

print()
print("=" * 70)
print("3. 등분산 검정 (Levene) - 두 그룹의 분산이 같은가")
print("=" * 70)

for col in key_cols:
    stayed = df[df["attrition"] == 0][col]
    left = df[df["attrition"] == 1][col]
    stat, pval = stats.levene(stayed, left)
    print(f"  {col:<28} p={pval:.4f}  {'등분산 아님' if pval < 0.05 else '등분산 OK'}")

print()
print("=" * 70)
print("4. 분포 형태 (히스토그램 대용 - 사분위수)")
print("=" * 70)

for col in ["monthly_income", "total_working_years", "age", "years_at_company"]:
    print(f"\n  --- {col} ---")
    print(f"  전체: {df[col].describe()[['25%','50%','75%']].to_dict()}")
    print(f"  잔류: {df[df['attrition']==0][col].describe()[['25%','50%','75%']].to_dict()}")
    print(f"  이직: {df[df['attrition']==1][col].describe()[['25%','50%','75%']].to_dict()}")

print()
print("=" * 70)
print("5. 비모수 검정 (Mann-Whitney U) - 정규분포 가정 불필요")
print("   point-biserial과 결과가 같은지 비교")
print("=" * 70)

print(f"  {'컬럼':<28} {'U-stat':>12} {'p-value':>12} {'유의미':>6}  {'pb와 일치':>8}")
print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*6}  {'-'*8}")

# point-biserial 결과도 같이 계산
for col in key_cols:
    stayed = df[df["attrition"] == 0][col]
    left = df[df["attrition"] == 1][col]

    # Mann-Whitney U (비모수)
    u_stat, u_pval = stats.mannwhitneyu(stayed, left, alternative="two-sided")

    # Point-biserial (모수)
    pb_corr, pb_pval = stats.pointbiserialr(df["attrition"], df[col])

    u_sig = "***" if u_pval < 0.001 else ("**" if u_pval < 0.01 else ("*" if u_pval < 0.05 else ""))
    pb_sig = "***" if pb_pval < 0.001 else ("**" if pb_pval < 0.01 else ("*" if pb_pval < 0.05 else ""))
    match = "일치" if u_sig == pb_sig else "불일치"

    print(f"  {col:<28} {u_stat:>12.0f} {u_pval:>12.2e} {u_sig:>6}  {match:>8}")

conn.close()
