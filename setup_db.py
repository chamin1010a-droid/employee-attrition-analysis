"""
setup_db.py
CSV -> SQLite DB 변환 + 데이터 정제

원본: IBM HR Analytics Employee Attrition & Performance (Kaggle)
- WA_Fn-UseC_-HR-Employee-Attrition.csv: 1,470건, 35개 컬럼
- IBM이 만든 가상 데이터셋 (직원 이직 예측 연습용)
"""
import sqlite3
import pandas as pd

CSV_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
DB_PATH = "employee.db"


def load_and_clean(csv_path):
    """CSV 로드 + 정제"""
    df = pd.read_csv(csv_path)
    original_count = len(df)

    # 1. 상수 컬럼 제거 (모든 행이 같은 값)
    #    EmployeeCount=1, Over18='Y', StandardHours=80
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_cols)
    print(f"[정제] 상수 컬럼 제거: {constant_cols}")

    # 2. EmployeeNumber는 ID이므로 분석에서 제외하지만 DB에는 보관
    #    (인덱스 역할)

    # 3. 타겟 변환: Attrition (Yes/No) -> attrition (1/0)
    df["attrition"] = (df["Attrition"] == "Yes").astype(int)
    df = df.drop(columns=["Attrition"])

    # 4. 컬럼명 snake_case로 통일
    df.columns = [camel_to_snake(col) for col in df.columns]

    print(f"[정제] 원본: {original_count}건 -> 정제 후: {len(df)}건")
    print(f"[정제] 컬럼 수: 35 -> {len(df.columns)}개 (상수 3개 제거, attrition 변환)")
    print(f"[정제] 이직률: {df['attrition'].mean()*100:.1f}%")

    return df


def camel_to_snake(name):
    """CamelCase -> snake_case"""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def create_lookup_tables(conn):
    """만족도/교육 등급 코드표 생성"""

    # 교육 수준
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS education_level (
            id INTEGER PRIMARY KEY,
            description TEXT
        );
        INSERT OR REPLACE INTO education_level VALUES
            (1, 'Below College'),
            (2, 'College'),
            (3, 'Bachelor'),
            (4, 'Master'),
            (5, 'Doctor');
    """)

    # 환경 만족도
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS satisfaction_level (
            id INTEGER PRIMARY KEY,
            description TEXT
        );
        INSERT OR REPLACE INTO satisfaction_level VALUES
            (1, 'Low'),
            (2, 'Medium'),
            (3, 'High'),
            (4, 'Very High');
    """)

    # 워라밸
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS work_life_balance_level (
            id INTEGER PRIMARY KEY,
            description TEXT
        );
        INSERT OR REPLACE INTO work_life_balance_level VALUES
            (1, 'Bad'),
            (2, 'Good'),
            (3, 'Better'),
            (4, 'Best');
    """)

    # 성과 등급
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS performance_level (
            id INTEGER PRIMARY KEY,
            description TEXT
        );
        INSERT OR REPLACE INTO performance_level VALUES
            (1, 'Low'),
            (2, 'Good'),
            (3, 'Excellent'),
            (4, 'Outstanding');
    """)

    print("[코드표] education_level, satisfaction_level, work_life_balance_level, performance_level 생성")


def print_summary(conn):
    """DB 요약 출력"""
    cur = conn.cursor()

    # 전체 건수
    total = cur.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    attrition = cur.execute("SELECT COUNT(*) FROM employees WHERE attrition = 1").fetchone()[0]
    print(f"\n=== DB 요약 ===")
    print(f"전체: {total}건")
    print(f"이직: {attrition}건 ({attrition/total*100:.1f}%)")
    print(f"잔류: {total - attrition}건 ({(total-attrition)/total*100:.1f}%)")

    # 부서별
    print(f"\n--- 부서별 이직률 ---")
    rows = cur.execute("""
        SELECT department,
               COUNT(*) as cnt,
               SUM(attrition) as left_cnt,
               ROUND(AVG(attrition)*100, 1) as attrition_pct
        FROM employees
        GROUP BY department
        ORDER BY attrition_pct DESC
    """).fetchall()
    for dept, cnt, left, pct in rows:
        print(f"  {dept}: {cnt}명 중 {left}명 이직 ({pct}%)")

    # 직무별
    print(f"\n--- 직무별 이직률 (상위 5) ---")
    rows = cur.execute("""
        SELECT job_role,
               COUNT(*) as cnt,
               SUM(attrition) as left_cnt,
               ROUND(AVG(attrition)*100, 1) as attrition_pct
        FROM employees
        GROUP BY job_role
        ORDER BY attrition_pct DESC
        LIMIT 5
    """).fetchall()
    for role, cnt, left, pct in rows:
        print(f"  {role}: {cnt}명 중 {left}명 이직 ({pct}%)")


def main():
    # 1. CSV 로드 + 정제
    df = load_and_clean(CSV_PATH)

    # 2. SQLite DB 생성
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("employees", conn, if_exists="replace", index=False)
    print(f"\n[DB] employees 테이블 생성 -> {DB_PATH}")

    # 3. 코드표 생성
    create_lookup_tables(conn)

    # 4. 요약 출력
    print_summary(conn)

    conn.close()
    print(f"\n[완료] {DB_PATH} 생성됨")


if __name__ == "__main__":
    main()
