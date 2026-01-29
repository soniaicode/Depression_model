"""
=========================================================
Preprocessing for Synthetic Depression Questionnaire Data
=========================================================
Input  : data/raw/depression_questionnaire_data.csv
Output : data/processed/*.npy + scaler.pkl + feature_names.pkl

Target Label → depression_label
0 = Low
1 = Moderate/High
=========================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ===================== PATH CONFIG ===================== #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "depression_questionnaire_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TEST_SIZE = 0.15
VAL_SIZE  = 0.15
RAND_SEED = 42

# ===================== MAPPING TABLES ===================== #

PHQ_MAP = {"Not at all":0,"Several days":1,"More than half the days":2,"Nearly every day":3}
IMPACT_MAP = {"No impact":0,"Mild":1,"Moderate":2,"Severe":3}
VITD_MAP = {"Insufficient":0,"Sufficient":1,"Optimal":2}
SOCIAL_MAP = {"Weak":0,"Moderate":1,"Strong":2}
FINANCIAL_MAP = {"None":0,"Low":1,"Moderate":2,"High":3}
TRAUMA_MAP = {"None":0,"One":1,"Two":2,"Three or more":3}
SUB_MAP = {"None":0,"Occasional":1,"Heavy":2}
CHRONIC_MAP = {"None":0,"Mild":1,"Moderate":2,"Severe":3}
GENDER_MAP = {"Female":0,"Male":1}


# ========================================================= #
# 1. ENCODING CATEGORICAL TEXT → NUMERIC
# ========================================================= #

def encode_categorical(df):
    print("\n[1] Encoding categorical columns...")

    # PHQ 1–9 -> numerical 0–3
    phq_cols = [f"{i}. " for i in range(1,10)]
    real_phq = [col for col in df.columns if col.startswith(tuple("123456789."))]
    for col in real_phq:
        df[col] = df[col].replace(PHQ_MAP)

    # Impact Q10–Q15
    impact_cols = list(IMPACT_MAP.keys())
    imp_range = [
        "10. Hormonal changes affecting mood (menstrual, pregnancy, menopause)?",
        "11. Postpartum mood changes (if applicable)?",
        "12. Body image concerns or dissatisfaction?",
        "13. Relationship stress or domestic issues?",
        "14. Work-life balance difficulties?",
        "15. Caregiving burden (children, elderly parents)?",
    ]
    for c in imp_range: df[c] = df[c].replace(IMPACT_MAP)

    df["25. Vitamin D level"] = df["25. Vitamin D level"].replace(VITD_MAP)
    df["26. Social support"] = df["26. Social support"].replace(SOCIAL_MAP)
    df["27. Financial stress level"] = df["27. Financial stress level"].replace(FINANCIAL_MAP)
    df["28. Traumatic events (past year)"] = df["28. Traumatic events (past year)"].replace(TRAUMA_MAP)
    df["29. Substance use"] = df["29. Substance use"].replace(SUB_MAP)
    df["30. Chronic illness or pain"] = df["30. Chronic illness or pain"].replace(CHRONIC_MAP)
    df["31. Family history of depression"] = df["31. Family history of depression"].replace({"Yes":1,"No":0})
    df["Gender (optional)"] = df["Gender (optional)"].replace(GENDER_MAP)

    print("✓ Encoding completed.")
    return df


# ========================================================= #
# 2. IMPUTE MISSING VALUES + OUTLIER TREATMENT
# ========================================================= #

def clean_data(df):
    print("\n[2] Handling Missing Values & Outliers...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(df[numeric_cols])

    # IQR Outlier Capping
    for col in numeric_cols:
        q1,q3 = df[col].quantile(.25), df[col].quantile(.75)
        iqr = q3-q1
        df[col]=np.clip(df[col],q1-1.5*iqr, q3+1.5*iqr)

    print("✓ Missing & Outlier Fix Complete.")
    return df


# ========================================================= #
# 3. SPLIT + SCALE + SAVE DATA
# ========================================================= #

def split_scale(df):

    # remove non-features
    df.drop(columns=["participant_id","Email Address"], inplace=True)

    print("\n[3] Converting depression risk → binary target")
    df["depression_label"] = df["depression_risk"].map({"Low":0,"Moderate":1,"High":1})

    # remove clinical leakage (INCLUDING risk_score_continuous!)
    remove_cols = [
        "depression_risk","phq9_total","risk_score_continuous",  # ← ADDED THIS!
        "1. Little interest or pleasure in doing things?",
        "2. Feeling down, depressed, or hopeless?",
        "3. Trouble falling/staying asleep, or sleeping too much?",
        "4. Feeling tired or having little energy?",
        "5. Poor appetite or overeating?",
        "6. Feeling bad about yourself or that you are a failure?",
        "7. Trouble concentrating on things?",
        "8. Moving/speaking slowly or being fidgety/restless?",
        "9. Thoughts of being better off dead or hurting yourself?"
    ]

    X=df.drop(columns=[c for c in remove_cols if c in df.columns]+["depression_label"])
    y=df["depression_label"].values
    features=list(X.columns)

    X_temp,X_test,y_temp,y_test=train_test_split(X,y,test_size=TEST_SIZE,random_state=RAND_SEED,stratify=y)
    X_train,X_val,y_train,y_val=train_test_split(X_temp,y_temp,test_size=VAL_SIZE/(1-TEST_SIZE),random_state=RAND_SEED,stratify=y_temp)

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_val=scaler.transform(X_val)
    X_test=scaler.transform(X_test)

    print(f"✓ Shapes → Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}")
    return X_train,X_val,X_test,y_train,y_val,y_test,features,scaler


# ========================================================= #

def main():
    print("\n=========================================================")
    print("⚡ RUNNING PREPROCESSING")
    print("=========================================================")

    df=pd.read_csv(RAW_CSV_PATH)
    df=encode_categorical(df)
    df=clean_data(df)

    X_train,X_val,X_test,y_train,y_val,y_test,features,scaler = split_scale(df)

    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    np.save(PROCESSED_DIR/"X_train.npy",X_train)
    np.save(PROCESSED_DIR/"X_val.npy",X_val)
    np.save(PROCESSED_DIR/"X_test.npy",X_test)
    np.save(PROCESSED_DIR/"y_train.npy",y_train)
    np.save(PROCESSED_DIR/"y_val.npy",y_val)
    np.save(PROCESSED_DIR/"y_test.npy",y_test)
    joblib.dump(scaler,PROCESSED_DIR/"scaler.pkl")
    joblib.dump(features,PROCESSED_DIR/"feature_names.pkl")

    print("\n🔥 Preprocessing Complete — Files Saved to /data/processed/")
    print("=========================================================\n")


if __name__=="__main__":
    main()
