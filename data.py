"""
Synthetic dataset generator for depression questionnaire (Indian women)

⚠ IMPORTANT (for ethics):
- This data is COMPLETELY SYNTHETIC.
- Use ONLY for model development / code testing.
- Do NOT present it as real survey data in thesis/paper.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ========= CONFIG =========
N = 1500  # kitni rows chahiye (change as needed)

np.random.seed(42)
random.seed(42)

# Date range: Sep 1, 2025 to Nov 30, 2025
start_date = datetime(2025, 9, 1)
end_date = datetime(2025, 11, 30)
date_range_sec = int((end_date - start_date).total_seconds())

# --------- Helper lists ---------
likert_4 = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day",
]

impact_levels = ["No impact", "Mild", "Moderate", "Severe"]
vitamin_status = ["Deficient", "Insufficient", "Sufficient", "Optimal"]
support_levels = ["Weak", "Moderate", "Strong"]
financial_stress = ["None", "Low", "Moderate", "High"]
trauma_levels = ["None", "One", "Two", "Three or more"]
substance_use = ["None", "Occasional", "Moderate", "Heavy"]
chronic_illness = ["None", "Mild", "Moderate", "Severe"]
yes_no = ["Yes", "No"]

# Typical Indian female names/surnames (synthetic only)
first_names = [
    "Anjali", "Priya", "Neha", "Pooja", "Kavita", "Yogita", "Komal",
    "Ishita", "Sneha", "Nandini", "Shruti", "Ritika", "Riya", "Simran",
    "Shreya", "Divya", "Meena", "Kiran", "Aarti", "Soni"
]

last_names = [
    "Sharma", "Verma", "Gupta", "Yadav", "Chauhan", "Tyagi", "Meena",
    "Prajapati", "Kaushik", "Rajoriya", "Singh", "Tripathi", "Sinha",
    "Rana", "Jain", "Patel"
]


def random_timestamp():
    """Random timestamp between Sep–Nov 2025 (same format as sample)."""
    offset = random.randint(0, date_range_sec)
    ts = start_date + timedelta(seconds=offset)
    # Format: 11/26/2025 18:28:09  (MM/DD/YYYY HH:MM:SS 24h)
    return ts.strftime("%m/%d/%Y %H:%M:%S")


def random_email(i):
    """
    Indian-style Gmail ID, synthetic.
    Example: anjali.sharma23@gmail.com or priyaverma1999@gmail.com
    """
    fn = random.choice(first_names).lower()
    ln = random.choice(last_names).lower()

    # kabhi kabhi dot, kabhi direct
    if random.random() < 0.5:
        base = f"{fn}.{ln}"
    else:
        base = f"{fn}{ln}"

    # numbers optional
    if random.random() < 0.7:
        year = random.randint(1985, 2010)
        email = f"{base}{year}@gmail.com"
    else:
        rand_num = random.randint(1, 9999)
        email = f"{base}{rand_num}@gmail.com"

    return email


rows = []

for i in range(1, N + 1):
    row = {}

    # ---------- Meta ----------
    row["Timestamp"] = random_timestamp()
    row["Email Address"] = random_email(i)

    # ---------- Q1–Q9 (PHQ-like) ----------
    row["1. Little interest or pleasure in doing things?"] = np.random.choice(likert_4)
    row["2. Feeling down, depressed, or hopeless?"] = np.random.choice(likert_4)
    row["3. Trouble falling/staying asleep, or sleeping too much?"] = np.random.choice(likert_4)
    row["4. Feeling tired or having little energy?"] = np.random.choice(likert_4)
    row["5. Poor appetite or overeating?"] = np.random.choice(likert_4)
    row["6. Feeling bad about yourself or that you are a failure?"] = np.random.choice(likert_4)
    row["7. Trouble concentrating on things?"] = np.random.choice(likert_4)
    row["8. Moving/speaking slowly or being fidgety/restless?"] = np.random.choice(likert_4)
    row["9. Thoughts of being better off dead or hurting yourself?"] = np.random.choice(likert_4)

    # ---------- Q10–Q15 (psychosocial / women-specific) ----------
    row["10. Hormonal changes affecting mood (menstrual, pregnancy, menopause)?"] = np.random.choice(impact_levels)
    row["11. Postpartum mood changes (if applicable)?"] = np.random.choice(impact_levels)
    row["12. Body image concerns or dissatisfaction?"] = np.random.choice(impact_levels)
    row["13. Relationship stress or domestic issues?"] = np.random.choice(impact_levels)
    row["14. Work-life balance difficulties?"] = np.random.choice(impact_levels)
    row["15. Caregiving burden (children, elderly parents)?"] = np.random.choice(impact_levels)

    # ---------- Physiological (Q16–Q21) ----------
    # Indian adult females: approximate ranges
    row["16. Resting heart rate (bpm)"] = int(np.clip(np.random.normal(80, 10), 55, 120))
    row["17. Heart rate variability (ms)"] = int(np.clip(np.random.normal(55, 15), 15, 120))
    row["18. Sleep duration (hours per night)"] = round(np.clip(np.random.normal(6.5, 1.5), 3, 10), 1)
    row["19. Sleep quality"] = random.randint(0, 10)   # 0–10 scale jaisa tumne use kiya
    row["20. Physical activity (minutes per week)"] = int(np.clip(np.random.normal(180, 100), 0, 1000))
    row["21. Stress level"] = random.randint(1, 10)    # 1–10

    # ---------- Clinical (Q22–Q25) ----------
    row["22. Blood Pressure - Systolic (mmHg)"] = int(np.clip(np.random.normal(115, 15), 85, 180))
    row["23. Blood Pressure - Diastolic (mmHg)"] = int(np.clip(np.random.normal(75, 10), 50, 120))
    row["24. BMI (Body Mass Index) - kg/m²"] = round(np.clip(np.random.normal(24, 4), 15, 45), 1)
    row["25. Vitamin D level"] = np.random.choice(vitamin_status)

    # ---------- Psychosocial (Q26–Q31) ----------
    row["26. Social support"] = np.random.choice(support_levels)
    row["27. Financial stress level"] = np.random.choice(financial_stress)
    row["28. Traumatic events (past year)"] = np.random.choice(trauma_levels)
    row["29. Substance use"] = np.random.choice(substance_use)
    row["30. Chronic illness or pain"] = np.random.choice(chronic_illness)
    row["31. Family history of depression"] = np.random.choice(yes_no)

    # ---------- Demographics ----------
    # Indian women, say 18–60 (you can tighten range if needed)
    row["Age (optional)"] = random.randint(18, 60)
    row["Gender (optional)"] = "Female"

    # Mark synthetic explicitly
    row["is_synthetic"] = True

    rows.append(row)

# ---------- DataFrame & Save ----------
df = pd.DataFrame(rows)

# Columns ko order me rakhne ke liye (same as tumhara header)
ordered_cols = [
    "Timestamp",
    "Email Address",
    "1. Little interest or pleasure in doing things?",
    "2. Feeling down, depressed, or hopeless?",
    "3. Trouble falling/staying asleep, or sleeping too much?",
    "4. Feeling tired or having little energy?",
    "5. Poor appetite or overeating?",
    "6. Feeling bad about yourself or that you are a failure?",
    "7. Trouble concentrating on things?",
    "8. Moving/speaking slowly or being fidgety/restless?",
    "9. Thoughts of being better off dead or hurting yourself?",
    "10. Hormonal changes affecting mood (menstrual, pregnancy, menopause)?",
    "11. Postpartum mood changes (if applicable)?",
    "12. Body image concerns or dissatisfaction?",
    "13. Relationship stress or domestic issues?",
    "14. Work-life balance difficulties?",
    "15. Caregiving burden (children, elderly parents)?",
    "16. Resting heart rate (bpm)",
    "17. Heart rate variability (ms)",
    "18. Sleep duration (hours per night)",
    "19. Sleep quality",
    "20. Physical activity (minutes per week)",
    "21. Stress level",
    "22. Blood Pressure - Systolic (mmHg)",
    "23. Blood Pressure - Diastolic (mmHg)",
    "24. BMI (Body Mass Index) - kg/m²",
    "25. Vitamin D level",
    "26. Social support",
    "27. Financial stress level",
    "28. Traumatic events (past year)",
    "29. Substance use",
    "30. Chronic illness or pain",
    "31. Family history of depression",
    "Age (optional)",
    "Gender (optional)",
    "is_synthetic"
]

df = df[ordered_cols]

# CSV file generate karo
df.to_csv("synthetic_indian_women_depression_data_sep_nov_2025.csv", index=False)

print("Synthetic dataset shape:", df.shape)
print(df.head())
