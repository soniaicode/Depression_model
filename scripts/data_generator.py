"""
Data Generation Module
Generate synthetic questionnaire-style dataset matching PhD research specifications
- 31 items + Age + Gender + Email
- Only female participants
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


class DepressionDataGenerator:
    """
    Generate synthetic dataset for depression prediction research
    Based on women's depression questionnaire (31 items)
    """

    def __init__(self, n_samples=1500, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)

        # Common categorical options
        self.likert_4 = [
            "Not at all",
            "Several days",
            "More than half the days",
            "Nearly every day",
        ]
        self.impact_4 = ["No impact", "Mild", "Moderate", "Severe"]

    # -----------------------
    # Helper methods
    # -----------------------
    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def _choice(self, options, probs=None, size=1):
        return np.random.choice(options, size=size, p=probs)

    def _generate_indian_emails(self):
        """Generate realistic Indian email addresses"""
        # Common Indian first names (female)
        first_names = [
            "Priya", "Anita", "Sunita", "Kavita", "Meera", "Sita", "Geeta", "Neeta",
            "Asha", "Usha", "Radha", "Shanti", "Lakshmi", "Saraswati", "Parvati",
            "Anjali", "Pooja", "Ritu", "Nisha", "Deepika", "Rekha", "Sushma",
            "Vandana", "Kiran", "Jyoti", "Mamta", "Seema", "Reema", "Veena",
            "Shilpa", "Preeti", "Swati", "Bharti", "Shweta", "Nikita", "Ruchi",
            "Archana", "Kalpana", "Sangita", "Rashmi", "Smita", "Nidhi", "Pallavi",
            "Manisha", "Sonia", "Tina", "Riya", "Shreya", "Divya", "Kavya"
        ]

        # Common Indian surnames
        surnames = [
            "Sharma", "Gupta", "Singh", "Kumar", "Agarwal", "Jain", "Bansal", "Mittal",
            "Chopra", "Malhotra", "Kapoor", "Arora", "Bhatia", "Sethi", "Khanna",
            "Verma", "Yadav", "Pandey", "Mishra", "Tiwari", "Dubey", "Shukla",
            "Srivastava", "Tripathi", "Chaturvedi", "Saxena", "Agrawal", "Goyal",
            "Joshi", "Nair", "Menon", "Pillai", "Reddy", "Rao", "Patel", "Shah",
            "Desai", "Mehta", "Modi", "Thakkar", "Parikh", "Trivedi", "Bhatt",
            "Iyer", "Krishnan", "Subramanian", "Raman", "Bose", "Ghosh", "Sen"
        ]

        # Popular email domains in India
        domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "rediffmail.com",
            "yahoo.co.in", "gmail.com", "gmail.com", "gmail.com"  # Gmail is most popular
        ]

        emails = []
        used_emails = set()

        for _ in range(self.n_samples):
            while True:
                first_name = np.random.choice(first_names)
                surname = np.random.choice(surnames)
                domain = np.random.choice(domains)

                # Different email format patterns
                format_type = np.random.choice([1, 2, 3, 4, 5])

                if format_type == 1:
                    # firstname.surname@domain
                    email = f"{first_name.lower()}.{surname.lower()}@{domain}"
                elif format_type == 2:
                    # firstnamesurname@domain
                    email = f"{first_name.lower()}{surname.lower()}@{domain}"
                elif format_type == 3:
                    # firstname.surname.number@domain
                    number = np.random.randint(1, 999)
                    email = f"{first_name.lower()}.{surname.lower()}.{number}@{domain}"
                elif format_type == 4:
                    # firstname_surname@domain
                    email = f"{first_name.lower()}_{surname.lower()}@{domain}"
                else:
                    # firstnamesurnamenumber@domain
                    number = np.random.randint(1, 99)
                    email = f"{first_name.lower()}{surname.lower()}{number}@{domain}"

                # Ensure uniqueness
                if email not in used_emails:
                    used_emails.add(email)
                    emails.append(email)
                    break

        return emails

    # -----------------------
    # Main generation methods
    # -----------------------
    def generate_demographics(self):
        """
        Generate demographic variables:
        - Age (18–55)
        - Gender (Female only)
        """
        ages = np.random.randint(18, 56, size=self.n_samples)
        genders = ["Female"] * self.n_samples

        return {
            "Age (optional)": ages,
            "Gender (optional)": genders,
        }

    def generate_questionnaire_items(self):
        """
        Generate all questionnaire items Q1–Q31 with realistic distributions.
        """

        data = {}

        # Q1–Q9: PHQ-type items (4-point Likert)
        # Slightly higher probability for mild–moderate symptoms
        symptom_probs = [0.22, 0.40, 0.23, 0.15]  # Not at all, Several, Half, Nearly

        q_map = {
            1: "1. Little interest or pleasure in doing things?",
            2: "2. Feeling down, depressed, or hopeless?",
            3: "3. Trouble falling/staying asleep, or sleeping too much?",
            4: "4. Feeling tired or having little energy?",
            5: "5. Poor appetite or overeating?",
            6: "6. Feeling bad about yourself or that you are a failure?",
            7: "7. Trouble concentrating on things?",
            8: "8. Moving/speaking slowly or being fidgety/restless?",
            9: "9. Thoughts of being better off dead or hurting yourself?",
        }

        for q in range(1, 10):
            data[q_map[q]] = self._choice(
                self.likert_4, probs=symptom_probs, size=self.n_samples
            )

        # Q10–Q15: Hormonal / psychosocial impact 4-point
        data["10. Hormonal changes affecting mood (menstrual, pregnancy, menopause)?"] = (
            self._choice(self.impact_4, probs=[0.35, 0.35, 0.20, 0.10], size=self.n_samples)
        )
        data["11. Postpartum mood changes (if applicable)?"] = self._choice(
            self.impact_4, probs=[0.55, 0.25, 0.15, 0.05], size=self.n_samples
        )
        data["12. Body image concerns or dissatisfaction?"] = self._choice(
            self.impact_4, probs=[0.25, 0.35, 0.25, 0.15], size=self.n_samples
        )
        data["13. Relationship stress or domestic issues?"] = self._choice(
            self.impact_4, probs=[0.25, 0.30, 0.25, 0.20], size=self.n_samples
        )
        data["14. Work-life balance difficulties?"] = self._choice(
            self.impact_4, probs=[0.20, 0.30, 0.30, 0.20], size=self.n_samples
        )
        data["15. Caregiving burden (children, elderly parents)?"] = self._choice(
            self.impact_4, probs=[0.35, 0.30, 0.20, 0.15], size=self.n_samples
        )

        # Q16–23: Physiological / lifestyle numeric measures
        # 16. Resting heart rate (bpm)
        resting_hr = np.random.normal(80, 15, self.n_samples)
        data["16. Resting heart rate (bpm)"] = [
            int(self._clamp(round(x), 50, 120)) for x in resting_hr
        ]

        # 17. Heart rate variability (ms)
        hrv = np.random.normal(60, 20, self.n_samples)
        data["17. Heart rate variability (ms)"] = [
            int(self._clamp(round(x), 20, 120)) for x in hrv
        ]

        # 18. Sleep duration (hours per night)
        sleep_hours = np.random.normal(6.5, 1.5, self.n_samples)
        data["18. Sleep duration (hours per night)"] = [
            round(self._clamp(x, 3, 10), 1) for x in sleep_hours
        ]

        # 19. Sleep quality (0–10)
        sleep_quality = np.random.normal(6, 2, self.n_samples)
        data["19. Sleep quality"] = [
            int(self._clamp(round(x), 0, 10)) for x in sleep_quality
        ]

        # 20. Physical activity (minutes per week)
        activity_levels = [0, 30, 60, 90, 120, 150, 200, 300, 400, 500]
        activity_probs = [0.05, 0.08, 0.12, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]
        data["20. Physical activity (minutes per week)"] = self._choice(
            activity_levels, probs=activity_probs, size=self.n_samples
        )

        # 21. Stress level (0–10)
        stress = np.random.normal(6.5, 2, self.n_samples)
        data["21. Stress level"] = [
            int(self._clamp(round(x), 0, 10)) for x in stress
        ]

        # 22–23. Blood pressure
        sbp = np.random.normal(115, 12, self.n_samples)  # systolic
        dbp = np.random.normal(78, 8, self.n_samples)    # diastolic
        data["22. Blood Pressure - Systolic (mmHg)"] = [
            int(self._clamp(round(x), 90, 160)) for x in sbp
        ]
        data["23. Blood Pressure - Diastolic (mmHg)"] = [
            int(self._clamp(round(x), 60, 110)) for x in dbp
        ]

        # 24. BMI
        bmi = np.random.normal(24, 4.5, self.n_samples)
        data["24. BMI (Body Mass Index) - kg/m²"] = [
            round(self._clamp(x, 16, 40), 1) for x in bmi
        ]

        # 25. Vitamin D level
        vitd_options = ["Insufficient", "Sufficient", "Optimal"]
        vitd_probs = [0.35, 0.45, 0.20]
        data["25. Vitamin D level"] = self._choice(
            vitd_options, probs=vitd_probs, size=self.n_samples
        )

        # 26. Social support
        social_opts = ["Weak", "Moderate", "Strong"]
        social_probs = [0.25, 0.45, 0.30]
        data["26. Social support"] = self._choice(
            social_opts, probs=social_probs, size=self.n_samples
        )

        # 27. Financial stress level
        fin_opts = ["None", "Low", "Moderate", "High"]
        fin_probs = [0.10, 0.25, 0.45, 0.20]
        data["27. Financial stress level"] = self._choice(
            fin_opts, probs=fin_probs, size=self.n_samples
        )

        # 28. Traumatic events (past year)
        trauma_opts = ["None", "One", "Two", "Three or more"]
        trauma_probs = [0.55, 0.25, 0.12, 0.08]
        data["28. Traumatic events (past year)"] = self._choice(
            trauma_opts, probs=trauma_probs, size=self.n_samples
        )

        # 29. Substance use
        substance_opts = ["None", "Occasional", "Heavy"]
        substance_probs = [0.70, 0.25, 0.05]
        data["29. Substance use"] = self._choice(
            substance_opts, probs=substance_probs, size=self.n_samples
        )

        # 30. Chronic illness or pain
        chronic_opts = ["None", "Mild", "Moderate", "Severe"]
        chronic_probs = [0.45, 0.30, 0.18, 0.07]
        data["30. Chronic illness or pain"] = self._choice(
            chronic_opts, probs=chronic_probs, size=self.n_samples
        )

        # 31. Family history of depression
        fam_opts = ["Yes", "No"]
        fam_probs = [0.40, 0.60]
        data["31. Family history of depression"] = self._choice(
            fam_opts, probs=fam_probs, size=self.n_samples
        )

        return data

    def _map_likert_to_score(self, value: str) -> int:
        """
        Map 4-point Likert to numeric score 0–3 for PHQ-like total.
        """
        mapping = {
            "Not at all": 0,
            "Several days": 1,
            "More than half the days": 2,
            "Nearly every day": 3,
        }
        return mapping.get(value, 0)

    def add_depression_risk_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create depression risk label using:
        - PHQ-9 (Q1–Q9)
        - Stress level (Q21)
        - Social support (Q26)
        - Financial stress level (Q27)
        Then compute a continuous risk score and split into
        Low / Moderate / High using percentiles.
        """

        # ---- 1) PHQ-9 total (0–27) ----
        phq_cols = [
            "1. Little interest or pleasure in doing things?",
            "2. Feeling down, depressed, or hopeless?",
            "3. Trouble falling/staying asleep, or sleeping too much?",
            "4. Feeling tired or having little energy?",
            "5. Poor appetite or overeating?",
            "6. Feeling bad about yourself or that you are a failure?",
            "7. Trouble concentrating on things?",
            "8. Moving/speaking slowly or being fidgety/restless?",
            "9. Thoughts of being better off dead or hurting yourself?",
        ]

        phq_numeric = []
        for col in phq_cols:
            phq_numeric.append(df[col].apply(self._map_likert_to_score))

        phq_total = sum(phq_numeric)
        df["phq9_total"] = phq_total

        # Normalize to 0–1
        phq_norm = df["phq9_total"] / 27.0

        # ---- 2) Stress level (0–10) → 0–1 ----
        stress = df["21. Stress level"].astype(float)
        stress_norm = stress / 10.0

        # ---- 3) Social support (Weak/Moderate/Strong) → risk score ----
        # Higher = more risk (Weak > Moderate > Strong)
        social_map = {"Weak": 2, "Moderate": 1, "Strong": 0}
        social_raw = df["26. Social support"].map(social_map).fillna(1.0)
        social_norm = social_raw / 2.0  # 0–1

        # ---- 4) Financial stress (None/Low/Moderate/High) → 0–3 → 0–1 ----
        fin_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
        fin_raw = df["27. Financial stress level"].map(fin_map).fillna(1.0)
        fin_norm = fin_raw / 3.0

        # ---- 5) Continuous risk score (0–1-ish) ----
        # Weights: PHQ most important, then stress, then financial, then social
        base_risk = (
            0.45 * phq_norm +
            0.20 * stress_norm +
            0.20 * fin_norm +
            0.15 * social_norm
        )

        # Add small noise to avoid too-perfect separation
        noise = np.random.normal(0, 0.07, size=len(df))
        risk_score = base_risk + noise

        # For reference, store continuous score (optional)
        df["risk_score_continuous"] = risk_score

        # ---- 6) Use percentiles to define Low / Moderate / High ----
        # ~40% low, 30% moderate, 30% high (approx)
        p_low = np.percentile(risk_score, 40)
        p_high = np.percentile(risk_score, 70)

        def risk_from_score(s):
            if s < p_low:
                return "Low"
            elif s < p_high:
                return "Moderate"
            else:
                return "High"

        df["depression_risk"] = risk_score.map(risk_from_score)

        # ---- 7) Binary label for ML (Low=0, Moderate/High=1) ----
        df["depression_label"] = df["depression_risk"].map(
            {"Low": 0, "Moderate": 1, "High": 1}
        )

        print("\n[Label Distribution]")
        print(df["depression_risk"].value_counts())

        return df

    def generate_complete_dataset(self):
        """Generate complete dataset combining all questionnaire items + demographics"""

        print("Generating women's depression questionnaire dataset...")
        print(f"Total samples: {self.n_samples}")

        # Questionnaire & demographics
        questionnaire = self.generate_questionnaire_items()
        demographics = self.generate_demographics()

        # Combine into DataFrame
        data = {**questionnaire, **demographics}
        df = pd.DataFrame(data)

        # Email Address (realistic Indian emails)
        indian_emails = self._generate_indian_emails()
        df.insert(0, "Email Address", indian_emails)

        # Participant ID
        df.insert(0, "participant_id", [f"P{i:04d}" for i in range(1, self.n_samples + 1)])

        # Add depression_risk & depression_label
        df = self.add_depression_risk_label(df)

        print("✓ Dataset generated successfully!")
        print(f"  - Low risk: {(df['depression_risk'] == 'Low').sum()}")
        print(f"  - Moderate risk: {(df['depression_risk'] == 'Moderate').sum()}")
        print(f"  - High risk: {(df['depression_risk'] == 'High').sum()}")
        print(f"  - Total features (including labels & IDs): {len(df.columns)}")

        return df

    def save_dataset(self, df, output_dir=None):
        """Save dataset to CSV, Excel, and metadata JSON.

        Output path is fixed to <project_root>/data/raw/
        """

        if output_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            output_path = project_root / "data" / "raw"
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        csv_path = output_path / "depression_questionnaire_data.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✓ Saved: {csv_path}")

        # Save as Excel (optional - requires openpyxl)
        try:
            excel_path = output_path / "depression_questionnaire_data.xlsx"
            df.to_excel(excel_path, index=False)
            print(f"✓ Saved: {excel_path}")
        except ImportError:
            print("⚠ Excel export skipped (openpyxl not installed)")
        except Exception as e:
            print(f"⚠ Excel export failed: {e}")

        # Metadata
        risk_counts = df["depression_risk"].value_counts().to_dict()

        metadata = {
            "n_samples": int(len(df)),
            "n_features": int(len(df.columns) - 4),  # exclude ID, Email, risk, label helper
            "risk_distribution": risk_counts,
            "features": list(df.columns),
            "generation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"✓ Saved: {metadata_path}")

        return csv_path


if __name__ == "__main__":
    # Generate dataset
    generator = DepressionDataGenerator(n_samples=1500, random_state=42)
    df = generator.generate_complete_dataset()

    # Save dataset under <project_root>/data/raw/
    generator.save_dataset(df)

    # Display sample
    print("\nDataset Preview:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
