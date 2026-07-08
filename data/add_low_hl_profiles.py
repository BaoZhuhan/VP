"""Add 3 low-HL profiles to balance the dataset to 8/8/8."""

import json, os, sys

profiles_path = os.path.join(os.path.dirname(__file__), "patient_profiles.json")

with open(profiles_path) as f:
    data = json.load(f)

new_profiles = [
    {
        "patient_id": "P025",
        "demographics": {
            "age": 52,
            "gender": "male",
            "occupation": "warehouse_worker"
        },
        "clinical_attributes": {
            "primary_symptom": "headache",
            "secondary_symptoms": ["dizziness", "nausea"],
            "symptom_duration": "1-2_weeks",
            "disease_severity": 3,
            "temperature_c": 36.8,
            "chronic_conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "allergies": ["none"],
            "smoking_status": "current",
            "alcohol_use": "occasional"
        },
        "affective_attributes": {
            "anxiety_level": 3,
            "pain_level": 3,
            "mood": "irritable",
            "cooperation_willingness": 3,
            "health_literacy": "low",
            "communication_style": "terse"
        },
        "scenario": {
            "category": "neurological",
            "urgency": "routine",
            "setting": "primary_care",
            "chief_complaint": "I keep getting these bad headaches that won't go away, and sometimes I feel dizzy."
        }
    },
    {
        "patient_id": "P026",
        "demographics": {
            "age": 39,
            "gender": "female",
            "occupation": "cashier"
        },
        "clinical_attributes": {
            "primary_symptom": "abdominal_pain",
            "secondary_symptoms": ["nausea", "loss_of_appetite"],
            "symptom_duration": "2-4_weeks",
            "disease_severity": 2,
            "temperature_c": 37.0,
            "chronic_conditions": ["none"],
            "medications": ["none"],
            "allergies": ["none"],
            "smoking_status": "never",
            "alcohol_use": "none"
        },
        "affective_attributes": {
            "anxiety_level": 2,
            "pain_level": 3,
            "mood": "resigned",
            "cooperation_willingness": 4,
            "health_literacy": "low",
            "communication_style": "terse"
        },
        "scenario": {
            "category": "gastrointestinal",
            "urgency": "routine",
            "setting": "primary_care",
            "chief_complaint": "My stomach's been hurting off and on for a few weeks. I don't feel like eating much."
        }
    },
    {
        "patient_id": "P027",
        "demographics": {
            "age": 61,
            "gender": "female",
            "occupation": "hotel_housekeeper"
        },
        "clinical_attributes": {
            "primary_symptom": "fatigue",
            "secondary_symptoms": ["weight_loss", "muscle_ache"],
            "symptom_duration": "3-6_months",
            "disease_severity": 3,
            "temperature_c": 36.6,
            "chronic_conditions": ["diabetes_type2"],
            "medications": ["metformin"],
            "allergies": ["none"],
            "smoking_status": "never",
            "alcohol_use": "none"
        },
        "affective_attributes": {
            "anxiety_level": 4,
            "pain_level": 2,
            "mood": "anxious",
            "cooperation_willingness": 4,
            "health_literacy": "low",
            "communication_style": "stoic"
        },
        "scenario": {
            "category": "endocrine",
            "urgency": "routine",
            "setting": "primary_care",
            "chief_complaint": "I'm tired all the time no matter how much I sleep, and I've been losing weight without trying."
        }
    }
]

# Verify no duplicate IDs
existing_ids = {p["patient_id"] for p in data["profiles"]}
for np in new_profiles:
    assert np["patient_id"] not in existing_ids, f"Duplicate ID: {np['patient_id']}"
    existing_ids.add(np["patient_id"])

data["profiles"].extend(new_profiles)

# Verify final distribution
hl_dist = {"low": 0, "medium": 0, "high": 0}
for p in data["profiles"]:
    hl_dist[p["affective_attributes"]["health_literacy"]] += 1

print("Adding 3 low-HL profiles...")
for np in new_profiles:
    print(f"  {np['patient_id']}: {np['demographics']['age']}yo {np['demographics']['gender']} "
          f"{np['scenario']['category']} — {np['affective_attributes']['communication_style']}")
print(f"\nNew distribution: {hl_dist}")
print(f"Total profiles: {len(data['profiles'])}")

with open(profiles_path, "w") as f:
    json.dump(data, f, indent=2)

print("Saved to patient_profiles.json")
