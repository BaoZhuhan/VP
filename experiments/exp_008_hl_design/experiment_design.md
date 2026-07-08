# Experiment Design: Health Literacy Consistency in LLM-based Virtual Patient Simulation

## Overview

This document describes the detailed experimental design for evaluating health literacy consistency in LLM-based virtual patient simulation. It implements the framework outlined in `VP.md` and should be read alongside that document.

## Conditions

### Condition Matrix

We compare across five distinct generation conditions:

| # | Condition | Family | N | Status | Source |
|---|-----------|--------|---|--------|--------|
| C1 | Zero-shot Standard | Simple prompting | 24 | ✅ Existing | Our dialogue_generator.py |
| C2 | Zero-shot Consistency | Simple prompting | 24 | ✅ Existing | Our dialogue_generator.py |
| C3 | Zero-shot Structured State | Simple prompting | 24 | ✅ Existing | Our dialogue_generator.py |
| C4 | PatientSim (native language proficiency) | Deployed system | 24 | 🔴 Generate | github.com/dek924/PatientSim |
| C5 | PatientSim + HL injection | Deployed system | 24 | 🔴 Generate | github.com/dek924/PatientSim |
| C6 | AgentClinic (default patient profile) | Deployed system | 24 | 🔴 Generate | github.com/samuelschmidgall/agentclinic |
| C7 | AgentClinic + HL injection | Deployed system | 24 | 🔴 Generate | github.com/samuelschmidgall/agentclinic |
| C8 | Literacy-Anchored (proposed) | Behavioral specification | 24 | 🔴 Generate | Our dialogue_generator.py |

**Total existing**: 72 dialogues (C1–C3)
**Total to generate**: 120 dialogues (C4–C8)
**Grand total**: 192 dialogues

### Existing Open-Source Frameworks for Deployment

Below are the deployable virtual patient frameworks we consider for conditions C4–C7. Ranked by deployability and relevance to health literacy.

| Framework | Year | Venue | Persona Customization | HL-Related Axis | Deployment |
|-----------|------|-------|----------------------|-----------------|------------|
| **PatientSim** | 2025 | NeurIPS Spotlight | 4 axes (personality, language proficiency, recall, confusion) | Language proficiency (CEFR A/B/C) | `pip install patientsim` |
| **AgentClinic** | 2026 | npj Digital Medicine | Patient agent via scenario config | General patient profile (none specific to HL) | `git clone https://github.com/samuelschmidgall/agentclinic` |
| **Roleplay-doh** | 2024 | EMNLP | Domain-expert-defined behavioral principles | None (principle-based) | [roleplay-doh.github.io](https://roleplay-doh.github.io/) |
| **EvoPatient** | 2025 | ACL | Patient co-evolution via multi-turn diagnosis | None (evolves from dialogue) | [github.com/ZJUMAI/EvoPatient](https://github.com/ZJUMAI/EvoPatient) |
| **Agent Hospital** | 2024 | arXiv | Multi-agent with patient profile | None (general profile) | [github.com/wisdom-pan/Agent_Hospital](https://github.com/wisdom-pan/Agent_Hospital) |
| **MSPRP** | 2026 | arXiv | 5D persona vector (personality, emotion, recall, comprehension, fluency) | Comprehension, Fluency | [github.com/SerajJon/MSPRP](https://github.com/SerajJon/MSPRP) |

#### Selection Criteria

- **PatientSim** is selected as the primary comparison target because it: (a) has an installable Python package, (b) already includes language proficiency as a persona axis (the closest existing analogue to health literacy), (c) supports custom attribute injection via `additional_patient_conditions`, and (d) was validated by clinicians and published at a top venue.
- **AgentClinic** is selected as a secondary target because it: (a) represents a clinically grounded, peer-reviewed benchmark (npj Digital Medicine), (b) supports multi-backend LLMs, and (c) covers nine medical specialties providing good domain coverage.

Other frameworks are noted in the table but not deployed in this initial study. We select two to keep the experiment manageable while providing sufficient coverage across different system architectures.

### PatientSim Deployment Protocol

For each of the 24 profiles:

**C4 (native):**
```python
from patientsim import PatientAgent

# Map our HL level to PatientSim's language proficiency axis
proficiency_map = {"low": "A", "medium": "B", "high": "C"}

agent = PatientAgent(
    model="deepseek-v4-flash",
    visit_type="outpatient",
    language_proficiency_level=proficiency_map[profile.hl_level],
    personality="plain",
    recall_level="low" if profile.hl_level == "low" else "medium",
    confusion_level="moderate" if profile.hl_level == "low" else "normal",
)
```

**C5 (HL injection):**
```python
agent = PatientAgent(
    model="deepseek-v4-flash",
    visit_type="outpatient",
    language_proficiency_level=proficiency_map[profile.hl_level],
    personality="plain",
    recall_level="low" if profile.hl_level == "low" else "medium",
    confusion_level="moderate" if profile.hl_level == "low" else "normal",
    additional_patient_conditions={
        "health_literacy": profile.hl_level,
        "communication_style": profile.communication_style,
    },
)
```

### AgentClinic Deployment Protocol

AgentClinic requires configuring the patient agent through a scenario JSON. We will create 24 scenario files mapping our profiles to AgentClinic's patient configuration format, with the HL attribute added to the patient description for C7.

## Prompt Templates (For Our Generator)

### C8: Literacy-Anchored Prompting

```text
You are role-playing a patient in a medical consultation.

Your identity:
- You are a {age}-year-old {gender} patient who works as {occupation}.
- You are visiting a {setting} for a {urgency} visit.
- Your reason for coming: {chief_complaint}

Your communication style is: {communication_style}.
Your health literacy level is: {health_literacy}.

[BEHAVIORAL GUIDELINES FOR YOUR HEALTH LITERACY LEVEL]

If your health literacy is LOW:
- You tend to use very simple language and struggle to explain your symptoms precisely.
- You rarely ask the doctor questions—you assume the doctor knows best.
- When the doctor asks if you understand, you usually say "yes" even when you don't.
- You let the doctor make decisions and rarely state preferences.
- Your responses are short, and you avoid giving detailed descriptions unless asked directly.

If your health literacy is MEDIUM:
- You can describe your symptoms adequately but may sometimes be imprecise.
- You ask a few questions but mainly when prompted.
- You sometimes state preferences but often defer to the doctor.
- Your responses are moderate in length.

If your health literacy is HIGH:
- You use appropriate medical terms and describe your condition clearly.
- You ask questions and want to understand your condition.
- You express your preferences and participate in decision-making.
- Your responses are detailed and structured.

ABSOLUTE RULES:
- Stay in character at all times. You are the patient, never break the fourth wall.
- Answer naturally, like a real person talking to their doctor.
- Never mention your clinical attributes explicitly.
- Never reveal that you are an AI.
- Keep responses concise (1-4 sentences per turn).
```

## Evaluation Framework

### Behavioral-Level Metrics

Each dialogue is scored on five behavioral dimensions:

| Dimension | Metric | Low HL Expected | High HL Expected |
|-----------|--------|----------------|------------------|
| **Question-asking** | Count of patient-initiated questions | 4–5 per encounter | 7–9 per encounter |
| **Response complexity** | Words per response; lexical diversity (TTR) | Short, low TTR | Longer, higher TTR |
| **Jargon usage** | Medical term count from a curated list | Near zero | Moderate |
| **Preference expression** | Binary presence of preference statements per turn | Rare (<20% of turns) | Frequent (>40% of turns) |
| **Comprehension signaling** | Rate of unqualified "yes" to comprehension checks | High (>50%) | Low (<20%) |
| **Decision role** | Passive (defers) vs. active (participates) per turn | Mostly passive | Mixed active |

### Attribute-Level Metrics

- Health literacy level classification accuracy (3-class confusion matrix)
- Communication style classification accuracy (7 categories)

### Temporal Dynamics

For each dialogue, we compute per-turn behavioral scores in a rolling window of 3 turns and classify the trajectory:
- **Consistent**: behavioral scores remain within ±1 SD of the mean throughout
- **Abrupt contradiction**: single-turn score deviates >2 SD from the rolling mean
- **Gradual drift**: significant slope in a linear regression of score over turn number
- **Prompt-sensitive**: specific clinician questions consistently trigger score deviation

### Statistical Analysis

**Primary analysis**: 3 (generation family) × 3 (HL level) mixed ANOVA
- Family: Simple prompting / Deployed system / Behavioral specification
- Within-subject: HL level (nested within profiles)
- DV: Composite behavioral faithfulness score

**Secondary analyses**:
- Per-dimension ANOVA to identify which behaviors are most sensitive to strategy choice
- Logistic regression to predict drift type from generation condition
- Confusion matrix symmetry analysis for attribute-level reconstruction

## Data Requirements

| Item | Count | Format |
|------|-------|--------|
| Patient profiles | 24 | JSON (existing `patient_profiles.json`) |
| Existing dialogues | 72 | JSON (existing in `data/dialogues/`) |
| New dialogues to generate | 120 | JSON |
| API calls needed | ~1,800 | Each dialogue ~15 patient turns × 2 models |
| Estimated tokens (generation) | ~1.2M input, ~360K output | Mainly system prompts + clinician questions |
| Estimated tokens (evaluation) | ~960K | 192 dialogues × ~5K tokens per eval call |

## Expected Outcomes

| Scenario | Prediction | Implication |
|----------|-----------|-------------|
| C1–C3 show no HL differentiation | HL consistency is not automatic | Confirms problem exists |
| C4–C5 show no HL differentiation | Existing persona systems don't solve it | Contribution stands against SOTA |
| C6–C7 show partial differentiation | Multi-agent architectures help but insufficient | Reveals where architecture helps vs. doesn't |
| C8 shows significant improvement | Behavioral specification works | Main positive result |
| All conditions show low-HL hardest to simulate | Fundamental limitation of LLM role-playing | Important qualification for clinical use |
