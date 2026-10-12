"""
Clinical Entity and Negation/Certainty Extractor.

Extracts medical observations, anatomical locations, negation status (NegEx algorithm),
and clinical certainty levels from radiological findings and statements.
"""

from enum import Enum
import re
from typing import List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    OBSERVATION = "Observation"
    ANATOMY = "Anatomy"
    SEVERITY = "Severity"


class Polarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class Certainty(str, Enum):
    CONFIRMED = "CONFIRMED"
    PROBABLE = "PROBABLE"
    POSSIBLE = "POSSIBLE"
    ABSENT = "ABSENT"


class ClinicalEntity(BaseModel):
    """Extracted clinical semantic entity."""
    text: str = Field(..., description="Entity surface mention")
    canonical_name: str = Field(..., description="Standardized finding or anatomical term")
    entity_type: EntityType = Field(..., description="Observation, Anatomy, or Severity")
    polarity: Polarity = Field(Polarity.POSITIVE, description="Positive vs Negated")
    certainty: Certainty = Field(Certainty.PROBABLE, description="Certainty level")
    anatomical_site: Optional[str] = Field(None, description="Linked anatomical region if resolved")
    context_span: str = Field("", description="Surrounding sentence context")


# Clinical Finding Dictionary
KNOWN_FINDINGS = {
    "pneumonia": ["pneumonia", "consolidation", "infiltrate", "infiltrates", "bronchopneumonia"],
    "pleural_effusion": ["pleural effusion", "effusion", "fluid collection", "blunted costophrenic", "costophrenic blunting"],
    "pneumothorax": ["pneumothorax", "pleural air", "visceral pleural line"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement", "prominent cardiac silhouette"],
    "atelectasis": ["atelectasis", "lung collapse", "volume loss", "discoidal atelectasis", "subsegmental collapse"],
    "pulmonary_edema": ["pulmonary edema", "alveolar edema", "interstitial edema", "fluid overload", "vascular congestion"],
    "consolidation": ["consolidation", "airspace consolidation", "dense opacity"],
    "normal": ["normal", "unremarkable", "clear lungs", "healthy lungs", "no acute cardiopulmonary abnormality"],
}

# Anatomical Regions Dictionary
KNOWN_ANATOMY = {
    "lung_fields": ["lung", "lungs", "parenchyma", "pulmonary", "bilateral lung fields", "airspace"],
    "costophrenic_angles": ["costophrenic angle", "costophrenic angles", "costophrenic sulcus"],
    "cardiac_silhouette": ["cardiac silhouette", "heart", "pericardium", "mediastinum"],
    "pleural_space": ["pleura", "pleural space", "chest wall"],
    "hilar_structures": ["hilum", "hila", "perihilar", "bronchovascular"],
}

# Negation Triggers (NegEx inspired)
PRE_NEGATION_TRIGGERS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bno evidence of\b",
    r"\bno signs of\b",
    r"\bfree of\b",
    r"\brules out\b",
    r"\bruling out\b",
    r"\bclear of\b",
    r"\babsence of\b",
    r"\bno focal\b",
]

# Certainty Modifiers
CERTAINTY_TRIGGERS = {
    Certainty.CONFIRMED: [r"\bmarked\b", r"\bsevere\b", r"\bdefinite\b", r"\bconfirmed\b", r"\bprominent\b"],
    Certainty.PROBABLE: [r"\bsuggestive of\b", r"\bconsistent with\b", r"\blikely\b", r"\bprobable\b", r"\bsuspicious for\b"],
    Certainty.POSSIBLE: [r"\bpossible\b", r"\bpotential\b", r"\bcannot exclude\b", r"\bquestionable\b", r"\bborderline\b", r"\bmay represent\b"],
}


def extract_entities_from_text(text: str) -> List[ClinicalEntity]:
    """Parse text and extract standardized clinical entities with polarity and certainty."""
    sentences = re.split(r"[.;\n]", text)
    entities: List[ClinicalEntity] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_lower = sentence.lower()

        # Check for anatomical locations mentioned in sentence
        linked_anatomy = None
        for anat_key, anat_terms in KNOWN_ANATOMY.items():
            for term in anat_terms:
                if re.search(r"\b" + re.escape(term) + r"\b", sentence_lower):
                    linked_anatomy = anat_key
                    break
            if linked_anatomy:
                break

        # Check for clinical observation findings
        for finding_key, finding_terms in KNOWN_FINDINGS.items():
            for term in finding_terms:
                match = re.search(r"\b" + re.escape(term) + r"\b", sentence_lower)
                if match:
                    # Check for negation preceding the mention
                    prefix = sentence_lower[: match.start()]
                    is_negated = any(re.search(trigger, prefix) for trigger in PRE_NEGATION_TRIGGERS)

                    # Determine certainty level
                    if is_negated:
                        certainty = Certainty.ABSENT
                        polarity = Polarity.NEGATIVE
                    else:
                        polarity = Polarity.POSITIVE
                        certainty = Certainty.PROBABLE
                        for cert_level, cert_patterns in CERTAINTY_TRIGGERS.items():
                            if any(re.search(pat, sentence_lower) for pat in cert_patterns):
                                certainty = cert_level
                                break

                    entities.append(
                        ClinicalEntity(
                            text=match.group(0),
                            canonical_name=finding_key,
                            entity_type=EntityType.OBSERVATION,
                            polarity=polarity,
                            certainty=certainty,
                            anatomical_site=linked_anatomy,
                            context_span=sentence,
                        )
                    )
                    break  # Avoid duplicate matches for same finding group in single sentence

    return entities