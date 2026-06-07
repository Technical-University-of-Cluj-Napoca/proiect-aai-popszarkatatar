from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

class ClauseType(str, Enum):
    PENALITATE = "penalitate"
    OBLIGATIE = "obligatie"
    DREPT = "drept"
    FORTA_MAJORA = "forta_majora"
    CONFIDENTIALITATE = "confidentialitate"
    REZILIERE = "reziliere"
    DATE_PERSONALE = "date_personale"
    ALTELE = "altele"

class RiskLevel(str, Enum):
    RIDICAT = "RIDICAT"
    MEDIU = "MEDIU"
    SCAZUT = "SCAZUT"
    CONFORM = "CONFORM"
    NECUNOSCUT = "NECUNOSCUT"

class PartyDTO(BaseModel):
    name: str
    cui_cnp: str
    address: str

class SectionDTO(BaseModel):
    title: str
    start_page: int

class ClauseDTO(BaseModel):
    id: str
    section: str
    text: str
    page: int
    type: ClauseType

class DocumentMetadataDTO(BaseModel):
    title: str
    page_count: int
    parties: List[PartyDTO]
    signing_date: Optional[str] = None
    effective_date: Optional[str] = None
    value: str
    duration: str

class ParsedDocumentDTO(BaseModel):
    metadata: DocumentMetadataDTO
    sections: List[SectionDTO]
    clauses: List[ClauseDTO]

class RetrievalResultDTO(BaseModel):
    text: str
    source: str
    score: float

class RiskAssessmentDTO(BaseModel):
    clause_id: str
    risk_level: RiskLevel
    issues: List[str]
    references: List[str]
    context_was_empty: bool = False

class RecommendationDTO(BaseModel):
    clause_id: str
    original_text: str
    reformulated_text: str
    explanation: str
    sources: List[str]
    candidates: Optional[List[str]] = None