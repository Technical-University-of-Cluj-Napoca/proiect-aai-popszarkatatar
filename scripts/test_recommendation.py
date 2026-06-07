import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.recommendation_agent import RecommendationAgent
from src.dtos import ClauseDTO, ClauseType, RetrievalResultDTO, RiskAssessmentDTO, RiskLevel

clause = ClauseDTO(
    id="test_1",
    section="Penalitati",
    text="Prestatorul datoreaza penalitati nelimitate pentru orice intarziere.",
    page=1,
    type=ClauseType.PENALITATE,
)
risk = RiskAssessmentDTO(
    clause_id="test_1",
    risk_level=RiskLevel.RIDICAT,
    issues=["Penalitatile sunt nelimitate si pot crea dezechilibru contractual."],
    references=["legea_98_2016.pdf"],
)
context = [RetrievalResultDTO(text="Penalitatile trebuie sa fie proportionale.", source="legea_98_2016.pdf", score=0.8)]
rec = RecommendationAgent().recommend(clause, risk, context)
print(rec.model_dump_json(indent=2))
