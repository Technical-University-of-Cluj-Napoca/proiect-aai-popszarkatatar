from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.dtos import (
    ClauseDTO,
    RetrievalResultDTO,
    RiskAssessmentDTO,
    RiskLevel
)


class RiskAssessmentAgent:

    def __init__(self):

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            request_timeout=20,
            max_retries=1
        )

        self.parser = JsonOutputParser(
            pydantic_object=RiskAssessmentDTO
        )

    def assess(
        self,
        clause: ClauseDTO,
        context_chunks: list[RetrievalResultDTO]
    ) -> RiskAssessmentDTO:

        if len(context_chunks) == 0:

            return RiskAssessmentDTO(
                clause_id=clause.id,
                risk_level=RiskLevel.NECUNOSCUT,
                issues=[],
                references=[],
                context_was_empty=True
            )

        legal_context = "\n\n".join(
            [
                f"SURSA: {chunk.source}\n{chunk.text}"
                for chunk in context_chunks
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Esti expert juridic roman.

Clasifica fiecare clauza intr-una dintre:

RIDICAT
MEDIU
SCAZUT
CONFORM
NECUNOSCUT

Explica pe scurt motivele.

Foloseste exclusiv contextul furnizat.
Nu inventa legi.
Nu inventa articole.
Nu inventa surse.

Returneaza doar JSON valid.

                   Structura JSON:

{{
  "clause_id": "...",
  "risk_level": "RIDICAT | MEDIU | SCAZUT | CONFORM | NECUNOSCUT",
  "issues": [],
  "references": [],
  "context_was_empty": false
}}
                    """
                ),
                (
                    "human",
                    """
                    CLAUZA:

                    {clause}

                    CONTEXT JURIDIC:

                    {context}
                    """
                )
            ]
        )

        chain = (
            prompt
            | self.llm
            | self.parser
        )

        try:

            result = chain.invoke(
                {
                    "clause": clause.text,
                    "context": legal_context
                }
            )

            return RiskAssessmentDTO(
                **result
            )

        except Exception as e:

            print(f"Risk assessment error: {e}")

            return RiskAssessmentDTO(
                clause_id=clause.id,
                risk_level=RiskLevel.NECUNOSCUT,
                issues=[],
                references=[],
                context_was_empty=False
            )