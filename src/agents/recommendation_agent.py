import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.dtos import ClauseDTO, RecommendationDTO, RetrievalResultDTO, RiskAssessmentDTO, RiskLevel

load_dotenv()


class RecommendationAgent:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.parser = JsonOutputParser(pydantic_object=RecommendationDTO)
        self.llm = None
        if os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                request_timeout=20,
                max_retries=1
            )

    def recommend(
        self,
        clause: ClauseDTO,
        risk_assessment: RiskAssessmentDTO,
        context_chunks: list[RetrievalResultDTO],
    ) -> RecommendationDTO:
        if risk_assessment.risk_level not in [RiskLevel.RIDICAT, RiskLevel.MEDIU]:
            return RecommendationDTO(
                clause_id=clause.id,
                original_text=clause.text,
                reformulated_text="",
                explanation="Nu se genereaza reformulare deoarece nivelul de risc nu este RIDICAT sau MEDIU.",
                sources=self._sources(context_chunks),
                candidates=None,
            )

        if not self.llm:
            return self._fallback_recommendation(clause, risk_assessment, context_chunks)

        if risk_assessment.risk_level == RiskLevel.RIDICAT:
            return self._fallback_recommendation(
                clause,
                risk_assessment,
                context_chunks,
        )

        return self._generate_recommendation(clause, risk_assessment, context_chunks)

    def generate_report(self, results: list[dict], output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Raport analiza contract juridic",
            "",
            "## Rezumat",
            "",
        ]

        total = len(results)
        high = sum(1 for item in results if self._risk_value(item.get("risk")) == RiskLevel.RIDICAT.value)
        medium = sum(1 for item in results if self._risk_value(item.get("risk")) == RiskLevel.MEDIU.value)
        unknown = sum(1 for item in results if self._risk_value(item.get("risk")) == RiskLevel.NECUNOSCUT.value)

        lines.extend([
            f"- Numar total de clauze analizate: {total}",
            f"- Clauze cu risc ridicat: {high}",
            f"- Clauze cu risc mediu: {medium}",
            f"- Clauze necunoscute: {unknown}",
            "",
            "## Clauze analizate",
            "",
        ])

        for item in results:
            clause = item.get("clause")
            risk = item.get("risk")
            recommendation = item.get("recommendation")
            context = item.get("context", [])

            clause_id = getattr(clause, "id", "clauza")
            section = getattr(clause, "section", "")
            clause_text = getattr(clause, "text", "")
            risk_level = self._risk_value(risk)
            issues = getattr(risk, "issues", []) if risk is not None else []
            references = getattr(risk, "references", []) if risk is not None else []
            reformulated = getattr(recommendation, "reformulated_text", "") if recommendation is not None else ""
            explanation = getattr(recommendation, "explanation", "") if recommendation is not None else ""
            sources = getattr(recommendation, "sources", []) if recommendation is not None else self._sources(context)

            lines.extend([
                f"### {clause_id}",
                "",
                f"**Sectiune:** {section}",
                "",
                f"**Nivel risc:** {risk_level}",
                "",
                "**Text original:**",
                "",
                clause_text,
                "",
            ])

            if issues:
                lines.append("**Probleme identificate:**")
                lines.append("")
                lines.extend([f"- {issue}" for issue in issues])
                lines.append("")

            if references:
                lines.append("**Referinte juridice:**")
                lines.append("")
                lines.extend([f"- {ref}" for ref in references])
                lines.append("")

            if reformulated:
                lines.extend([
                    "**Reformulare propusa:**",
                    "",
                    reformulated,
                    "",
                ])

            if explanation:
                lines.extend([
                    "**Explicatie:**",
                    "",
                    explanation,
                    "",
                ])

            if sources:
                lines.append("**Surse folosite:**")
                lines.append("")
                lines.extend([f"- {source}" for source in sources])
                lines.append("")

            lines.append("---")
            lines.append("")

        report = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(report)
        return output_path

    def _generate_recommendation(
        self,
        clause: ClauseDTO,
        risk_assessment: RiskAssessmentDTO,
        context_chunks: list[RetrievalResultDTO],
    ) -> RecommendationDTO:
        prompt = self._prompt()
        chain = prompt | self.llm | self.parser
        try:
            result = chain.invoke(self._prompt_input(clause, risk_assessment, context_chunks))
            return RecommendationDTO(**result)
        except Exception:
            return self._fallback_recommendation(clause, risk_assessment, context_chunks)

    def _generate_candidate(
        self,
        clause: ClauseDTO,
        risk_assessment: RiskAssessmentDTO,
        context_chunks: list[RetrievalResultDTO],
        candidate_number: int,
    ) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Esti jurist roman. Propui o singura reformulare clara, echilibrata contractual si ancorata in sursele primite."),
            ("human", "Clauza:\n{clause}\n\nRisc:\n{risk}\n\nContext:\n{context}\n\nGenereaza varianta {candidate_number} de reformulare. Raspunde doar cu textul reformulat."),
        ])
        try:
            response = (prompt | self.llm).invoke({
                "clause": clause.text,
                "risk": risk_assessment.model_dump_json(),
                "context": self._legal_context(context_chunks),
                "candidate_number": candidate_number,
            })
            return response.content.strip()
        except Exception:
            return self._fallback_text(clause, risk_assessment)

    def _select_best_candidate(
        self,
        clause: ClauseDTO,
        risk_assessment: RiskAssessmentDTO,
        context_chunks: list[RetrievalResultDTO],
        candidates: list[str],
    ) -> RecommendationDTO:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Alegi cea mai buna reformulare juridica dintre candidati si returnezi JSON valid pentru RecommendationDTO."),
            ("human", "Clauza originala:\n{clause}\n\nRisc:\n{risk}\n\nContext:\n{context}\n\nCandidati:\n{candidates}\n\nReturneaza JSON cu campurile: clause_id, original_text, reformulated_text, explanation, sources, candidates."),
        ])
        chain = prompt | self.llm | self.parser
        try:
            result = chain.invoke({
                "clause": clause.text,
                "risk": risk_assessment.model_dump_json(),
                "context": self._legal_context(context_chunks),
                "candidates": json.dumps(candidates, ensure_ascii=False),
            })
            return RecommendationDTO(**result)
        except Exception:
            return RecommendationDTO(
                clause_id=clause.id,
                original_text=clause.text,
                reformulated_text=candidates[0] if candidates else self._fallback_text(clause, risk_assessment),
                explanation="A fost selectata varianta cea mai prudenta disponibila. Selectia automata cu LLM nu a putut fi finalizata.",
                sources=self._sources(context_chunks),
                candidates=candidates,
            )

    def _prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
Esti un asistent juridic pentru contracte in limba romana.
Generezi recomandari numai pe baza contextului juridic furnizat.
Nu inventa legi, articole sau surse.
Stilul trebuie sa fie juridic, formal, clar si echilibrat contractual.
Returneaza doar JSON valid cu structura RecommendationDTO:
{
  "clause_id": "...",
  "original_text": "...",
  "reformulated_text": "...",
  "explanation": "...",
  "sources": [],
  "candidates": null
}
"""),
            ("human", "Clauza:\n{clause}\n\nEvaluare risc:\n{risk}\n\nContext juridic:\n{context}")
        ])

    def _prompt_input(
        self,
        clause: ClauseDTO,
        risk_assessment: RiskAssessmentDTO,
        context_chunks: list[RetrievalResultDTO],
    ) -> dict:
        return {
            "clause": clause.text,
            "risk": risk_assessment.model_dump_json(),
            "context": self._legal_context(context_chunks),
        }

    def _fallback_recommendation(
        self,
        clause: ClauseDTO,
        risk_assessment: RiskAssessmentDTO,
        context_chunks: list[RetrievalResultDTO],
    ) -> RecommendationDTO:
        return RecommendationDTO(
            clause_id=clause.id,
            original_text=clause.text,
            reformulated_text=self._fallback_text(clause, risk_assessment),
            explanation="Reformulare generata local, ca fallback, pe baza problemelor identificate si a principiului de echilibru contractual. Pentru citare juridica exacta este necesar context RAG disponibil.",
            sources=self._sources(context_chunks),
            candidates=[self._fallback_text(clause, risk_assessment)] if risk_assessment.risk_level == RiskLevel.RIDICAT else None,
        )

    def _fallback_text(self, clause: ClauseDTO, risk_assessment: RiskAssessmentDTO) -> str:
        issues = "; ".join(risk_assessment.issues) if risk_assessment.issues else "riscul identificat"
        return (
            f"Partile vor aplica prezenta clauza cu respectarea legislatiei aplicabile, intr-un mod proportional, "
            f"transparent si echilibrat. Orice drept unilateral, penalitate, limitare de raspundere sau incetare a "
            f"contractului va fi exercitata numai cu notificare prealabila, termen rezonabil de remediere si justificare "
            f"obiectiva. Reformularea are in vedere: {issues}."
        )

    def _legal_context(self, context_chunks: list[RetrievalResultDTO]) -> str:
        return "\n\n".join([f"SURSA: {chunk.source}\n{chunk.text}" for chunk in context_chunks])

    def _sources(self, context_chunks: Iterable[RetrievalResultDTO]) -> list[str]:
        sources = []
        for chunk in context_chunks:
            source = getattr(chunk, "source", "unknown")
            if source not in sources:
                sources.append(source)
        return sources

    def _risk_value(self, risk) -> str:
        if risk is None:
            return RiskLevel.NECUNOSCUT.value
        value = getattr(risk, "risk_level", risk)
        return value.value if hasattr(value, "value") else str(value)
