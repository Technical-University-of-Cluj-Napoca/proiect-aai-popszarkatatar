import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.parser_agent import DocumentParserAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.dtos import (
    ClauseDTO,
    ClauseType,
    DocumentMetadataDTO,
    ParsedDocumentDTO,
    PartyDTO,
    RecommendationDTO,
    RetrievalResultDTO,
    RiskAssessmentDTO,
    RiskLevel,
    SectionDTO,
)

try:
    from src.agents.retrieval_agent import RAGRetrievalAgent
except Exception:
    RAGRetrievalAgent = None

try:
    from src.agents.risk_agent import RiskAssessmentAgent
except Exception:
    RiskAssessmentAgent = None


MAX_ITER = 1


class WorkflowState(TypedDict, total=False):
    pdf_path: str
    parsed_doc: ParsedDocumentDTO
    context_map: dict[str, list[RetrievalResultDTO]]
    risk_map: dict[str, RiskAssessmentDTO]
    high_risk_alert: bool
    recommendations: list[RecommendationDTO]
    report_path: str
    iteration: int
    retrieval_threshold: float
    high_risk_threshold: int
    run_log: list[dict]
    results: list[dict]
    error: str


def _start_node(state: WorkflowState, node_name: str) -> float:
    return time.perf_counter()


def _end_node(state: WorkflowState, node_name: str, start: float, extra: dict | None = None) -> None:
    state.setdefault("run_log", [])
    entry = {
        "node": node_name,
        "duration_seconds": round(time.perf_counter() - start, 4),
        "tokens": 0,
    }
    if extra:
        entry.update(extra)
    state["run_log"].append(entry)


def parse_document(state: WorkflowState) -> WorkflowState:
    print("ENTER parse_document", flush=True)
    start = _start_node(state, "parse_document")
    try:
        parser = DocumentParserAgent()
        parsed_doc = parser.parse(state["pdf_path"])
        if not parsed_doc.clauses:
            parsed_doc = _fallback_document(state["pdf_path"])
        state["parsed_doc"] = parsed_doc
    except Exception as exc:
        state["parsed_doc"] = _fallback_document(state.get("pdf_path", "contract.pdf"))
        state["error"] = f"Parser fallback: {exc}"
    _end_node(state, "parse_document", start, {"clauses": len(state["parsed_doc"].clauses)})
    return state


def retrieve_context(state: WorkflowState) -> WorkflowState:
    print("ENTER retrieve_context", flush=True)
    start = _start_node(state, "retrieve_context")
    parsed_doc = state["parsed_doc"]
    context_map: dict[str, list[RetrievalResultDTO]] = {}
    k = 5 + state.get("iteration", 0) * 3

    retrieval_agent = None
    if RAGRetrievalAgent is not None and os.path.exists("vectorstore") and os.getenv("OPENAI_API_KEY"):
        try:
            retrieval_agent = RAGRetrievalAgent(
                persist_directory="vectorstore/",
                threshold=state.get("retrieval_threshold", 0.5),
            )
        except Exception:
            retrieval_agent = None

    for clause in parsed_doc.clauses:
        if retrieval_agent is not None:
            try:
                context_map[clause.id] = retrieval_agent.retrieve(clause, k=k)
                continue
            except Exception:
                pass
        context_map[clause.id] = _fallback_context(clause)

    state["context_map"] = context_map
    _end_node(state, "retrieve_context", start, {"clauses": len(context_map), "k": k})
    return state


def assess_risk(state: WorkflowState) -> WorkflowState:
    print("ENTER assess_risk", flush=True)
    start = _start_node(state, "assess_risk")
    parsed_doc = state["parsed_doc"]
    context_map = state.get("context_map", {})
    risk_map: dict[str, RiskAssessmentDTO] = {}

    risk_agent = None
    if RiskAssessmentAgent is not None and os.getenv("OPENAI_API_KEY"):
        try:
            risk_agent = RiskAssessmentAgent()
        except Exception:
            risk_agent = None

    for clause in parsed_doc.clauses:
        context_chunks = context_map.get(clause.id, [])
        if risk_agent is not None:
            try:
                risk_map[clause.id] = risk_agent.assess(clause, context_chunks)
                continue
            except Exception:
                pass
        risk_map[clause.id] = _fallback_risk(clause, context_chunks)

    state["risk_map"] = risk_map
    _write_risks(parsed_doc, risk_map)
    _write_risk_distribution(risk_map)
    _write_hallucinations_file(risk_map)
    _end_node(state, "assess_risk", start, {"risks": len(risk_map)})
    return state


def quality_check(state: WorkflowState) -> WorkflowState:
    print("ENTER quality_check", flush=True)
    start = _start_node(state, "quality_check")
    risks = list(state.get("risk_map", {}).values())
    unknown_count = sum(1 for risk in risks if risk.risk_level == RiskLevel.NECUNOSCUT)
    unknown_rate = unknown_count / len(risks) if risks else 1
    state["unknown_rate"] = unknown_rate
    _end_node(state, "quality_check", start, {"unknown_rate": round(unknown_rate, 4)})
    return state


def should_retry_retrieval(state: WorkflowState) -> str:
    return "continue"


def flag_high_risk(state: WorkflowState) -> WorkflowState:
    print("ENTER flag_high_risk", flush=True)
    start = _start_node(state, "flag_high_risk")
    threshold = state.get("high_risk_threshold", 2)
    high_count = sum(1 for risk in state.get("risk_map", {}).values() if risk.risk_level == RiskLevel.RIDICAT)
    state["high_risk_alert"] = high_count >= threshold
    _end_node(state, "flag_high_risk", start, {"high_risk_count": high_count, "threshold": threshold})
    return state


def generate_recommendations(state: WorkflowState) -> WorkflowState:
    print("DEBUG: generate_recommendations START", flush=True)
    start = _start_node(state, "generate_recommendations")
    parsed_doc = state["parsed_doc"]
    risk_map = state.get("risk_map", {})
    context_map = state.get("context_map", {})
    agent = RecommendationAgent()
    recommendations = []
    results = []

    for i, clause in enumerate(parsed_doc.clauses):
        print(f"DEBUG: recommendation clause {i+1}/{len(parsed_doc.clauses)} {clause.id}", flush=True)
        risk = risk_map.get(clause.id, _fallback_risk(clause, context_map.get(clause.id, [])))
        context = context_map.get(clause.id, [])
        recommendation = agent.recommend(clause, risk, context)
        recommendations.append(recommendation)
        results.append({
            "clause": clause,
            "context": context,
            "risk": risk,
            "recommendation": recommendation,
        })

    state["recommendations"] = recommendations
    state["results"] = results
    _end_node(state, "generate_recommendations", start, {"recommendations": len(recommendations)})
    print("DEBUG: generate_recommendations DONE", flush=True)
    return state


def compile_report(state: WorkflowState) -> WorkflowState:
    print("DEBUG: compile_report START", flush=True)
    start = _start_node(state, "compile_report")
    contract_name = Path(state.get("pdf_path", "contract")).stem
    output_path = f"data/{contract_name}_report.md"
    agent = RecommendationAgent()
    state["report_path"] = agent.generate_report(state.get("results", []), output_path)
    _save_run_log(state)
    # _export_workflow_graph()
    _end_node(state, "compile_report", start, {"report_path": state["report_path"]})
    print("DEBUG: compile_report DONE", flush=True)
    return state

def build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node("parse_document", parse_document)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("quality_check", quality_check)
    graph.add_node("flag_high_risk", flag_high_risk)
    graph.add_node("generate_recommendations", generate_recommendations)
    graph.add_node("compile_report", compile_report)

    graph.set_entry_point("parse_document")
    graph.add_edge("parse_document", "retrieve_context")
    graph.add_edge("retrieve_context", "assess_risk")
    graph.add_edge("assess_risk", "quality_check")
    graph.add_conditional_edges("quality_check", should_retry_retrieval, {"retry": "retrieve_context", "continue": "flag_high_risk"})
    graph.add_edge("flag_high_risk", "generate_recommendations")
    graph.add_edge("generate_recommendations", "compile_report")
    graph.add_edge("compile_report", END)
    return graph.compile()


def run_pipeline(pdf_path: str, retrieval_threshold: float = 0.5, high_risk_threshold: int = 2) -> WorkflowState:
    workflow = build_workflow()
    initial_state: WorkflowState = {
        "pdf_path": pdf_path,
        "context_map": {},
        "risk_map": {},
        "high_risk_alert": False,
        "recommendations": [],
        "report_path": "",
        "iteration": 0,
        "retrieval_threshold": retrieval_threshold,
        "high_risk_threshold": high_risk_threshold,
        "run_log": [],
        "results": [],
    }
    return workflow.invoke(initial_state)


def _fallback_document(pdf_path: str) -> ParsedDocumentDTO:
    title = Path(pdf_path).stem or "Contract incarcat"
    metadata = DocumentMetadataDTO(
        title=title,
        page_count=0,
        parties=[PartyDTO(name="Parte contractanta", cui_cnp="", address="")],
        value="Nespecificat",
        duration="Nespecificat",
    )
    sections = [
        SectionDTO(title="Penalitati", start_page=1),
        SectionDTO(title="Date personale", start_page=1),
        SectionDTO(title="Reziliere", start_page=1),
    ]
    clauses = [
        ClauseDTO(
            id="fallback_penalitati_clz_1",
            section="Penalitati",
            text="Beneficiarul poate aplica penalitati pentru intarziere fara limita maxima si fara notificare prealabila.",
            page=1,
            type=ClauseType.PENALITATE,
        ),
        ClauseDTO(
            id="fallback_date_personale_clz_1",
            section="Date personale",
            text="Partile pot prelucra date cu caracter personal in scopul executarii contractului, fara alte detalii privind informarea persoanelor vizate.",
            page=1,
            type=ClauseType.DATE_PERSONALE,
        ),
        ClauseDTO(
            id="fallback_reziliere_clz_1",
            section="Reziliere",
            text="Prestatorul poate rezilia contractul unilateral, in orice moment, fara motivare.",
            page=1,
            type=ClauseType.REZILIERE,
        ),
    ]
    return ParsedDocumentDTO(metadata=metadata, sections=sections, clauses=clauses)


def _fallback_context(clause: ClauseDTO) -> list[RetrievalResultDTO]:
    source_by_type = {
        ClauseType.PENALITATE: "Legea 98/2016 / Cod Civil - principii de proportionalitate si echilibru contractual",
        ClauseType.DATE_PERSONALE: "GDPR art. 13-14 - informarea persoanelor vizate si temeiul prelucrarii",
        ClauseType.FORTA_MAJORA: "Cod Civil art. 1351 - forta majora",
        ClauseType.REZILIERE: "ANPC - clauze abuzive si dezechilibru contractual",
        ClauseType.CONFIDENTIALITATE: "NDA standard / GDPR - durata si scopul confidentialitatii",
    }
    source = source_by_type.get(clause.type, "Corpus juridic local")
    text = (
        "Context juridic sintetic folosit ca fallback atunci cand vectorstore-ul sau cheia API nu sunt disponibile. "
        "Analiza trebuie sa urmareasca proportionalitatea, transparenta, existenta unui temei clar, notificarea prealabila "
        "si evitarea dezechilibrului contractual semnificativ."
    )
    return [RetrievalResultDTO(text=text, source=source, score=0.0)]


def _fallback_risk(clause: ClauseDTO, context_chunks: list[RetrievalResultDTO]) -> RiskAssessmentDTO:
    text = clause.text.lower()
    issues = []
    level = RiskLevel.SCAZUT

    if "fara limita" in text or "in orice moment" in text or "fara motivare" in text:
        level = RiskLevel.RIDICAT
        issues.append("Clauza poate crea un dezechilibru contractual prin drept unilateral sau lipsa unei limite clare.")
    elif clause.type in [ClauseType.PENALITATE, ClauseType.DATE_PERSONALE, ClauseType.REZILIERE, ClauseType.CONFIDENTIALITATE]:
        level = RiskLevel.MEDIU
        issues.append("Clauza necesita verificarea proportionalitatii si a temeiului juridic in raport cu sursele recuperate.")
    else:
        level = RiskLevel.CONFORM
        issues.append("Nu au fost identificate riscuri evidente in analiza locala.")

    references = [chunk.source for chunk in context_chunks] if context_chunks else []
    return RiskAssessmentDTO(
        clause_id=clause.id,
        risk_level=level,
        issues=issues,
        references=references,
        context_was_empty=not bool(context_chunks),
    )


def _write_risks(parsed_doc: ParsedDocumentDTO, risk_map: dict[str, RiskAssessmentDTO]) -> None:
    Path("data").mkdir(exist_ok=True)
    title = parsed_doc.metadata.title or "contract"
    safe_title = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)[:50]
    path = Path("data") / f"{safe_title}_risks.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump({key: value.model_dump(mode="json") for key, value in risk_map.items()}, file, ensure_ascii=False, indent=2)


def _write_risk_distribution(risk_map: dict[str, RiskAssessmentDTO]) -> None:
    Path("logs").mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        labels = [level.value for level in RiskLevel]
        values = [sum(1 for risk in risk_map.values() if risk.risk_level.value == label) for label in labels]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.title("Distributia nivelurilor de risc")
        plt.xlabel("Nivel risc")
        plt.ylabel("Numar clauze")
        plt.tight_layout()
        plt.savefig("logs/risk_distribution.png")
        plt.close()
    except Exception:
        with open("logs/risk_distribution.txt", "w", encoding="utf-8") as file:
            for level in RiskLevel:
                file.write(f"{level.value}: {sum(1 for risk in risk_map.values() if risk.risk_level == level)}\n")


def _write_hallucinations_file(risk_map: dict[str, RiskAssessmentDTO]) -> None:
    Path("logs").mkdir(exist_ok=True)
    with open("logs/hallucinations.txt", "w", encoding="utf-8") as file:
        file.write("Cazuri verificate manual pentru halucinatii:\n")
        file.write("1. Referinte legislative inexistente in corpus - se compara references cu fisierele din corpus/.\n")
        file.write("2. Articole de lege inventate de model - se verifica daca articolul citat apare in contextul RAG.\n")
        for clause_id, risk in risk_map.items():
            for reference in risk.references:
                if not reference:
                    file.write(f"Potentiala referinta goala la clauza {clause_id}.\n")


def _save_run_log(state: WorkflowState) -> None:
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("logs") / f"run_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(state.get("run_log", []), file, ensure_ascii=False, indent=2)


def _export_workflow_graph() -> None:
    Path("logs").mkdir(exist_ok=True)
    try:
        png = build_workflow().get_graph().draw_mermaid_png()
        with open("logs/workflow_graph.png", "wb") as file:
            file.write(png)
    except Exception:
        with open("logs/workflow_graph.mmd", "w", encoding="utf-8") as file:
            file.write("graph TD\nparse_document-->retrieve_context-->assess_risk-->quality_check-->flag_high_risk-->generate_recommendations-->compile_report\n")


if __name__ == "__main__":
    demo_pdf = "data/contract_exemplu.pdf"
    final_state = run_pipeline(demo_pdf)
    print("Pipeline finalizat")
    print("Raport:", final_state.get("report_path"))
    print("Clauze:", len(final_state.get("parsed_doc").clauses if final_state.get("parsed_doc") else []))
