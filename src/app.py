import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.dtos import RiskLevel
from src.graph.workflow import run_pipeline

load_dotenv()

st.set_page_config(page_title="Legal Contract Analyzer", layout="wide")


RISK_COLORS = {
    "RIDICAT": "background-color: #ffe3e3; color: #000000;",
    "MEDIU": "background-color: #fff3bf; color: #000000;",
    "SCAZUT": "background-color: #fff9c4; color: #000000;",
    "CONFORM": "background-color: #d3f9d8; color: #000000;",
    "NECUNOSCUT": "background-color: #f8f9fa; color: #000000;",
}


def style_risk(value):
    return RISK_COLORS.get(str(value), "")


def risk_value(risk):
    value = getattr(risk, "risk_level", RiskLevel.NECUNOSCUT)
    return value.value if hasattr(value, "value") else str(value)


def build_rows(state):
    rows = []
    for item in state.get("results", []):
        clause = item.get("clause")
        risk = item.get("risk")
        recommendation = item.get("recommendation")
        rows.append({
            "ID clauza": clause.id,
            "Sectiune": clause.section,
            "Tip": clause.type.value if hasattr(clause.type, "value") else str(clause.type),
            "Pagina": clause.page,
            "Risc": risk_value(risk),
            "Probleme": "; ".join(getattr(risk, "issues", [])),
            "Reformulare": getattr(recommendation, "reformulated_text", ""),
        })
    return rows


def save_uploaded_pdf(uploaded_file):
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    safe_name = uploaded_file.name.replace(" ", "_")
    path = Path("data/uploads") / safe_name
    with open(path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return str(path)


def render_summary(state):
    parsed_doc = state.get("parsed_doc")
    risk_map = state.get("risk_map", {})
    if not parsed_doc:
        return

    high_count = sum(1 for risk in risk_map.values() if risk.risk_level == RiskLevel.RIDICAT)
    medium_count = sum(1 for risk in risk_map.values() if risk.risk_level == RiskLevel.MEDIU)
    unknown_count = sum(1 for risk in risk_map.values() if risk.risk_level == RiskLevel.NECUNOSCUT)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clauze", len(parsed_doc.clauses))
    col2.metric("Risc ridicat", high_count)
    col3.metric("Risc mediu", medium_count)
    col4.metric("Necunoscut", unknown_count)

    if state.get("high_risk_alert"):
        st.warning("Contractul contine un numar mare de clauze cu risc RIDICAT.")

    st.markdown(f"**Titlu document:** {parsed_doc.metadata.title}")
    st.markdown(f"**Pagini:** {parsed_doc.metadata.page_count}")
    st.markdown(f"**Valoare:** {parsed_doc.metadata.value or 'Nespecificat'}")
    st.markdown(f"**Durata:** {parsed_doc.metadata.duration or 'Nespecificat'}")


def render_results(state):
    rows = build_rows(state)
    if not rows:
        st.info("Nu exista rezultate de afisat.")
        return

    df = pd.DataFrame(rows)
    styled = df.style.map(style_risk, subset=["Risc"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Detalii pe clauze")
    for item in state.get("results", []):
        clause = item.get("clause")
        risk = item.get("risk")
        recommendation = item.get("recommendation")
        context = item.get("context", [])
        label = f"{clause.id} | {clause.section} | {risk_value(risk)}"
        with st.expander(label):
            st.markdown("**Text original**")
            st.write(clause.text)
            st.markdown("**Probleme identificate**")
            issues = getattr(risk, "issues", [])
            if issues:
                for issue in issues:
                    st.write(f"- {issue}")
            else:
                st.write("Nu au fost raportate probleme.")

            st.markdown("**Referinte**")
            references = getattr(risk, "references", [])
            if references:
                for reference in references:
                    st.write(f"- {reference}")
            else:
                st.write("Nu exista referinte disponibile.")

            reformulated = getattr(recommendation, "reformulated_text", "")
            if reformulated:
                st.markdown("**Reformulare propusa**")
                st.write(reformulated)

            explanation = getattr(recommendation, "explanation", "")
            if explanation:
                st.markdown("**Explicatie**")
                st.write(explanation)

            if context:
                st.markdown("**Context RAG folosit**")
                for chunk in context:
                    st.caption(f"{chunk.source} | scor: {chunk.score}")
                    st.write(chunk.text[:800])


def render_downloads(state):
    report_path = state.get("report_path")
    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as file:
            report_text = file.read()
        st.download_button(
            "Descarca raport Markdown",
            data=report_text,
            file_name=Path(report_path).name,
            mime="text/markdown",
        )

    rows = build_rows(state)
    if rows:
        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("Descarca tabel CSV", data=csv, file_name="analiza_clauze.csv", mime="text/csv")


def main():
    st.title("Analizator contracte juridice")
    st.write("Sistem multi-agent pentru analizarea contractelor, evaluarea riscurilor si generarea reformularilor.")

    with st.sidebar:
        st.header("Analiza")
        uploaded_file = st.file_uploader("Incarca PDF", type=["pdf"])
        retrieval_threshold = st.slider("Prag relevanta RAG", 0.0, 1.0, 0.5, 0.05)
        high_risk_threshold = st.slider("Prag alerta risc ridicat", 1, 10, 2, 1)
        run_button = st.button("Ruleaza analiza", type="primary", use_container_width=True)

        st.divider()
        st.caption("Daca OPENAI_API_KEY sau vectorstore lipsesc, aplicatia foloseste fallback local pentru demo.")

    if "analysis_state" not in st.session_state:
        st.session_state.analysis_state = None

    if run_button:
        if not uploaded_file:
            st.error("Incarca mai intai un fisier PDF.")
            return

        pdf_path = save_uploaded_pdf(uploaded_file)
        try:
            with st.spinner("Se analizeaza documentul..."):
                state = run_pipeline(
                    pdf_path,
                    retrieval_threshold=retrieval_threshold,
                    high_risk_threshold=high_risk_threshold
            )

            st.session_state.analysis_state = state
            st.success("Analiza finalizata.")
        except Exception as exc:
            st.error(f"Analiza a esuat: {exc}")
            return

    state = st.session_state.analysis_state
    if state:
        render_summary(state)
        st.divider()
        render_results(state)
        st.divider()
        render_downloads(state)

        with st.expander("Log workflow"):
            st.json(state.get("run_log", []))
    else:
        st.info("Incarca un PDF si apasa «Ruleaza analiza». Aplicatia pastreaza rezultatele in session_state.")


if __name__ == "__main__":
    main()
