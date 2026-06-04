import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import time
import random
import streamlit as st
import pandas as pd
from src.agents.parser_agent import DocumentParserAgent

st.set_page_config(page_title="Legal Contract Analyzer", layout="wide")


def style_risk_level(val):
    colors = {
        'RIDICAT': '#ffe3e3',
        'MEDIU': '#fff3bf',
        'SCAZUT': '#fff9c4',
        'CONFORM': '#d3f9d8',
        'NECUNOSCUT': '#f8f9fa'
    }
    color = colors.get(val, '')
    if color:
        return f'background-color: {color}; color: #000000;'
    return ''


def mock_langgraph_pipeline(parsed_doc):
    risk_levels = ['RIDICAT', 'MEDIU', 'SCAZUT', 'CONFORM']
    results = []

    for clause in parsed_doc.clauses:
        simulated_risk = random.choice(risk_levels)
        results.append({
            'Sectiune': clause.section,
            'ID Clauza': clause.id,
            'Tip Clauza': clause.type.value if hasattr(clause.type, 'value') else clause.type,
            'Nivel Risc': simulated_risk,
            'Continut Text': clause.text
        })
    return results


def main():
    st.title("Analizator Contracte Juridice")
    st.markdown("Sistem multi-agent pentru identificarea clauzelor riscante.")

    with st.sidebar:
        st.header("Meniu")
        uploaded_file = st.file_uploader("Incarca un contract PDF", type=["pdf"])

        st.divider()
        st.markdown("### Setari Analiza")

        st.divider()
        run_analysis = st.button("Ruleaza analiza completa", type="primary")

    if run_analysis and uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_pdf_path = tmp_file.name

        progress_text = "Se initializeaza pipeline-ul..."
        my_bar = st.progress(0, text=progress_text)

        time.sleep(1)
        my_bar.progress(25, text="Agent Parsare: Extragere structura document...")
        parser = DocumentParserAgent()
        parsed_doc = parser.parse(temp_pdf_path)

        time.sleep(1)
        my_bar.progress(50, text="Agent RAG: Recuperare context legislativ (Mocked)...")

        time.sleep(1)
        my_bar.progress(75, text="Agent Risc: Evaluare clauze (Mocked)...")
        analysis_results = mock_langgraph_pipeline(parsed_doc)

        time.sleep(1)
        my_bar.progress(100, text="Analiza finalizata!")
        time.sleep(0.5)
        my_bar.empty()

        os.unlink(temp_pdf_path)

        meta = parsed_doc.metadata
        st.header(f"{meta.title if meta.title else 'Contract'}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valoare", meta.value if meta.value else "Nespecificat")
            st.write(f"Data semnarii: {meta.signing_date if meta.signing_date else 'N/A'}")
        with col2:
            st.metric("Durata", meta.duration if meta.duration else "Nespecificat")
            st.write(f"Data intrarii in vigoare: {meta.effective_date if meta.effective_date else 'N/A'}")
        with col3:
            st.metric("Nr. Pagini", meta.page_count)
            st.write(f"Parti: {len(meta.parties) if meta.parties else 0}")

        st.divider()
        st.subheader("Rezultate Evaluare Risc")

        if analysis_results:
            df = pd.DataFrame(analysis_results)

            high_risk_count = len(df[df['Nivel Risc'] == 'RIDICAT'])
            if high_risk_count > 0:
                st.warning(f"Alerte Risc: Au fost identificate {high_risk_count} clauze cu risc RIDICAT.")

            styled_df = df.style.map(style_risk_level, subset=['Nivel Risc'])

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=500
            )
        else:
            st.warning("Nu au fost extrase clauze.")

    elif run_analysis and uploaded_file is None:
        st.error("Te rog sa incarci un document PDF inainte de a rula analiza.")


if __name__ == "__main__":
    main()