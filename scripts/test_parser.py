import os
import sys

import pdfplumber
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.parser_agent import DocumentParserAgent

load_dotenv()

def main():
    pdf_path = "data/contract_exemplu2.pdf"

    if not os.path.exists(pdf_path):
        print(f"Eroare: Nu am gasit {pdf_path}. Pune un contract de test pentru a verifica.")
        return
    agent = DocumentParserAgent()
    parsed_doc = agent.parse(pdf_path)


    print("REZULTATE PARSARE:")
    print(f"Metadate extrase cu succes pentru: {parsed_doc.metadata.title}")
    print(f"Numar sectiuni identificate: {len(parsed_doc.sections)}")
    print(f"Numar clauze extrase: {len(parsed_doc.clauses)}")

    for section in parsed_doc.sections:
        print(section.title)

    print("\nPRIMELE 3 CLAUZE")
    for i, clause in enumerate(parsed_doc.clauses[:3]):
        print(f"\n[{i + 1}] ID: {clause.id}")
        print(f"Tip clauza: {clause.type.value}")
        print(f"Sectiune: {clause.section} (Pagina {clause.page})")
        print(f"Text: {clause.text}")


if __name__ == "__main__":
    main()