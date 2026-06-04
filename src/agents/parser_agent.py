import os
import re
import unicodedata
import pdfplumber
from langchain_openai import ChatOpenAI
from src.dtos import ClauseDTO, ClauseType, DocumentMetadataDTO, ParsedDocumentDTO, SectionDTO


class DocumentParserAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.metadata_parser = self.llm.with_structured_output(DocumentMetadataDTO)
        self.clause_pattern = re.compile(r"^\d+\.\d+")

    def _normalize(self, text: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")

    def _extract_metadata(self, text: str) -> DocumentMetadataDTO:
        prompt = f"Extrage metadatele contractului.\nText:\n{text}"
        return self.metadata_parser.invoke(prompt)

    def _is_section(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return False

        words = line.split()
        if len(words) > 12:
            return False

        if line.upper().startswith("PARTEA") or re.match(r"^1\.\s+In temeiul", line, re.IGNORECASE):
            return False

        if self.clause_pattern.match(line):
            return False

        if re.match(r"^(Articolul|Art\.?|Clauza|Capitolul|Cap\.?|Sectiunea?)\s*\d+", line, re.IGNORECASE):
            if line.startswith("art.") and len(words) > 5:
                return False
            return True

        match_numeric = re.match(r"^(\d+)\.?\s*([A-ZĂÎÂȘȚ])", line)
        if match_numeric:
            num_str = match_numeric.group(1)
            if num_str.isdigit() and int(num_str) < 100:
                return True

        if re.match(r"^(X{0,3})(IX|IV|V?I{0,3})\.\s+[A-ZĂÎÂȘȚ]", line):
            return True

        return False

    def _classify_with_llm(self, text: str) -> ClauseType:
        prompt = f"Clasifica urmatoarea clauza intr-una dintre valorile: penalitate, obligatie, drept, forta_majora, confidentialitate, reziliere, date_personale, altele. Raspunde doar cu valoarea.\nClauza: {text[:3000]}"
        try:
            result = self.llm.invoke(prompt).content.strip().lower()
            mapping = {
                "penalitate": ClauseType.PENALITATE,
                "obligatie": ClauseType.OBLIGATIE,
                "drept": ClauseType.DREPT,
                "forta_majora": ClauseType.FORTA_MAJORA,
                "confidentialitate": ClauseType.CONFIDENTIALITATE,
                "reziliere": ClauseType.REZILIERE,
                "date_personale": ClauseType.DATE_PERSONALE,
            }
            return mapping.get(result, ClauseType.ALTELE)
        except Exception:
            return ClauseType.ALTELE

    def _classify_clause(self, text: str) -> ClauseType:
        text_norm = self._normalize(text)
        if "forta majora" in text_norm: return ClauseType.FORTA_MAJORA
        if "penalitat" in text_norm or "intarziere" in text_norm: return ClauseType.PENALITATE
        if "date cu caracter personal" in text_norm or "gdpr" in text_norm or "prelucrarea datelor" in text_norm: return ClauseType.DATE_PERSONALE
        if "reziliere" in text_norm or "incetare" in text_norm: return ClauseType.REZILIERE
        if "confidential" in text_norm: return ClauseType.CONFIDENTIALITATE
        if any(t in text_norm for t in
               ["obligatia", "obligatiile", "se obliga", "are obligatia"]): return ClauseType.OBLIGATIE
        if any(t in text_norm for t in ["dreptul", "drepturile", "are dreptul"]): return ClauseType.DREPT

        return self._classify_with_llm(text)

    def _make_clause_id(self, section: str, index: int) -> str:
        section = self._normalize(section)
        safe_section = re.sub(r"[^a-z0-9]+", "_", section.lower()).strip("_")
        return f"{safe_section[:25]}_clz_{index}"

    def parse(self, pdf_path: str) -> ParsedDocumentDTO:
        contract_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"data/{contract_name}_parsed.json"

        try:
            sections = []
            clauses = []
            first_pages_text = ""
            current_section = None
            clause_buffer = []
            clause_page = 1
            clause_index = 1

            def _flush_buffer():
                nonlocal clause_buffer, clause_index
                text = " ".join(clause_buffer).strip()
                if len(text) > 30 and current_section:
                    clauses.append(ClauseDTO(
                        id=self._make_clause_id(current_section, clause_index),
                        section=current_section,
                        text=text,
                        page=clause_page,
                        type=self._classify_clause(text)
                    ))
                    clause_index += 1
                clause_buffer = []

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if page_number <= 2:
                        first_pages_text += text + "\n"

                    lines = [line.strip() for line in text.splitlines() if line.strip()]

                    for line in lines:
                        if self._is_section(line):
                            _flush_buffer()
                            current_section = line
                            sections.append(SectionDTO(title=line, start_page=page_number))
                            clause_index = 1
                            continue

                        if current_section is None:
                            continue

                        if self.clause_pattern.match(line):
                            _flush_buffer()
                            clause_buffer = [line]
                            clause_page = page_number
                        else:
                            if not clause_buffer:
                                clause_buffer = [line]
                                clause_page = page_number
                            else:
                                clause_buffer.append(line)

                _flush_buffer()

            metadata = self._extract_metadata(first_pages_text)
            metadata.page_count = total_pages

            parsed_document = ParsedDocumentDTO(metadata=metadata, sections=sections, clauses=clauses)
            os.makedirs("data", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(parsed_document.model_dump_json(indent=4))

            return parsed_document

        except Exception:
            return ParsedDocumentDTO(
                metadata=DocumentMetadataDTO(title="Unknown", page_count=0, parties=[], value="", duration=""),
                sections=[], clauses=[])