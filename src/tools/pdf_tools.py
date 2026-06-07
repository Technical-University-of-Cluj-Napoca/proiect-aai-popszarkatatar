import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document


def load_corpus(corpus_dir: str) -> list[Document]:
    all_files=[]
    type_mapping = {
        "gdpr": "regulation",
        "legi": "law",
        "contracte": "contract",
        "uncitral": "model_law",
        "anpc": "guideline"
    }

    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                parent_folder = os.path.basename(root)
                doc_type = type_mapping.get(parent_folder, "unknown")

                loader = PDFPlumberLoader(file_path)
                pages = loader.load()

                for page in pages:
                    page.metadata["type"] = doc_type
                    page.metadata["source"] = file
                    all_files.append(page)
    return all_files