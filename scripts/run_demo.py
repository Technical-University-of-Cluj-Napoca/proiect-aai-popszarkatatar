import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.workflow import run_pipeline

if __name__ == "__main__":
    pdf = "data/contract_demo.pdf"
    if not os.path.exists("vectorstore/local_index.json"):
        print("Indexul nu exista. Rulez build_index.py --force ...")
        os.system(f"{sys.executable} scripts/build_index.py --force")
    state = run_pipeline(pdf)
    print("Raport generat:", state.get("report_path"))
    print("High risk alert:", state.get("high_risk_alert"))
