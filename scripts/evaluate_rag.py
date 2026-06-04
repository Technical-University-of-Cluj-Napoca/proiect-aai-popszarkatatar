import os
import sys
import json

from datasets import Dataset
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import LLMContextPrecisionWithReference, Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
load_dotenv()


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_evaluation():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory="vectorstore/", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    llm = ChatOpenAI(model="gpt-4o-mini")

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    questions = [
        "Care sunt condițiile de forță majoră conform Codului Civil?",
        "Ce penalități de întârziere se pot aplica într-un contract de achiziție publică conform Legii 98/2016?",
        "Cum trebuie obținut consimțământul pentru prelucrarea datelor conform GDPR?",
        "Ce reprezintă o clauză abuzivă în contractele pentru consumatori conform ghidurilor ANPC?",
        "Cum se definește semnătura electronică în legea model UNCITRAL?",
        "Când poate interveni rezilierea unilaterală a unui contract de prestări servicii?",
        "Care este termenul legal de notificare a unei breșe de securitate GDPR?",
        "Sunt valabile clauzele care exclud total răspunderea în Codul Civil român?",
        "Cum se tratează comunicările electronice internaționale conform UNCITRAL?",
        "Ce drepturi principale au consumatorii la achiziția unui imobil nou de la un dezvoltator?"
    ]

    ground_truths = {
        "Care sunt condițiile de forță majoră conform Codului Civil?": "Forta majora este un eveniment extern, imprevizibil, absolut invincibil si inevitabil (art. 1351 C. civ.).",
        "Ce penalități de întârziere se pot aplica într-un contract de achiziție publică conform Legii 98/2016?": "Penalitatile sunt stabilite proportional cu perioada de intarziere si se calculeaza ca procent pe zi de intarziere din valoarea obligatiei neexecutate.",
        "Cum trebuie obținut consimțământul pentru prelucrarea datelor conform GDPR?": "Consimtamantul trebuie sa fie liber exprimat, specific, informat si lipsit de ambiguitate, printr-o actiune neechivoca (Art. 4 GDPR).",
        "Ce reprezintă o clauză abuzivă în contractele pentru consumatori conform ghidurilor ANPC?": "O clauza nenegociata direct care creeaza un dezechilibru semnificativ intre drepturile si obligatiile partilor, in detrimentul consumatorului.",
        "Cum se definește semnătura electronică în legea model UNCITRAL?": "Date in format electronic, atasate sau asociate logic cu un mesaj de date, folosite pentru a identifica semnatarul si a indica aprobarea informatiilor.",
        "Când poate interveni rezilierea unilaterală a unui contract de prestări servicii?": "Poate interveni daca este prevazuta expres (pact comisoriu) sau prin notificare in cazul neexecutarii culpabile a obligatiilor esentiale.",
        "Care este termenul legal de notificare a unei breșe de securitate GDPR?": "Fara intarzieri nejustificate si, daca este posibil, in termen de cel mult 72 de ore de la data la care a luat la cunostinta de aceasta.",
        "Sunt valabile clauzele care exclud total răspunderea în Codul Civil român?": "Nu, clauzele care exclud sau limiteaza raspunderea pentru prejudiciile cauzate intentionat sau din culpa grava nu sunt valabile (art. 1355 C. civ.).",
        "Cum se tratează comunicările electronice internaționale conform UNCITRAL?": "Sunt considerate valabile si produc aceleasi efecte juridice ca si documentele pe suport de hartie, daca sunt accesibile pentru o referinta ulterioara.",
        "Ce drepturi principale au consumatorii la achiziția unui imobil nou de la un dezvoltator?": "Dreptul la informare completa, remedierea viciilor ascunse, respectarea calitatii si clauze contractuale fara penalitati abuzive la retragere."
    }

    samples = []

    for q in questions:
        docs = retriever.invoke(q)
        contexts = [doc.page_content for doc in docs]

        reference = ground_truths[q]

        context_str = "\n---\n".join(contexts)
        prompt = f"Raspunde la intrebare folosind doar contextul de mai jos:\nContext:\n{context_str}\n\nIntrebare: {q}"
        answer = llm.invoke(prompt).content

        sample = SingleTurnSample(
            user_input=q,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        AnswerRelevancy()
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print(result)

    os.makedirs("logs", exist_ok=True)
    with open("logs/rag_evaluation.json", "w", encoding="utf-8") as f:
        try:
            new_result = dict(result)
        except Exception:
            new_result = str(result)
        json.dump(new_result, f, indent=4)


if __name__ == "__main__":
    run_evaluation()

