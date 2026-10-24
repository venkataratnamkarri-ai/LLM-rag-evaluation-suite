"""
Test 01: Complete 5-Metric Contextual RAG Evaluation on PDF Pipeline
Evaluates an end-to-end PDF RAG pipeline across the entire RAG Triad and context retrieval metrics:
1. Contextual Precision Metric
2. Contextual Recall Metric
3. Contextual Relevancy Metric
4. Faithfulness Metric
5. Answer Relevancy Metric

Uses custom CohereModel as the LLM-as-a-Judge.
"""

import os
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import cohere
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from src.cohere_evalmodel import CohereModel

PDF_PATH = BASE_DIR / "data" / "Agile_Process_Guide_5_Pages.pdf"

if not PDF_PATH.is_file():
    raise FileNotFoundError(f"Agile Guide PDF not found at: {PDF_PATH}")

# 1. Curated Golden Reference Dataset
goldens = [
    Golden(
        input="Tell the exact role and responsibilities of Scrum Master and what kind of value Scrum Master brings to the team.",
        expected_output="""
The Scrum Master is responsible for helping establish the Scrum framework, improving team effectiveness,
and addressing impediments. They play a crucial role in facilitating the Scrum process and ensuring that
the team adheres to Scrum principles.
"""
    ),
    Golden(
        input="What are Story Points?",
        expected_output="""
Story points are relative units used to estimate the effort required to implement a user story.
"""
    ),
    Golden(
        input="What is Planning Poker?",
        expected_output="""
Planning Poker is a consensus-based estimation technique using cards representing story point values.
"""
    )
]

# 2. Extract Text & Chunk PDF
def build_vector_index(pdf_path: Path, chunk_size: int = 700, overlap: int = 100):
    pdf_text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"

    chunks = []
    start = 0
    while start < len(pdf_text):
        end = start + chunk_size
        chunks.append(pdf_text[start:end])
        start = end - overlap

    return chunks

# 3. Main Test Function
def test_pdf_rag_evaluation():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in .env")

    co = cohere.ClientV2(api_key=api_key)
    judge = CohereModel()

    print(f"Loading and chunking PDF: {PDF_PATH}")
    chunks = build_vector_index(PDF_PATH)
    print(f"Created {len(chunks)} overlapping chunks.")

    print("Embedding document chunks using 'embed-english-v3.0'...")
    embedding_response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    chunk_embeddings = np.array(embedding_response.embeddings.float)

    def ask_rag(question: str) -> tuple[str, list[str]]:
        query_embedding = co.embed(
            texts=[question],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"]
        )
        query_vector = np.array(query_embedding.embeddings.float)

        scores = cosine_similarity(query_vector, chunk_embeddings)[0]
        top_k = 3
        top_indices = scores.argsort()[-top_k:][::-1]
        retrieved_chunks = [chunks[i] for i in top_indices]

        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
Use ONLY the following context to answer the question.
If the answer is not found in the context, reply: 'I couldn't find this information in the document.'

Context:
{context}

Question:
{question}
"""
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )

        answer = "".join(
            item.text for item in response.message.content
            if getattr(item, "text", None)
        ).strip()

        return answer, retrieved_chunks

    # Build evaluation test cases from Goldens
    dataset = EvaluationDataset(goldens=goldens)
    test_cases = []

    print("\nExecuting RAG queries across Golden dataset...")
    for golden in dataset.goldens:
        answer, retrieved = ask_rag(golden.input)
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=answer,
                expected_output=golden.expected_output,
                retrieval_context=retrieved
            )
        )

    # 5 Key RAG Metrics
    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7, include_reason=True),
        AnswerRelevancyMetric(model=judge, threshold=0.7, include_reason=True),
        ContextualPrecisionMetric(model=judge, threshold=0.7, include_reason=True),
        ContextualRecallMetric(model=judge, threshold=0.7, include_reason=True),
        ContextualRelevancyMetric(model=judge, threshold=0.7, include_reason=True),
    ]

    print("\nRunning DeepEval Evaluation Suite with Cohere LLM Judge...")
    evaluate(test_cases, metrics)


if __name__ == "__main__":
    test_pdf_rag_evaluation()
