"""
Test 02: Dynamic REST API RAG Golden Benchmark Evaluation
Ingests live DummyJSON recipes, performs dense semantic retrieval, and evaluates
multi-constraint generated recommendations against verified Golden standards using:
- FaithfulnessMetric
- AnswerRelevancyMetric
- HallucinationMetric
- GEval (Instruction Following)
"""

import os
import sys
from pathlib import Path
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
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
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from src.cohere_evalmodel import CohereModel

# 1. Golden Reference Dataset for Recipe Recommendations
goldens = [
    Golden(
        input=(
            "I want to have Quinoa Salad with Avocado. If its calories per "
            "serving are below 300, suggest a supplement or upgrade to increase "
            "the calorie count to 400 or above."
        ),
        expected_output=(
            "DummyJSON lists Quinoa Salad with Avocado at 280 calories per "
            "serving, which is below 300. Explain that at least 120 additional "
            "calories per serving are needed to reach 400. Suggest adding a "
            "measured ingredient such as extra avocado, feta, or dressing, and "
            "advise checking package nutrition labels for exact counts."
        ),
    ),
    Golden(
        input=(
            "I have Parmesan cheese, chicken, bell peppers, cucumber, avocado, "
            "and mangoes at home. Which recipes can I try with these ingredients?"
        ),
        expected_output=(
            "Recommend repository recipes that use several listed ingredients, "
            "including Chicken Alfredo Pasta, Mango Salsa Chicken, and Quinoa Salad with Avocado. "
            "State which ingredients match and disclose that each recipe needs other ingredients."
        ),
    ),
]


def test_api_rag_golden_benchmark():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in .env")

    co = cohere.ClientV2(api_key=api_key)
    judge = CohereModel()

    # 2. Ingest live recipes from REST API
    print("Fetching live recipes from DummyJSON API...")
    response = requests.get("https://dummyjson.com/recipes?limit=0", timeout=20)
    response.raise_for_status()
    recipes = response.json().get("recipes", [])
    print(f"Total Live Recipes Ingested: {len(recipes)}")

    # Format into text documents
    documents = []
    for r in recipes:
        doc = (
            f"Recipe ID: {r.get('id')}\n"
            f"Name: {r.get('name')}\n"
            f"Calories per serving: {r.get('caloriesPerServing')}\n"
            f"Cuisine: {r.get('cuisine')}\n"
            f"Ingredients: {', '.join(r.get('ingredients', []))}\n"
            f"Instructions: {' '.join(r.get('instructions', []))}"
        )
        documents.append(doc)

    print("Generating dense embeddings for recipe collection...")
    embed_resp = co.embed(
        texts=documents[:50],  # Index representative sample for fast CI execution
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    doc_vectors = np.array(embed_resp.embeddings.float)

    def retrieve_and_generate(query: str) -> tuple[str, list[str]]:
        query_resp = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"]
        )
        q_vec = np.array(query_resp.embeddings.float)
        sims = cosine_similarity(q_vec, doc_vectors)[0]
        top_indices = sims.argsort()[-4:][::-1]
        retrieved = [documents[i] for i in top_indices]

        context = "\n\n---\n\n".join(retrieved)
        prompt = f"""
Use ONLY the following recipe catalog context to answer the user inquiry.
Context:
{context}

Question:
{query}
"""
        ans_resp = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        answer = "".join(
            item.text for item in ans_resp.message.content
            if getattr(item, "text", None)
        ).strip()

        return answer, retrieved

    # 3. Build test cases from Goldens
    test_cases = []
    for golden in goldens:
        answer, retrieved = retrieve_and_generate(golden.input)
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=answer,
                expected_output=golden.expected_output,
                retrieval_context=retrieved,
                context=retrieved
            )
        )

    # 4. Evaluation Metrics
    instruction_metric = GEval(
        name="Instruction Following & Accuracy",
        criteria="The response should follow all requirements in the question and accurately match ingredients and calorie calculations.",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=judge,
    )

    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7, include_reason=True),
        AnswerRelevancyMetric(model=judge, threshold=0.7, include_reason=True),
        HallucinationMetric(model=judge, threshold=0.7),
        instruction_metric,
    ]

    print("\nEvaluating REST API RAG pipeline across Golden benchmarks...")
    evaluate(test_cases, metrics)


if __name__ == "__main__":
    test_api_rag_golden_benchmark()
