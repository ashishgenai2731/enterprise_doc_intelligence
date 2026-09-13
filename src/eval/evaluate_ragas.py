import os
import sys
from pathlib import Path

# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from config.settings import settings

# below import for Updating src/eval/evaluate_ragas.py to use
# local Ollama (ChatOllama) and local
# HuggingFace embeddings (BAAI/bge-large-en-v1.5)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Export API key to environment variables for RAGAS / LangChain judge models
# error occurs because RAGAS relies on LangChain's ChatOpenAI under the hood, which strictly checks system environment variables (os.environ["OPENAI_API_KEY"]). While Pydantic reads .env into settings.OPENAI_API_KEY, it doesn't automatically export it into os.environ.
if settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

def run_evaluation(test_samples: list[dict]):
    """Calculates faithfulness, answer relevance, and context recall over test sets."""
    data_dict = {
        "question": [s["question"] for s in test_samples],
        "answer": [s["generated_answer"] for s in test_samples],
        "contexts": [s["retrieved_contexts"] for s in test_samples],
        "ground_truth": [s["ground_truth"] for s in test_samples]
    }

    dataset = Dataset.from_dict(data_dict)

    # Below two steps if you don't have credit in OPENAI
    # Initialize local Ollama evaluator LLM & local embedding model
    eval_llm = LangchainLLMWrapper(ChatOllama(model="mistral", temperature=0.0))
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
    )

    # Bind local models to RAGAS metrics
    faithfulness.llm = eval_llm
    answer_relevancy.llm = eval_llm
    answer_relevancy.embeddings = eval_embeddings
    context_recall.llm = eval_llm
    # End of two steps

    scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall
        ],
        llm=eval_llm,
        embeddings=eval_embeddings
    )

    print("=== RAGAS Metric Evaluation Results ===")
    print(scores)
    return scores


if __name__ == "__main__":
    # Test sample with full context to aid local LLM question generation
    mock_samples = [{
        "question": "What were the total operating expenses for 2015?",
        "generated_answer": "Total operating expenses for 2015 were $1.8 billion.",
        "retrieved_contexts": [
            "In 2015, total operating expenses reached $1.8 billion, compared to $1.2 billion in 2014."],
        "ground_truth": "$1.8 billion"
    }]
    run_evaluation(mock_samples)
