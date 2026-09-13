import json
from pathlib import Path
from typing import List, Dict, Any


def format_qa_to_jsonl(raw_data: List[Dict[str, Any]],
                       output_file: str = "data/fin_qa_dataset.jsonl"):
    """Converts Context-Question triplets into structured JSON training prompts for Mistral SFTTrainer."""
    Path("data").mkdir(exist_ok=True)

    system_prompt = (
        "You are an enterprise financial specialist. Parse the retrieved context "
        "and return structured JSON matching the declared response schema precisely."
    )

    formatted_records = []
    with open(output_file, "w", encoding="utf-8") as f:
        for item in raw_data:
            user_content = f"Context:\n{item['context']}\n\nQuestion: {item['question']}"

            # Format target answer as structured JSON matching DocumentInsightResponse
            structured_answer = json.dumps({
                "query": item["question"],
                "summary": item["answer"],
                "metrics": item.get("metrics", []),
                "confidence_score": item.get("confidence_score", 0.95)
            })

            prompt = (
                f"<s>[INST] {system_prompt}\n\n{user_content} [/INST] "
                f"{structured_answer}</s>"
            )
            record = {"text": prompt}
            f.write(json.dumps(record) + "\n")
            formatted_records.append(record)

    print(f"Successfully serialized {len(formatted_records)} records to {output_file}")


if __name__ == "__main__":
    sample_data = [{
        "context": "Operating income for Q3 2024 reached $14.2M.",
        "question": "What was the operating income for Q3 2024?",
        "answer": "Operating income reached $14.2M for Q3 2024.",
        "metrics": [{"category": "Operating Income", "amount": "$14.2M"}],
        "confidence_score": 0.98
    }]
    format_qa_to_jsonl(sample_data)