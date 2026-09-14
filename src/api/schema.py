from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    # Updated to Pydantic V2 syntax for examples
    query: str = Field(..., examples=["What were total Q3 operating expenses?"])


class FinancialMetric(BaseModel):
    category: str = Field(..., description="Target line item category name")
    amount: str = Field(..., description="Extracted numerical figure with currency symbol")


class DocumentInsightResponse(BaseModel):
    query: str
    summary: str = Field(..., description="Synthesized executive context summary")
    metrics: List[FinancialMetric] = Field(..., description="Structured line-item metrics array")

    # Set default factory to list so the LLM doesn't have to hallucinate these before main.py injects them
    source_chunks: List[str] = Field(
        default_factory=list,
        description="Source context doc IDs used (injected by backend)"
    )

    # Optional default so the LLM doesn't crash if it forgets to output a confidence score
    confidence_score: float = Field(
        default=0.95,
        description="Model self-evaluated confidence score"
    )