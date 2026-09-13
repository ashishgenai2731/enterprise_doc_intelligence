from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(..., example="What were total Q3 operating expenses?")

class FinancialMetric(BaseModel):
    category: str = Field(..., description="Target line item category name")
    amount: str = Field(..., description="Extracted numerical figure with currency symbol")

class DocumentInsightResponse(BaseModel):
    query: str
    summary: str = Field(..., description="Synthesized executive context summary")
    metrics: List[FinancialMetric] = Field(..., description="Structured line-item metrics array")
    source_chunks: List[str] = Field(..., description="Source context doc IDs used")
    confidence_score: float = Field(..., description="Reranker output relevance metric")