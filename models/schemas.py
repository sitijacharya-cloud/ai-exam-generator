"""Pydantic models for data validation"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Metadata for a document chunk"""
    source_file: str
    page: Optional[int] = None
    chunk_id: int
    file_type: str = "pdf"
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class RetrievedContext(BaseModel):
    """A retrieved context from vector store"""
    content: str
    metadata: Dict
    relevance_score: float

class QuestionChoice(BaseModel):
    """Multiple choice option"""
    label: str  # A, B, C, D
    text: str

class Question(BaseModel):
    """Generated question"""
    id: str
    topic: str
    type: str  # MCQ, True/False, Short Answer, Long Answer
    difficulty: str  # Easy, Moderate, Hard
    prompt: str
    choices: Optional[List[QuestionChoice]] = None
    answer: str
    explanation: Optional[str] = None
    marks: int
    retrieved_contexts: List[RetrievedContext]

class QuestionBlueprint(BaseModel):
    """Blueprint for question generation"""
    question_type: str
    count: int
    difficulty: str
    marks_per_question: int

class ExamConfig(BaseModel):
    """Configuration for exam generation"""
    title: str
    instructions: str
    syllabus_topics: List[str]
    blueprints: List[QuestionBlueprint]
