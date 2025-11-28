

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple, Optional
import openai
from config.settings import OPENAI_API_KEY, LLM_MODEL
from models.schemas import Question, RetrievedContext
import json
import numpy as np
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL
    
    
    # 1. RETRIEVAL QUALITY METRICS
  
    
    def evaluate_context_relevance(self, query: str, contexts: List[RetrievedContext]) -> Dict:
        """
        Metric: Context Relevance
        Question: Are retrieved contexts actually about the query topic?
        
        Key for: Detecting when system retrieves irrelevant content
        (e.g., ML content for dentistry questions)
        """
        if not contexts:
            return {
                "metric": "context_relevance",
                "score": 0.0,
                "details": "No contexts retrieved",
                "status": "skipped"
            }
        
        # Calculate average relevance score
        relevance_scores = [ctx.relevance_score for ctx in contexts]
        avg_relevance = np.mean(relevance_scores)
        min_relevance = np.min(relevance_scores)
        
        # Determine status
        if avg_relevance >= 0.7:
            status = "excellent"
        elif avg_relevance >= 0.5:
            status = "good"
        elif avg_relevance >= 0.3:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "metric": "context_relevance",
            "score": float(avg_relevance),
            "min_score": float(min_relevance),
            "max_score": float(np.max(relevance_scores)),
            "status": status,
            "details": f"Average relevance: {avg_relevance:.2%}, Min: {min_relevance:.2%}"
        }
    
    def evaluate_context_precision(self, question: Question) -> Dict:
        """
        Metric: Context Precision
        Question: What proportion of retrieved contexts are actually relevant?
        
        Uses LLM-as-judge to verify if context matches topic
        """
        if not question.retrieved_contexts:
            return {
                "metric": "context_precision",
                "score": 1.0,  # Perfect score if no irrelevant context used
                "status": "no_context_used",
                "details": "Generated from general knowledge"
            }
        
        # Check if context is about the question topic
        context_sample = question.retrieved_contexts[0].content[:800]
        
        prompt = f"""You are evaluating a RAG system for exam generation.

Topic: {question.topic}

Retrieved Context:
{context_sample}

Is this context RELEVANT to the topic "{question.topic}"?

Answer with JSON:
{{
    "is_relevant": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            score = float(result.get("confidence", 0.5)) if result.get("is_relevant", False) else 0.0
            
            return {
                "metric": "context_precision",
                "score": score,
                "is_relevant": result.get("is_relevant", False),
                "reason": result.get("reason", ""),
                "status": "good" if score >= 0.7 else "fair" if score >= 0.4 else "poor"
            }
        except Exception as e:
            logger.error(f"Context precision evaluation failed: {e}")
            return {
                "metric": "context_precision",
                "score": 0.5,
                "status": "error",
                "details": str(e)
            }
    
    def evaluate_context_recall(self, question: Question, k: int = 3) -> Dict:
        """
        Metric: Context Recall
        Question: Did we retrieve enough relevant contexts?
        
        Checks if number of retrieved contexts meets expectations
        """
        expected = k
        actual = len(question.retrieved_contexts) if question.retrieved_contexts else 0
        
        recall = actual / expected if expected > 0 else 0.0
        
        return {
            "metric": "context_recall",
            "score": recall,
            "expected": expected,
            "retrieved": actual,
            "status": "good" if recall >= 0.8 else "fair" if recall >= 0.5 else "poor"
        }
    
    def evaluate_source_diversity(self, questions: List[Question]) -> Dict:
        """
        Metric: Source Diversity
        Question: Are questions using diverse sources or just one book?
        
        Important for comprehensive coverage
        """
        all_sources = []
        for q in questions:
            if q.retrieved_contexts:
                sources = [ctx.metadata['source_file'] for ctx in q.retrieved_contexts]
                all_sources.extend(sources)
        
        if not all_sources:
            return {
                "metric": "source_diversity",
                "score": 0.0,
                "unique_sources": 0,
                "status": "no_sources"
            }
        
        source_counts = Counter(all_sources)
        unique_sources = len(source_counts)
        total_retrievals = len(all_sources)
        
        # Diversity score (Shannon entropy normalized)
        entropy = -sum((count/total_retrievals) * np.log2(count/total_retrievals) 
                      for count in source_counts.values())
        max_entropy = np.log2(unique_sources) if unique_sources > 1 else 1
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            "metric": "source_diversity",
            "score": float(diversity_score),
            "unique_sources": unique_sources,
            "total_retrievals": total_retrievals,
            "source_distribution": dict(source_counts),
            "status": "excellent" if diversity_score >= 0.7 else "good" if diversity_score >= 0.4 else "poor"
        }
    
    # ========================================================================
    # 2. GENERATION QUALITY METRICS
    # ========================================================================
    
    def evaluate_faithfulness(self, question: Question) -> Dict:
        """
        Metric: Faithfulness (Most Important for RAG!)
        Question: Is the question/answer grounded in the retrieved context?
        
        Detects hallucination - critical for exam quality
        """
        if not question.retrieved_contexts:
            return {
                "metric": "faithfulness",
                "score": 1.0,
                "status": "general_knowledge",
                "details": "No context used - generated from general knowledge"
            }
        
        context_text = "\n\n".join([ctx.content[:500] for ctx in question.retrieved_contexts[:2]])
        
        prompt = f"""You are evaluating faithfulness in a RAG system for exam generation.

Retrieved Context:
{context_text}

Generated Question: {question.prompt}
Answer: {question.answer}
{f'Explanation: {question.explanation}' if question.explanation else ''}

Is this question FAITHFULLY derived from the context?
- Can the question be answered using ONLY the provided context?
- Is the answer supported by the context?
- Are there any hallucinated facts not in the context?

Respond with JSON:
{{
    "faithfulness_score": 0.0-1.0,
    "is_faithful": true/false,
    "issues": ["list any hallucinations or unsupported claims"],
    "explanation": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            score = float(result.get("faithfulness_score", 0.5))
            
            return {
                "metric": "faithfulness",
                "score": score,
                "is_faithful": result.get("is_faithful", False),
                "issues": result.get("issues", []),
                "explanation": result.get("explanation", ""),
                "status": "excellent" if score >= 0.8 else "good" if score >= 0.6 else "poor"
            }
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return {
                "metric": "faithfulness",
                "score": 0.5,
                "status": "error",
                "details": str(e)
            }
    
    def evaluate_answer_relevance(self, question: Question) -> Dict:
        """
        Metric: Answer Relevance
        Question: Does the question actually test the intended topic?
        
        Ensures questions are on-topic and educationally appropriate
        """
        prompt = f"""You are evaluating an exam question.

Topic: {question.topic}
Difficulty: {question.difficulty}
Question Type: {question.type}

Question: {question.prompt}

Rate how well this question tests knowledge of "{question.topic}":
- Is it directly about the topic?
- Is it at the appropriate difficulty level?
- Is it a valid exam question?

Respond with JSON:
{{
    "relevance_score": 0.0-1.0,
    "is_on_topic": true/false,
    "appropriateness": "appropriate/too_easy/too_hard/off_topic",
    "reason": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            score = float(result.get("relevance_score", 0.5))
            
            return {
                "metric": "answer_relevance",
                "score": score,
                "is_on_topic": result.get("is_on_topic", False),
                "appropriateness": result.get("appropriateness", "unknown"),
                "reason": result.get("reason", ""),
                "status": "excellent" if score >= 0.8 else "good" if score >= 0.6 else "poor"
            }
        except Exception as e:
            logger.error(f"Answer relevance evaluation failed: {e}")
            return {
                "metric": "answer_relevance",
                "score": 0.5,
                "status": "error",
                "details": str(e)
            }
    
    def evaluate_answer_correctness(self, question: Question) -> Dict:
        """
        Metric: Answer Correctness
        Question: Is the provided answer actually correct?
        
        Critical for exam validity
        """
        if question.type == "Multiple Choice":
            choices_text = "\n".join([f"{c.label}. {c.text}" for c in question.choices])
            question_text = f"""Question: {question.prompt}

Choices:
{choices_text}

Stated Correct Answer: {question.answer}"""
        else:
            question_text = f"""Question: {question.prompt}
Answer: {question.answer}"""

        prompt = f"""{question_text}

Is this answer correct? Verify the accuracy.

Respond with JSON:
{{
    "correctness_score": 0.0-1.0,
    "is_correct": true/false,
    "issues": ["any errors found"],
    "explanation": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            score = float(result.get("correctness_score", 0.5))
            
            return {
                "metric": "answer_correctness",
                "score": score,
                "is_correct": result.get("is_correct", False),
                "issues": result.get("issues", []),
                "explanation": result.get("explanation", ""),
                "status": "excellent" if score >= 0.9 else "good" if score >= 0.7 else "poor"
            }
        except Exception as e:
            logger.error(f"Answer correctness evaluation failed: {e}")
            return {
                "metric": "answer_correctness",
                "score": 0.5,
                "status": "error",
                "details": str(e)
            }
    
    # ========================================================================
    # 3. RAGAS FRAMEWORK (Industry Standard)
    # ========================================================================
    
    def calculate_ragas_metrics(self, question: Question) -> Dict:
        """
        RAGAS (RAG Assessment Score) Framework
        Industry-standard evaluation for RAG systems
        
        Components:
        1. Faithfulness (40%) - No hallucination
        2. Answer Relevance (30%) - On-topic
        3. Context Precision (20%) - Relevant contexts
        4. Context Recall (10%) - Sufficient contexts
        """
        logger.info(f"Calculating RAGAS metrics for: {question.topic}")
        
        # Evaluate each component
        faithfulness = self.evaluate_faithfulness(question)
        relevance = self.evaluate_answer_relevance(question)
        precision = self.evaluate_context_precision(question)
        recall = self.evaluate_context_recall(question)
        
        # Calculate weighted RAGAS score
        ragas_score = (
            faithfulness['score'] * 0.40 +
            relevance['score'] * 0.30 +
            precision['score'] * 0.20 +
            recall['score'] * 0.10
        )
        
        # Interpretation
        if ragas_score >= 0.8:
            interpretation = "Excellent - High quality RAG output"
            grade = "A"
        elif ragas_score >= 0.7:
            interpretation = "Good - Acceptable quality"
            grade = "B"
        elif ragas_score >= 0.6:
            interpretation = "Fair - Minor issues"
            grade = "C"
        elif ragas_score >= 0.5:
            interpretation = "Poor - Needs improvement"
            grade = "D"
        else:
            interpretation = "Critical - Major issues detected"
            grade = "F"
        
        return {
            "ragas_score": float(ragas_score),
            "grade": grade,
            "interpretation": interpretation,
            "components": {
                "faithfulness": faithfulness,
                "answer_relevance": relevance,
                "context_precision": precision,
                "context_recall": recall
            }
        }
    
    
    
    def evaluate_question_clarity(self, question: Question) -> Dict:
        """
        Metric: Question Clarity
        Question: Is the question clear and unambiguous?
        """
        prompt = f"""Evaluate the clarity of this exam question.

Question: {question.prompt}
{f'Choices: {[c.text for c in question.choices]}' if question.choices else ''}

Rate clarity on these dimensions:
- Is the question clearly worded?
- Is it unambiguous?
- Is it free from grammatical errors?
- Can a student understand what's being asked?

Respond with JSON:
{{
    "clarity_score": 0.0-1.0,
    "is_clear": true/false,
    "issues": ["any clarity problems"],
    "suggestions": ["improvements if needed"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            score = float(result.get("clarity_score", 0.5))
            
            return {
                "metric": "question_clarity",
                "score": score,
                "is_clear": result.get("is_clear", False),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "status": "excellent" if score >= 0.9 else "good" if score >= 0.7 else "poor"
            }
        except Exception as e:
            logger.error(f"Clarity evaluation failed: {e}")
            return {
                "metric": "question_clarity",
                "score": 0.5,
                "status": "error"
            }
    
    # ========================================================================
    # 5. BATCH EVALUATION
    # ========================================================================
    
    def evaluate_exam_batch(self, questions: List[Question], sample_size: Optional[int] = None) -> Dict:
        """
        Evaluate an entire exam or batch of questions
        
        Args:
            questions: List of questions to evaluate
            sample_size: If provided, randomly sample this many questions
        
        Returns comprehensive evaluation report
        """
        import random
        
        if sample_size and sample_size < len(questions):
            questions = random.sample(questions, sample_size)
        
        logger.info(f"Evaluating batch of {len(questions)} questions...")
        
        # Aggregate metrics
        all_metrics = {
            "total_questions": len(questions),
            "ragas_scores": [],
            "faithfulness_scores": [],
            "relevance_scores": [],
            "correctness_scores": [],
            "questions_with_context": 0,
            "questions_without_context": 0
        }
        
        detailed_results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {question.topic}")
            
            # Calculate RAGAS metrics
            ragas = self.calculate_ragas_metrics(question)
            
            all_metrics["ragas_scores"].append(ragas["ragas_score"])
            all_metrics["faithfulness_scores"].append(ragas["components"]["faithfulness"]["score"])
            all_metrics["relevance_scores"].append(ragas["components"]["answer_relevance"]["score"])
            
            # Count context usage
            if question.retrieved_contexts:
                all_metrics["questions_with_context"] += 1
            else:
                all_metrics["questions_without_context"] += 1
            
            detailed_results.append({
                "question_id": question.id,
                "topic": question.topic,
                "type": question.type,
                "ragas": ragas
            })
        
        # Calculate aggregate statistics
        summary = {
            "overall_ragas_score": float(np.mean(all_metrics["ragas_scores"])),
            "avg_faithfulness": float(np.mean(all_metrics["faithfulness_scores"])),
            "avg_relevance": float(np.mean(all_metrics["relevance_scores"])),
            "context_usage_rate": all_metrics["questions_with_context"] / len(questions),
            "score_distribution": {
                "excellent (≥0.8)": sum(1 for s in all_metrics["ragas_scores"] if s >= 0.8),
                "good (0.7-0.8)": sum(1 for s in all_metrics["ragas_scores"] if 0.7 <= s < 0.8),
                "fair (0.6-0.7)": sum(1 for s in all_metrics["ragas_scores"] if 0.6 <= s < 0.7),
                "poor (<0.6)": sum(1 for s in all_metrics["ragas_scores"] if s < 0.6)
            }
        }
        
        # Source diversity
        source_diversity = self.evaluate_source_diversity(questions)
        
        return {
            "summary": summary,
            "source_diversity": source_diversity,
            "detailed_results": detailed_results,
            "recommendations": self._generate_recommendations(summary, source_diversity)
        }
    
  
    
    # ========================================================================
    # 6. EXPORT AND REPORTING
    # ========================================================================
    
    def generate_evaluation_report(self, evaluation_results: Dict, output_path: str):
        """Generate a comprehensive evaluation report"""
        import json
        from datetime import datetime
        
        report = {
            "evaluation_date": datetime.now().isoformat(),
            "evaluator_version": "1.0",
            "results": evaluation_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {output_path}")


# ========================================================================
# USAGE EXAMPLE
# ========================================================================

if __name__ == "__main__":
    
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Example: Evaluate single question
    # question = Question(...)  # Your question object
    # ragas_score = evaluator.calculate_ragas_metrics(question)
    # print(f"RAGAS Score: {ragas_score['ragas_score']:.2f} ({ragas_score['grade']})")
    
    
    print("RAG Evaluator initialized successfully!")
    print("Import this module to use in your exam generator.")