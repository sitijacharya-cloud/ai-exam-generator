"""LLM-based question generation with fallback for no context"""

from typing import List
import openai
import json
from models.schemas import Question, QuestionChoice, QuestionBlueprint, RetrievedContext
from config.settings import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    """Generates questions using LLM based on retrieved context or topic alone"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL
        self.temperature = TEMPERATURE
    
    def build_prompt_with_context(self, context: str, blueprint: QuestionBlueprint, topic: str) -> List[dict]:
        """Build prompt for LLM when context is available"""
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert exam question creator. You generate high-quality assessment questions "
                "based on the provided context from textbooks. Questions must be directly answerable "
                "from the given context. Return responses in strict JSON format."
            )
        }
        
        user_content = f"""Based on the following context from textbooks, create {blueprint.count} exam questions.

CONTEXT:
{context}

REQUIREMENTS:
- Topic: {topic}
- Question Type: {blueprint.question_type}
- Difficulty: {blueprint.difficulty}
- Count: {blueprint.count}
- Marks per question: {blueprint.marks_per_question}

IMPORTANT RULES:
1. Questions MUST be answerable from the provided context only
2. Do not ask about information not present in the context
3. For MCQs, provide 4 options (A, B, C, D) with only ONE correct answer

Return a JSON object with key 'questions'. Each question must include:
- 'type': question type
- 'difficulty': difficulty level
- 'prompt': the question text
- 'choices': for MCQ only, array of objects with 'label' (A-D) and 'text'
- 'answer': the correct answer (for MCQ: just the letter A/B/C/D)
- 'explanation': brief explanation of why this is correct
- 'marks': marks for this question
"""
        
        user_msg = {"role": "user", "content": user_content}
        return [system_msg, user_msg]
    
    def build_prompt_without_context(self, blueprint: QuestionBlueprint, topic: str) -> List[dict]:
        """Build prompt for LLM when NO context is available - generate from topic knowledge"""
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert exam question creator. Generate high-quality assessment questions "
                "based on the given topic using your knowledge. Create academically sound questions "
                "appropriate for the topic. Return responses in strict JSON format."
            )
        }
        
        user_content = f"""Create {blueprint.count} exam questions for the following topic using your general knowledge.

TOPIC: {topic}

REQUIREMENTS:
- Question Type: {blueprint.question_type}
- Difficulty: {blueprint.difficulty}
- Count: {blueprint.count}
- Marks per question: {blueprint.marks_per_question}

IMPORTANT RULES:
1. Generate questions based on standard knowledge of this topic
2. Questions should be academically appropriate and well-formed
3. For MCQs, provide 4 options (A, B, C, D) with only ONE correct answer
4. Ensure questions are clear and unambiguous

Return a JSON object with key 'questions'. Each question must include:
- 'type': question type
- 'difficulty': difficulty level
- 'prompt': the question text
- 'choices': for MCQ only, array of objects with 'label' (A-D) and 'text'
- 'answer': the correct answer (for MCQ: just the letter A/B/C/D)
- 'explanation': brief explanation of why this is correct
- 'marks': marks for this question

NOTE: Generate questions appropriate for an academic exam on "{topic}".
"""
        
        user_msg = {"role": "user", "content": user_content}
        return [system_msg, user_msg]
    
    def generate_questions(
        self,
        topic: str,
        blueprint: QuestionBlueprint,
        contexts: List[RetrievedContext]
    ) -> List[Question]:
        """
        Generate questions for a topic using retrieved contexts
        If no contexts available, generate from topic knowledge alone
        """
        
        # Determine if we have usable context
        has_context = contexts and len(contexts) > 0
        
        if has_context:
            # Use context-based generation
            combined_context = "\n\n".join([ctx.content for ctx in contexts])
            messages = self.build_prompt_with_context(combined_context, blueprint, topic)
            logger.info(f"Generating {blueprint.count} {blueprint.question_type} with context for: {topic}")
        else:
            # Use topic-only generation (fallback)
            messages = self.build_prompt_without_context(blueprint, topic)
            logger.warning(f"No context found. Generating {blueprint.count} {blueprint.question_type} from topic knowledge: {topic}")
        
        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            raw_json = json.loads(content)
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return []
        
        # Parse questions
        questions = []
        question_list = raw_json.get("questions", [])
        
        for idx, q_dict in enumerate(question_list):
            try:
                # Generate unique ID
                q_id = f"{topic.replace(' ', '_')[:30]}_{blueprint.question_type}_{idx+1}"
                
                # Parse choices for MCQ
                choices = None
                if blueprint.question_type.lower() in ["mcq", "multiple choice"]:
                    raw_choices = q_dict.get("choices", [])
                    if isinstance(raw_choices, dict):
                        choices = [
                            QuestionChoice(label=str(k), text=str(v))
                            for k, v in raw_choices.items()
                        ]
                    elif isinstance(raw_choices, list):
                        choices = [
                            QuestionChoice(**choice) if isinstance(choice, dict) else choice
                            for choice in raw_choices
                        ]
                
                # Normalize answer
                answer = q_dict.get("answer", "")
                if not isinstance(answer, str):
                    answer = str(answer)
                
                # Create question object
                question = Question(
                    id=q_id,
                    topic=topic,
                    type=blueprint.question_type,
                    difficulty=blueprint.difficulty,
                    prompt=q_dict.get("prompt", ""),
                    choices=choices,
                    answer=answer,
                    explanation=q_dict.get("explanation"),
                    marks=blueprint.marks_per_question,
                    retrieved_contexts=contexts[:3] if has_context else []  # Empty if no context
                )
                
                questions.append(question)
                
            except Exception as e:
                logger.error(f"Error parsing question {idx}: {e}")
                continue
        
        logger.info(f"✓ Generated {len(questions)} questions")
        return questions