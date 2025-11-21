


import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from typing import List, Tuple
import json
import math
import re

from config.settings import CHROMA_DB_DIR, COLLECTION_NAME, TOP_K_CONTEXTS
from vectorstore.embeddings import EmbeddingManager
from vectorstore.store import VectorStore
from generation.retriever_old import ContextRetriever
from generation.question_generator import QuestionGenerator
from models.schemas import QuestionBlueprint, ExamConfig, Question

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="RAG Based AI Exam Creator ",
    page_icon="🎓",
    layout="wide"
)

def initialize_system():
    """Initialize the RAG system"""
    if "system_initialized" not in st.session_state:
        with st.spinner(" Initializing RAG system..."):
            try:
                # Load embeddings
                embedding_manager = EmbeddingManager()
                embeddings = embedding_manager.get_embeddings()
                
                # Load vectorstore
                vectorstore_manager = VectorStore(
                    persist_directory=str(CHROMA_DB_DIR),
                    embeddings=embeddings,
                    collection_name=COLLECTION_NAME
                )
                vectorstore = vectorstore_manager.load_vectorstore()
                
                # Initialize retriever with relevance threshold
                # Threshold 0.5 = moderate strictness (recommended)
                # Higher (0.7-0.8) = very strict, only very relevant content
                # Lower (0.3-0.4) = lenient, accepts broader matches
                retriever = ContextRetriever(vectorstore, relevance_threshold=0.5)
                generator = QuestionGenerator()
                
                # Store in session state
                st.session_state["vectorstore_manager"] = vectorstore_manager
                st.session_state["retriever"] = retriever
                st.session_state["generator"] = generator
                st.session_state["system_initialized"] = True
                
                logger.info("✓ System initialized successfully")
                
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                st.info("Make sure you've run: python scripts/setup_knowledge_base.py")
                st.stop()

def extract_topics_with_weights(syllabus_text: str) -> List[Tuple[str, float]]:
    """
    Extract topics with credit hours/weights from syllabus text
    Returns list of tuples: (topic_name, credit_hours)
    
    Examples:
    - "1. Machine Learning (7 hr)" -> ("Machine Learning", 7.0)
    - "Unit 1: SQL Basics (3 hrs)" -> ("SQL Basics", 3.0)
    - "2. sql" -> ("sql", 1.0)
    - "Introduction to AI" -> ("Introduction to AI", 1.0)  # default weight
    """
    topics_with_weights = []
    lines = syllabus_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip only completely empty lines
        if len(line) < 1:
            continue
        
        # Try to extract credit hours from brackets or parentheses
        # Patterns: (7 hr), (7hrs), (7 hours), [7hr], etc.
        weight_match = re.search(r'[\(\[](\d+(?:\.\d+)?)\s*(?:hr|hrs|hour|hours)?[\)\]]', line, re.IGNORECASE)
        
        if weight_match:
            weight = float(weight_match.group(1))
            # Remove the weight part from topic name
            topic = re.sub(r'[\(\[](\d+(?:\.\d+)?)\s*(?:hr|hrs|hour|hours)?[\)\]]', '', line, flags=re.IGNORECASE)
        else:
            weight = 1.0  # Default weight if not specified
            topic = line
        
        # Clean up topic name - remove numbering, bullets, etc.
        topic = re.sub(r'^[•\-*\d\.]+\s*', '', topic)
        topic = re.sub(r'^Unit\s*\d+\s*:?\s*', '', topic, flags=re.IGNORECASE)
        topic = topic.strip()
        
        # Only add if topic has at least 2 characters (allows "ML", "AI", "SQL", etc.)
        if len(topic) >= 2:
            topics_with_weights.append((topic, weight))
    
    return topics_with_weights[:20]  # Limit to 20 topics

def calculate_weighted_distribution(topics_with_weights: List[Tuple[str, float]], total_questions: int) -> List[Tuple[str, float, int]]:
    """
    Calculate how many questions each topic should get based on weights
    
    Args:
        topics_with_weights: List of (topic, weight) tuples
        total_questions: Total number of questions to distribute
        
    Returns:
        List of (topic, weight, question_count) tuples
    """
    if not topics_with_weights:
        return []
    
    # Calculate total weight
    total_weight = sum(weight for _, weight in topics_with_weights)
    
    # Calculate questions per topic (proportional to weight)
    distribution = []
    allocated = 0
    
    for i, (topic, weight) in enumerate(topics_with_weights):
        # Calculate proportional share
        if i == len(topics_with_weights) - 1:
            # Last topic gets remaining questions to ensure exact total
            questions = total_questions - allocated
        else:
            questions = round((weight / total_weight) * total_questions)
        
        allocated += questions
        distribution.append((topic, weight, questions))
    
    return distribution

def main():
    st.title("RAG based AI Exam Creator ")
    
    st.markdown("""
    ### How it works:
    1.  **Knowledge Base**: PDFs are pre-processed and stored in ChromaDB
    2.  **Syllabus Input**: Provide topics with credit hours (optional)
    3.  **Weighted Distribution**: Questions distributed by credit hours
    4.  **Smart Retrieval**: Finds relevant contexts when available
    5.  **Flexible Generation**: Uses context OR generates from topic knowledge
    """)
    
    # Initialize system
    initialize_system()
    
    # Get system components
    vectorstore_manager = st.session_state["vectorstore_manager"]
    retriever = st.session_state["retriever"]
    generator = st.session_state["generator"]
    
    # Sidebar: System stats
    with st.sidebar:
        st.header("📊 System Status")
        
        stats = vectorstore_manager.get_statistics()
        st.metric("Total Vectors", f"{stats.get('total_vectors', 0):,}")
       
        st.info(f"**Collection:** {stats.get('collection_name', 'N/A')}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Relevance Settings")
        
        # Relevance threshold slider
        relevance_threshold = st.slider(
            "Context Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="""
            Controls how relevant retrieved content must be:
            • 0.3-0.4: Lenient (accepts broader matches)
            • 0.5-0.6: Moderate (recommended)
            • 0.7-0.8: Strict (only very relevant content)
            
            If context is below threshold, uses LLM general knowledge instead.
            """
        )
        
        # Update retriever threshold if changed
        if retriever.relevance_threshold != relevance_threshold:
            retriever.relevance_threshold = relevance_threshold
            st.success(f"✓ Threshold updated to {relevance_threshold:.2f}")
        
        st.markdown("---")
        st.markdown("### 💡 Features")
        
        st.success(" Weight-based distribution by credit hours")
        st.success(" Relevance filtering for accurate questions")
        st.success(" Generates from Knowledge Base OR general knowledge")
        
        
    
    # Main content
    tab1, tab2 = st.tabs(["📋 Create Exam", "📖 View Generated Exam"])
    
    with tab1:
        st.header("Step 1: Provide Syllabus")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            syllabus_text = st.text_area(
                "Enter syllabus topics (one per line, credit hours optional)",
                height=300,
                placeholder="""Enter the topics or chapters from which you want to generate questions.""",
                help="Format: Topic name (credit hours). Short names like 'SQL', 'AI' work too! If no hours specified, weight = 1",
                key="syllabus_input"
            )
            
            # Button to process syllabus
            process_button = st.button("📋 Process Syllabus", type="secondary", use_container_width=True)
        
        with col2:
            st.markdown("**Tips:**")
            st.markdown("- Add credit hours in brackets: `(7 hr)`")
            st.markdown("- Formats: `(hr)`, `(hrs)`, `(hours)`")
            st.markdown("- Without hours = weight of 1")
            
            st.markdown("- Questions distributed by weight")
        
        # Initialize session state for topics
        if "extracted_topics" not in st.session_state:
            st.session_state["extracted_topics"] = []
        
        # Only extract topics when button is clicked
        if process_button and syllabus_text.strip():
            topics_with_weights = extract_topics_with_weights(syllabus_text)
            st.session_state["extracted_topics"] = topics_with_weights
            
            if topics_with_weights:
                st.success(f"✅ Extracted {len(topics_with_weights)} topics")
            else:
                st.warning("⚠️ No topics found. Please check your format.")
        
        # Display extracted topics if available
        if st.session_state["extracted_topics"]:
            topics_with_weights = st.session_state["extracted_topics"]
            
            with st.expander("📚 Extracted Topics & Weights", expanded=True):
                total_weight = sum(w for _, w in topics_with_weights)
                
                # Display in a nice table format
                st.markdown("| # | Topic | Credit Hours | Weight % |")
                st.markdown("|---|-------|--------------|----------|")
                for i, (topic, weight) in enumerate(topics_with_weights, 1):
                    weight_percent = (weight / total_weight) * 100
                    st.markdown(f"| {i} | {topic} | {weight} | {weight_percent:.1f}% |")
                
                st.info(f"📊 Total credit hours: {total_weight}")
            
            st.markdown("---")
            st.header("Step 2: Configure Exam Blueprint")
            
            exam_title = st.text_input("Exam Title", value="Summative Assessment")
            exam_instructions = st.text_area(
                "Exam Instructions",
                value="Answer all questions. Each question is based on your course materials.",
                height=80
            )
            
            st.subheader("Question Types")
            st.info("💡 **Note**: Questions will be distributed across topics based on their credit hours")
            
            blueprints = []
            
            # MCQ
            with st.expander("📝 Multiple Choice Questions (MCQ)"):
                col1, col2, col3 = st.columns(3)
                mcq_count = col1.number_input("Total Count", 0, 100, 10, key="mcq_count", 
                                              help="Total MCQs (distributed by weight)")
                mcq_diff = col2.selectbox("Difficulty", ["Easy", "Moderate", "Hard"], index=1, key="mcq_diff")
                mcq_marks = col3.number_input("Marks", 1, 10, 1, key="mcq_marks")
                
                if mcq_count > 0:
                    blueprints.append(QuestionBlueprint(
                        question_type="Multiple Choice",
                        count=mcq_count,
                        difficulty=mcq_diff,
                        marks_per_question=mcq_marks
                    ))
                    
                    # Show distribution preview
                    if mcq_count > 0:
                        distribution = calculate_weighted_distribution(topics_with_weights, mcq_count)
                        st.markdown("**Distribution Preview:**")
                        for topic, weight, q_count in distribution:
                            if q_count > 0:
                                st.caption(f"  • {topic}: {q_count} MCQs (weight: {weight})")
            
            # True/False
            with st.expander("✔️ True/False Questions"):
                col1, col2, col3 = st.columns(3)
                tf_count = col1.number_input("Total Count", 0, 100, 5, key="tf_count",
                                             help="Total T/F (distributed by weight)")
                tf_diff = col2.selectbox("Difficulty", ["Easy", "Moderate", "Hard"], index=0, key="tf_diff")
                tf_marks = col3.number_input("Marks", 1, 10, 1, key="tf_marks")
                
                if tf_count > 0:
                    blueprints.append(QuestionBlueprint(
                        question_type="True/False",
                        count=tf_count,
                        difficulty=tf_diff,
                        marks_per_question=tf_marks
                    ))
            
            # Short Answer
            with st.expander("📄 Short Answer Questions"):
                col1, col2, col3 = st.columns(3)
                sa_count = col1.number_input("Total Count", 0, 100, 5, key="sa_count",
                                             help="Total short answer (distributed by weight)")
                sa_diff = col2.selectbox("Difficulty", ["Easy", "Moderate", "Hard"], index=1, key="sa_diff")
                sa_marks = col3.number_input("Marks", 1, 10, 2, key="sa_marks")
                
                if sa_count > 0:
                    blueprints.append(QuestionBlueprint(
                        question_type="Short Answer",
                        count=sa_count,
                        difficulty=sa_diff,
                        marks_per_question=sa_marks
                    ))
            
            # Long Answer
            with st.expander("📋 Long Answer Questions"):
                col1, col2, col3 = st.columns(3)
                la_count = col1.number_input("Total Count", 0, 100, 2, key="la_count",
                                             help="Total long answer (distributed by weight)")
                la_diff = col2.selectbox("Difficulty", ["Easy", "Moderate", "Hard"], index=2, key="la_diff")
                la_marks = col3.number_input("Marks", 1, 20, 5, key="la_marks")
                
                if la_count > 0:
                    blueprints.append(QuestionBlueprint(
                        question_type="Long Answer",
                        count=la_count,
                        difficulty=la_diff,
                        marks_per_question=la_marks
                    ))
            
            st.markdown("---")
            
            # Show overall distribution preview
            if blueprints:
                st.subheader("📊 Overall Question Distribution")
                
                for bp in blueprints:
                    st.markdown(f"**{bp.question_type}** ({bp.count} questions):")
                    distribution = calculate_weighted_distribution(topics_with_weights, bp.count)
                    
                    cols = st.columns(min(len(distribution), 4))
                    for idx, (topic, weight, q_count) in enumerate(distribution):
                        if q_count > 0:
                            col_idx = idx % 4
                            cols[col_idx].metric(
                                f"{topic[:20]}...",
                                f"{q_count} Q",
                                f"{weight} hr"
                            )
            
            # Generate button
            if st.button("🚀 Generate Exam", type="primary", use_container_width=True):
                if not blueprints:
                    st.error("❌ Please configure at least one question type")
                else:
                    generate_exam_weighted(topics_with_weights, blueprints, exam_title, exam_instructions, retriever, generator)
        else:
            # Show message when no topics extracted yet
            st.info("👆 Enter your syllabus topics above and click 'Process Syllabus' to continue")
    
    with tab2:
        if "generated_questions" in st.session_state:
            display_generated_exam()
        else:
            st.info("👈 Generate an exam first in the 'Create Exam' tab")

def generate_exam_weighted(topics_with_weights, blueprints, title, instructions, retriever, generator):
    """
    Generate complete exam with weighted question distribution
    """
    all_questions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total questions needed
    total_questions_needed = sum(bp.count for bp in blueprints)
    questions_generated = 0
    
    # Track statistics
    questions_with_context = 0
    questions_without_context = 0
    
    st.info(f"📝 Generating {total_questions_needed} total questions with weighted distribution...")
    
    # Process each blueprint (question type)
    for blueprint in blueprints:
        st.markdown(f"### Processing: {blueprint.question_type} ({blueprint.count} questions)")
        
        # Calculate weighted distribution for this question type
        distribution = calculate_weighted_distribution(topics_with_weights, blueprint.count)
        
        # Show distribution
        dist_text = ", ".join([f"{topic[:15]}: {q_count}" for topic, _, q_count in distribution if q_count > 0])
        st.caption(f"Distribution: {dist_text}")
        
        # Generate questions for each topic according to distribution
        for topic, weight, q_count in distribution:
            if q_count <= 0:
                continue
            
            with st.spinner(f"🔍 Processing: {topic} ({q_count} questions, weight: {weight} hr)"):
                # Try to retrieve contexts
                contexts = retriever.retrieve_for_topic(topic, k=TOP_K_CONTEXTS)
                
                if not contexts or len(contexts) == 0:
                    # No context found - generate from topic knowledge
                    st.warning(f"⚠️ No relevant content in knowledge base for: {topic}")
                    st.info(f"🤖 Generating from general knowledge")
                    contexts = []
                else:
                    st.success(f"✓ Found {len(contexts)} relevant contexts")
                
                # Create blueprint for this topic
                topic_blueprint = QuestionBlueprint(
                    question_type=blueprint.question_type,
                    count=q_count,
                    difficulty=blueprint.difficulty,
                    marks_per_question=blueprint.marks_per_question
                )
                
                # Generate questions
                questions = generator.generate_questions(topic, topic_blueprint, contexts)
                
                # Track statistics
                if contexts:
                    questions_with_context += len(questions)
                else:
                    questions_without_context += len(questions)
                
                # Take exact number needed
                questions = questions[:q_count]
                all_questions.extend(questions)
                
                questions_generated += len(questions)
                
                # Update progress
                progress = questions_generated / total_questions_needed
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Generated {questions_generated}/{total_questions_needed} questions")
                
                source_type = "from knowledge base" if contexts else "from general knowledge"
                st.success(f"✓ Generated {len(questions)} questions {source_type}")
        
        st.markdown("---")
    
    if all_questions:
        st.session_state["generated_questions"] = all_questions
        st.session_state["exam_title"] = title
        st.session_state["exam_instructions"] = instructions
        
        # Show summary
        st.success(f"🎉 Successfully generated {len(all_questions)} questions!")
        
        # Show generation statistics
        st.markdown("### 📊 Generation Summary:")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Questions", len(all_questions))
        col2.metric("From Knowledge Base", questions_with_context)
        col3.metric("From General Knowledge", questions_without_context)
        
        # Count by type and topic
        type_counts = {}
        topic_counts = {}
        for q in all_questions:
            type_counts[q.type] = type_counts.get(q.type, 0) + 1
            topic_counts[q.topic] = topic_counts.get(q.topic, 0) + 1
        
        st.markdown("#### By Question Type:")
        cols = st.columns(len(type_counts))
        for i, (qtype, count) in enumerate(type_counts.items()):
            cols[i].metric(qtype, count)
        
        st.markdown("#### By Topic:")
        cols = st.columns(min(len(topic_counts), 4))
        for i, (topic, count) in enumerate(topic_counts.items()):
            col_idx = i % 4
            cols[col_idx].metric(topic[:20], count)
        
        st.balloons()
    else:
        st.error("❌ No questions were generated. Please try again.")

def display_generated_exam():
    """Display the generated exam"""
    questions = st.session_state["generated_questions"]
    title = st.session_state.get("exam_title", "Exam")
    instructions = st.session_state.get("exam_instructions", "")
    
    st.title(f"📄 {title}")
    st.info(instructions)
    
    # Show overall statistics
    st.markdown("### 📊 Exam Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    type_counts = {}
    topic_counts = {}
    total_marks = 0
    questions_with_sources = 0
    
    for q in questions:
        type_counts[q.type] = type_counts.get(q.type, 0) + 1
        topic_counts[q.topic] = topic_counts.get(q.topic, 0) + 1
        total_marks += q.marks
        if q.retrieved_contexts and len(q.retrieved_contexts) > 0:
            questions_with_sources += 1
    
    col1.metric("Total Questions", len(questions))
    col2.metric("Total Marks", total_marks)
    col3.metric("Topics Covered", len(topic_counts))
    col4.metric("With Sources", questions_with_sources)
    
    # Show topic distribution
    with st.expander("📊 Questions per Topic"):
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{topic}**: {count} questions")
    
    st.markdown("---")
    
    # Group by type
    questions_by_type = {}
    for q in questions:
        if q.type not in questions_by_type:
            questions_by_type[q.type] = []
        questions_by_type[q.type].append(q)
    
    # Display questions
    for qtype, qs in questions_by_type.items():
        st.header(f"📝 {qtype} ({len(qs)} questions)")
        
        for i, q in enumerate(qs, 1):
            with st.container():
                st.markdown(f"### Question {i} - {q.topic}")
                st.markdown(f"**Difficulty:** {q.difficulty} | **Marks:** {q.marks}")
                
                # Show source indicator
                if q.retrieved_contexts and len(q.retrieved_contexts) > 0:
                    st.caption("📚 Generated from knowledge base")
                else:
                    st.caption("🤖 Generated from general knowledge")
                
                st.markdown(q.prompt)
                
                # Show choices for MCQ
                if q.choices:
                    for choice in q.choices:
                        st.markdown(f"**{choice.label}.** {choice.text}")
                
                # Show retrieved contexts only if available
                if q.retrieved_contexts and len(q.retrieved_contexts) > 0:
                    with st.expander("🔍 View Source Context", expanded=False):
                        st.markdown("**This question was generated from:**")
                        
                        for idx, ctx in enumerate(q.retrieved_contexts, 1):
                            st.markdown(f"**📖 Source {idx}:** {ctx.metadata['source_file']} (Page {ctx.metadata['page']})")
                            st.info(ctx.metadata['preview'])
                            st.caption(f"Relevance: {ctx.relevance_score:.2%}")
                else:
                    with st.expander("ℹ️ Generation Info", expanded=False):
                        st.info(f"Generated using AI's general knowledge about '{q.topic}'")
                
                st.markdown("---")
    
    # Answer key
    st.header("🔑 Answer Key")
    for qtype, qs in questions_by_type.items():
        st.subheader(qtype)
        for i, q in enumerate(qs, 1):
            with st.expander(f"Q{i}: {q.prompt[:80]}..."):
                st.markdown(f"**Topic:** {q.topic}")
                st.markdown(f"**Answer:** {q.answer}")
                if q.explanation:
                    st.markdown(f"**Explanation:** {q.explanation}")
    
    # In ui/app.py, add evaluation after generation

from evaluation.rag_evaluator import RAGEvaluator

# After generating questions
if st.button("🔍 Evaluate Exam Quality"):
    evaluator = RAGEvaluator()
    
    # Evaluate batch (sample 10 questions for speed)
    results = evaluator.evaluate_exam_batch(
        questions=all_questions,
        sample_size=10
    )
    
    # Display results
    st.subheader("📊 Evaluation Results")
    st.metric("Overall RAGAS Score", f"{results['summary']['overall_ragas_score']:.2%}")
    st.metric("Faithfulness", f"{results['summary']['avg_faithfulness']:.2%}")
    st.metric("Relevance", f"{results['summary']['avg_relevance']:.2%}")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    for rec in results['recommendations']:
        st.info(rec)
    # Download options
    st.markdown("---")
    st.header("💾 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        export_data = {
            "title": title,
            "instructions": instructions,
            "questions": [q.dict() for q in questions]
        }
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            "📥 Download as JSON",
            json_str,
            f"{title.replace(' ', '_')}.json",
            "application/json"
        )
    
    with col2:
        # Text export
        text_parts = [f"{title}\n{'='*len(title)}\n\n{instructions}\n\n"]
        
        for qtype, qs in questions_by_type.items():
            text_parts.append(f"\n{qtype} ({len(qs)} questions)\n{'-'*len(qtype)}\n\n")
            for i, q in enumerate(qs, 1):
                text_parts.append(f"Q{i}. [{q.topic}] {q.prompt}\n")
                if q.choices:
                    for choice in q.choices:
                        text_parts.append(f"  {choice.label}. {choice.text}\n")
                text_parts.append("\n")
        
        text_parts.append(f"\n\nANSWER KEY\n{'='*50}\n\n")
        for qtype, qs in questions_by_type.items():
            text_parts.append(f"\n{qtype}\n")
            for i, q in enumerate(qs, 1):
                text_parts.append(f"Q{i}. {q.answer}\n")
        
        text_content = "".join(text_parts)
        
        st.download_button(
            "📥 Download as Text",
            text_content,
            f"{title.replace(' ', '_')}.txt",
            "text/plain"
        )


def show_evaluation_dashboard():
    """Display RAG evaluation dashboard with advanced metrics"""
    st.header("📊 RAG Evaluation Dashboard")
    
    questions = st.session_state.get("generated_questions", [])
    
    if not questions:
        st.warning("No questions to evaluate")
        return
    
    # Check if advanced evaluator is available
    if EVALUATOR_AVAILABLE and "evaluator" in st.session_state:
        show_advanced_evaluation(questions)
    else:
        show_basic_evaluation(questions)

def show_advanced_evaluation(questions):
    """Advanced evaluation using RAG Evaluator"""
    evaluator = st.session_state["evaluator"]
    
    st.info("🚀 Using Advanced RAG Evaluation (RAGAS Framework)")
    
    # Evaluation options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("Evaluating exam quality using industry-standard RAGAS metrics...")
    
    with col2:
        sample_size = st.selectbox(
            "Sample Size",
            [5, 10, 20, "All"],
            index=1,
            help="Number of questions to evaluate (smaller = faster)"
        )
    
    if st.button("🔍 Run Advanced Evaluation", type="primary"):
        with st.spinner("Evaluating questions... This may take a minute..."):
            # Determine sample size
            if sample_size == "All":
                eval_sample = None
            else:
                eval_sample = int(sample_size)
            
            # Run batch evaluation
            results = evaluator.evaluate_exam_batch(questions, sample_size=eval_sample)
            
            # Store results
            st.session_state["evaluation_results"] = results
    
    # Display results if available
    if "evaluation_results" in st.session_state:
        results = st.session_state["evaluation_results"]
        summary = results["summary"]
        
        st.markdown("---")
        st.markdown("### 🎯 Overall RAGAS Score")
        
        ragas_score = summary["overall_ragas_score"]
        
        # Determine grade
        if ragas_score >= 0.8:
            grade = "A"
            color = "🟢"
            status = "Excellent"
        elif ragas_score >= 0.7:
            grade = "B"
            color = "🟡"
            status = "Good"
        elif ragas_score >= 0.6:
            grade = "C"
            color = "🟠"
            status = "Fair"
        else:
            grade = "D"
            color = "🔴"
            status = "Needs Improvement"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric("RAGAS Score", f"{ragas_score:.1%}", help="Overall RAG quality (target: ≥70%)")
        
        with col2:
            st.metric("Grade", f"{color} {grade}")
        
        with col3:
            st.metric("Status", status)
        
        # Component scores
        st.markdown("### 📈 Component Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Faithfulness",
            f"{summary['avg_faithfulness']:.1%}",
            help="Are questions grounded in context? (target: ≥90%)"
        )
        
        col2.metric(
            "Relevance",
            f"{summary['avg_relevance']:.1%}",
            help="Do questions test the intended topic? (target: ≥80%)"
        )
        
        col3.metric(
            "Context Usage",
            f"{summary['context_usage_rate']:.1%}",
            help="% questions using textbook content"
        )
        
        # Score distribution
        st.markdown("### 📊 Score Distribution")
        
        dist = summary["score_distribution"]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Excellent (A)", dist["excellent (≥0.8)"])
        col2.metric("Good (B)", dist["good (0.7-0.8)"])
        col3.metric("Fair (C)", dist["fair (0.6-0.7)"])
        col4.metric("Poor (D-F)", dist["poor (<0.6)"])
        
        # Source diversity
        if "source_diversity" in results:
            st.markdown("### 🗂️ Source Diversity")
            
            diversity = results["source_diversity"]
            
            col1, col2 = st.columns(2)
            
            col1.metric("Diversity Score", f"{diversity['score']:.1%}")
            col2.metric("Unique Sources", diversity["unique_sources"])
            
            if "source_distribution" in diversity:
                with st.expander("View Source Distribution"):
                    for source, count in diversity["source_distribution"].items():
                        st.markdown(f"**{source}**: {count} retrievals")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        for rec in results["recommendations"]:
            if "⚠️" in rec:
                st.warning(rec)
            elif "💡" in rec:
                st.info(rec)
            else:
                st.success(rec)
        
        # Detailed results
        with st.expander("📋 Detailed Question Scores"):
            for result in results["detailed_results"][:10]:
                ragas = result["ragas"]
                
                st.markdown(f"**{result['topic']}** ({result['type']})")
                st.progress(ragas["ragas_score"])
                st.caption(f"Score: {ragas['ragas_score']:.1%} - {ragas['interpretation']}")
                st.markdown("---")
        
        # Export report
        st.markdown("### 💾 Export Report")
        
        report_json = json.dumps(results, indent=2)
        
        st.download_button(
            "📥 Download Evaluation Report",
            report_json,
            "ragas_evaluation_report.json",
            "application/json"
        )

def show_basic_evaluation(questions):
    """Basic evaluation (fallback when RAG evaluator not available)"""
    st.info("ℹ️ Using Basic Evaluation (Advanced evaluator not loaded)")
    
    import numpy as np
    
    # Context usage
    questions_with_context = sum(1 for q in questions if q.retrieved_contexts)
    questions_without_context = len(questions) - questions_with_context
    context_usage_rate = questions_with_context / len(questions) if questions else 0
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Questions", len(questions))
    col2.metric("With Context", questions_with_context)
    col3.metric("Without Context", questions_without_context)
    
    # Relevance analysis
    if questions_with_context > 0:
        st.markdown("### 🎯 Relevance Score Analysis")
        
        all_relevance_scores = []
        for q in questions:
            if q.retrieved_contexts:
                scores = [ctx.relevance_score for ctx in q.retrieved_contexts]
                all_relevance_scores.extend(scores)
        
        if all_relevance_scores:
            avg_relevance = np.mean(all_relevance_scores)
            min_relevance = np.min(all_relevance_scores)
            max_relevance = np.max(all_relevance_scores)
            
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Average", f"{avg_relevance:.1%}")
            col2.metric("Min", f"{min_relevance:.1%}")
            col3.metric("Max", f"{max_relevance:.1%}")
            
            if avg_relevance >= 0.7:
                st.success("✅ Excellent context relevance")
            elif avg_relevance >= 0.5:
                st.info("✓ Good context relevance")
            else:
                st.warning("⚠️ Consider adjusting threshold")
    
    # Topic distribution
    st.markdown("### 📚 Topic Distribution")
    
    topic_counts = {}
    for q in questions:
        topic_counts[q.topic] = topic_counts.get(q.topic, 0) + 1
    
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(questions)) * 100
        st.markdown(f"**{topic}**: {count} ({percentage:.1f}%)")
        st.progress(percentage / 100)
    
    # Source diversity
    st.markdown("### 🗂️ Source Diversity")
    
    source_counts = {}
    for q in questions:
        if q.retrieved_contexts:
            for ctx in q.retrieved_contexts:
                source = ctx.metadata.get('source_file', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
    
    if source_counts:
        unique_sources = len(source_counts)
        st.metric("Unique Sources", unique_sources)
        
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{source}**: {count} retrievals")
    else:
        st.info("No sources (all from general knowledge)")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    recommendations = []
    
    if context_usage_rate < 0.3:
        recommendations.append("⚠️ Low context usage (<30%). Lower threshold or add textbooks.")
    
    if questions_with_context > 0 and all_relevance_scores:
        if avg_relevance < 0.5:
            recommendations.append("⚠️ Low relevance. Retrieved contexts may not be relevant.")
        elif avg_relevance >= 0.7:
            recommendations.append("✅ Excellent context relevance!")
    
    if len(source_counts) == 1 and questions_with_context > 5:
        recommendations.append("💡 Single source. Add more textbooks for diversity.")
    elif len(source_counts) >= 3:
        recommendations.append("✅ Good source diversity!")
    
    if context_usage_rate > 0.7:
        recommendations.append("✅ High context usage! Questions grounded in textbooks.")
    
    if not recommendations:
        recommendations.append("✅ Exam quality looks good!")
    
    for rec in recommendations:
        if "⚠️" in rec:
            st.warning(rec)
        elif "💡" in rec:
            st.info(rec)
        else:
            st.success(rec)
    
    # Sample analysis
    st.markdown("### 🔍 Sample Questions")
    
    if questions_with_context > 0:
        questions_with_scores = []
        for q in questions:
            if q.retrieved_contexts:
                avg_score = sum(ctx.relevance_score for ctx in q.retrieved_contexts) / len(q.retrieved_contexts)
                questions_with_scores.append((q, avg_score))
        
        questions_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        st.markdown("#### ✅ Top Questions (Highest Relevance)")
        for i, (q, score) in enumerate(questions_with_scores[:3], 1):
            with st.expander(f"{i}. {q.topic} - {score:.1%}"):
                st.markdown(f"**Question:** {q.prompt}")
                st.markdown(f"**Relevance:** {score:.1%}")
    
    # Export
    st.markdown("### 💾 Export Report")
    
    report = {
        "total_questions": len(questions),
        "with_context": questions_with_context,
        "without_context": questions_without_context,
        "context_usage_rate": float(context_usage_rate),
        "topic_distribution": topic_counts,
        "source_distribution": source_counts,
        "recommendations": recommendations
    }
    
    report_json = json.dumps(report, indent=2)
    
    st.download_button(
        "📥 Download Basic Report",
        report_json,
        "basic_evaluation_report.json",
        "application/json"
    )
if __name__ == "__main__":
    main()
