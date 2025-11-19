"""
Main Streamlit UI for exam generation
UPDATED: 
1. Extract ALL topics (not just from knowledge base)
2. Weight-based question distribution by credit hours
"""

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
from generation.retriever import ContextRetriever
from generation.question_generator import QuestionGenerator
from models.schemas import QuestionBlueprint, ExamConfig, Question

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Exam Creator - Production RAG",
    page_icon="🎓",
    layout="wide"
)

def initialize_system():
    """Initialize the RAG system"""
    if "system_initialized" not in st.session_state:
        with st.spinner("🔄 Initializing RAG system..."):
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
                
                # Initialize retriever and generator
                retriever = ContextRetriever(vectorstore)
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
    - "Introduction to AI" -> ("Introduction to AI", 1.0)  # default weight
    """
    topics_with_weights = []
    lines = syllabus_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty or very short lines
        if len(line) < 5:
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
        
        # Only add if topic is meaningful
        if len(topic) > 5:
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
    st.title("🎓 AI Exam Creator - Production RAG System")
    
    st.markdown("""
    ### How it works:
    1. 📚 **Knowledge Base**: PDFs are pre-processed and stored in ChromaDB
    2. 📋 **Syllabus Input**: Provide topics with credit hours (optional)
    3. ⚖️ **Weighted Distribution**: Questions distributed by credit hours
    4. 🔍 **Smart Retrieval**: Finds relevant contexts when available
    5. 🤖 **Flexible Generation**: Uses context OR generates from topic knowledge
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
        st.metric("Embedding Dimensions", stats.get('embedding_dimensions', 'N/A'))
        st.info(f"**Collection:** {stats.get('collection_name', 'N/A')}")
        
        st.markdown("---")
        st.markdown("### 💡 Features")
        st.success("✅ Extracts ALL topics from syllabus")
        st.success("✅ Weight-based distribution by credit hours")
        st.success("✅ Generates from knowledge base OR general knowledge")
        
        st.markdown("---")
        st.markdown("### 📚 Credit Hour Format")
        st.code("""Examples:
1. Machine Learning (7 hr)
2. SQL Basics (3 hrs)
Unit 1: Python (5 hours)
- Data Science [4hr]
Neural Networks (no hours = 1)""")
    
    # Main content
    tab1, tab2 = st.tabs(["📋 Create Exam", "📖 View Generated Exam"])
    
    with tab1:
        st.header("Step 1: Provide Syllabus")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            syllabus_text = st.text_area(
                "Enter syllabus topics (one per line, credit hours optional)",
                height=300,
                placeholder="""1. Introduction to Machine Learning (7 hr)
2. Supervised Learning Algorithms (5 hrs)
3. Neural Networks and Deep Learning (6 hours)
4. SQL Database Basics (3 hr)
5. Natural Language Processing (4 hrs)
6. Computer Vision Basics (2 hr)""",
                help="Format: Topic name (credit hours). If no hours specified, weight = 1",
                key="syllabus_input"
            )
            
            # Button to process syllabus
            process_button = st.button("📋 Process Syllabus", type="secondary", use_container_width=True)
        
        with col2:
            st.markdown("**Tips:**")
            st.markdown("- Add credit hours in brackets: `(7 hr)`")
            st.markdown("- Formats: `(hr)`, `(hrs)`, `(hours)`")
            st.markdown("- Without hours = weight of 1")
            st.markdown("- Works for ANY topic")
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

if __name__ == "__main__":
    main()