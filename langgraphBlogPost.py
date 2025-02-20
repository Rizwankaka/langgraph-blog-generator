import streamlit as st
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import groq
import re
import os
from dotenv import load_dotenv
import time
from datetime import datetime

load_dotenv()

# Configure Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Configure Groq client
client = groq.Groq(api_key=GROQ_API_KEY)
# Define state structure
class BlogState(TypedDict):
    keyword: str
    titles: List[str]
    selected_title: Optional[str]
    blog_content: Optional[str]

# Initialize LangGraph workflow
def create_workflow():
    workflow = StateGraph(BlogState)

    # Define nodes
    def generate_titles(state: BlogState):
        prompt = f"""Generate 4 blog title options about {state['keyword']}.
        Return ONLY a numbered list following these rules:
        1. Include keyword in first 3 words
        2. Maximum 60 characters
        3. Use power words like 'Essential' or 'Definitive Guide'"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="qwen-2.5-32b",
                temperature=0.7,
                max_tokens=200
            )
            raw_output = re.sub(r'<\/?[a-zA-Z]+>', '', response.choices[0].message.content, flags=re.DOTALL).strip()
            titles = [line.split(". ", 1)[1].strip() for line in raw_output.split("\n") if ". " in line[:3]][:4]
            return {"titles": titles}
        except Exception as e:
            st.error(f"Title generation failed: {str(e)}")
            return {"titles": []}

    def generate_content(state: BlogState):
        prompt = f"""Write a comprehensive 1500-word blog post titled "{state['selected_title']}".
        Structure with markdown:
        # [Title]
        ## Introduction
        ## Main Content (3-5 sections)
        ### Subsections
        ## Conclusion
        Include practical examples and statistics."""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="qwen-2.5-32b",
                temperature=0.8,
                max_tokens=3000
            )
            content = re.sub(r'<\/?[a-zA-Z]+>', '', response.choices[0].message.content, flags=re.DOTALL).strip()
            return {"blog_content": content}
        except Exception as e:
            st.error(f"Content generation failed: {str(e)}")
            return {"blog_content": ""}

    # Add nodes to workflow
    workflow.add_node("generate_titles", generate_titles)
    workflow.add_node("generate_content", generate_content)

    # Define edges
    workflow.set_entry_point("generate_titles")
    
    def route_after_titles(state: BlogState):
        return "generate_content" if state.get("selected_title") else END

    workflow.add_conditional_edges(
        "generate_titles",
        route_after_titles
    )
    workflow.add_edge("generate_content", END)

    return workflow.compile()

# Set page config
st.set_page_config(
    page_title="AI Blog Wizard ✨",
    page_icon="📝",
    layout="wide"
)

# Initialize the app
app = create_workflow()

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        color: #1E88E5;
        padding: 2rem 0;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced UI
st.markdown("<h1 class='main-header'>✨ AI Blog Wizard</h1>", unsafe_allow_html=True)

# Initialize session state with more features
if 'blog_state' not in st.session_state:
    st.session_state.blog_state = {
        "keyword": "",
        "titles": [],
        "selected_title": None,
        "blog_content": None,
        "generation_history": [],
        "word_count": 0,
        "last_generated": None
    }

# Sidebar for settings and history
with st.sidebar:
    st.header("⚙️ Settings")
    tone_options = ["Professional", "Casual", "Technical", "Conversational"]
    selected_tone = st.selectbox("Writing Tone", tone_options)
    
    target_audience = st.selectbox("Target Audience", 
        ["General", "Beginners", "Experts", "Students", "Professionals"])
    
    word_count = st.slider("Target Word Count", 500, 2000, 1500, 100)
    
    st.header("📊 Statistics")
    if st.session_state.blog_state["last_generated"]:
        st.info(f"Last Generated: {st.session_state.blog_state['last_generated']}")
    if st.session_state.blog_state["word_count"]:
        st.info(f"Word Count: {st.session_state.blog_state['word_count']}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    keyword = st.text_input("🎯 Enter your blog topic keyword:", 
                           value=st.session_state.blog_state["keyword"],
                           placeholder="e.g., Artificial Intelligence")

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎨 Generate Titles", use_container_width=True):
        if keyword.strip():
            with st.spinner("🪄 Crafting engaging titles..."):
                new_state = app.invoke({
                    "keyword": keyword.strip(),
                    "titles": [],
                    "selected_title": None,
                    "blog_content": None
                })
                st.session_state.blog_state.update(new_state)
                st.session_state.blog_state["last_generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

# Display Titles with enhanced UI
if st.session_state.blog_state["titles"]:
    st.markdown("### 📑 Choose Your Title")
    with st.container():
        selected_idx = st.radio(
            "Select the most appealing title:",
            options=range(len(st.session_state.blog_state["titles"])),
            format_func=lambda x: f"✨ {st.session_state.blog_state['titles'][x]}",
            horizontal=True
        )
        st.session_state.blog_state["selected_title"] = st.session_state.blog_state["titles"][selected_idx]

# Generate Content with Progress
if st.session_state.blog_state["selected_title"] and not st.session_state.blog_state["blog_content"]:
    if st.button("📝 Generate Full Blog Post", use_container_width=True):
        with st.spinner("🚀 Creating your masterpiece..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            final_state = app.invoke(st.session_state.blog_state)
            st.session_state.blog_state.update(final_state)
            st.session_state.blog_state["word_count"] = len(st.session_state.blog_state["blog_content"].split())
            progress_bar.empty()

# Display Content with enhanced formatting
if st.session_state.blog_state["blog_content"]:
    st.markdown("---")
    st.markdown(f"## 📖 {st.session_state.blog_state['selected_title']}")
    
    # Add tabs for different views
    tab1, tab2 = st.tabs(["📄 Preview", "🔍 Raw Markdown"])
    
    with tab1:
        st.markdown(st.session_state.blog_state["blog_content"])
    
    with tab2:
        st.text_area("Raw Markdown", st.session_state.blog_state["blog_content"], height=300)
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download as Markdown",
            st.session_state.blog_state["blog_content"],
            file_name=f"{st.session_state.blog_state['selected_title']}.md",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            "📄 Download as Text",
            st.session_state.blog_state["blog_content"],
            file_name=f"{st.session_state.blog_state['selected_title']}.txt",
            use_container_width=True
        )

# Footer with reset button
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Start Fresh", use_container_width=True):
        st.session_state.blog_state = {
            "keyword": "",
            "titles": [],
            "selected_title": None,
            "blog_content": None,
            "generation_history": [],
            "word_count": 0,
            "last_generated": None
        }
        st.rerun()