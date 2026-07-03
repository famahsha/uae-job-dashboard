import streamlit as st
import pandas as pd
import plotly.express as px
import chromadb
import requests
import ast
from collections import Counter

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="UAE Job Market Dashboard",
    page_icon="💼",
    layout="wide"
)

# ── Load data ────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/jobs_sample.csv")
    return df

df = load_data()

# ── Load ChromaDB ────────────────────────────────────────
@st.cache_resource
def load_chromadb():
    client = chromadb.PersistentClient(path="rag/vectorstore/chroma_db")
    collection = client.get_collection("jobs")
    return collection

collection = load_chromadb()

# ── Title ────────────────────────────────────────────────
st.title("💼 UAE Job Market Dashboard")
st.markdown("Analyse 61,951 real data analyst job postings")

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "🤖 AI Chatbot"]
)

# ════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.header("📊 Job Market Analysis")

    # ── Metrics row ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", f"{len(df):,}")
    col2.metric("Companies", f"{df['company_name'].nunique():,}")
    col3.metric("Locations", f"{df['location'].nunique():,}")
    col4.metric("With Salary", f"{df['salary_standardized'].notna().sum():,}")

    st.divider()

    # ── Top skills chart ─────────────────────────────────
    st.subheader("🔧 Top 10 Most In-Demand Skills")

    all_skills = []
    for skills in df['skills'].dropna():
        try:
            skill_list = ast.literal_eval(skills)
            all_skills.extend(skill_list)
        except:
            pass

    skill_counts = Counter(all_skills)
    top_skills = pd.DataFrame(
        skill_counts.most_common(10),
        columns=['skill', 'count']
    )

    fig1 = px.bar(
        top_skills,
        x='count',
        y='skill',
        orientation='h',
        color='count',
        color_continuous_scale='Blues',
        title='Top 10 Skills in Job Postings'
    )
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ── Two columns for charts ───────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏢 Top 10 Companies Hiring")
        top_companies = df['company_name'].value_counts().head(10).reset_index()
        top_companies.columns = ['company', 'count']
        fig2 = px.bar(
            top_companies,
            x='count',
            y='company',
            orientation='h',
            color='count',
            color_continuous_scale='Greens',
        )
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("⏰ Job Schedule Types")
        schedule = df['schedule_type'].value_counts().head(6).reset_index()
        schedule.columns = ['type', 'count']
        fig3 = px.pie(
            schedule,
            values='count',
            names='type',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Salary distribution ──────────────────────────────
    st.subheader("💰 Salary Distribution")
    df_salary = df[df['salary_standardized'].notna()]
    fig4 = px.histogram(
        df_salary,
        x='salary_standardized',
        nbins=30,
        color_discrete_sequence=['#1F77B4'],
        title='Annual Salary Distribution (USD)'
    )
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 2 — AI CHATBOT
# ════════════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":

    st.header("🤖 Ask About UAE Job Market")
    st.markdown("Ask any question about the job data — powered by RAG + Llama3.2")

    # ── Chat history ─────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat input ───────────────────────────────────────
    if prompt := st.chat_input("Ask about jobs, skills, salaries..."):

        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get answer from RAG
        with st.chat_message("assistant"):
            with st.spinner("Searching job data..."):

                # Search ChromaDB
                results = collection.query(
                    query_texts=[prompt],
                    n_results=5
                )
                context = "\n".join(results['documents'][0])

                # Send to Ollama
                rag_prompt = f"""You are a UAE job market analyst.
Answer in exactly this format — nothing else:

ANSWER: [one sentence direct answer]

KEY POINTS:
- [point 1 — maximum 10 words]
- [point 2 — maximum 10 words]
- [point 3 — maximum 10 words]

DATA SOURCE: [which jobs/companies this came from]

Job Data:
{context}

Question: {prompt}

Follow the format exactly. Be specific. No paragraphs."""

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:1b",
                        "prompt": rag_prompt,
                        "stream": False
                    }
                )

                answer = response.json()["response"]
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # ── Suggested questions ──────────────────────────────
    st.divider()
    st.markdown("**💡 Try asking:**")
    col1, col2 = st.columns(2)
    col1.info("What skills are most needed for data analyst jobs?")
    col1.info("Which companies hire the most?")
    col2.info("What is the salary range for data analysts?")
    col2.info("What schedule types are most common?")