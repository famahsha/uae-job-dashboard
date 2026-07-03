import streamlit as st
import pandas as pd
import plotly.express as px
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

# ── Title ────────────────────────────────────────────────
st.title("💼 UAE Job Market Dashboard")
st.markdown("Analysing 5,000 real data analyst job postings")

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

    # ── Two columns ──────────────────────────────────────
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
    st.header("🤖 AI Job Market Chatbot")
    st.info("💡 RAG chatbot runs locally using Ollama + ChromaDB. Clone the repo and run locally to use this feature.")

    st.markdown("### How it works:")
    st.markdown("- **ChromaDB** stores 29,621 job embeddings locally")
    st.markdown("- **Llama3.2** runs on your machine via Ollama")
    st.markdown("- **RAG pipeline** finds relevant jobs then answers")

    st.markdown("### Run locally:")
    st.code("""
git clone https://github.com/famahsha/uae-job-dashboard.git
cd uae-job-dashboard
pip install -r requirements.txt
ollama pull llama3.2:1b
streamlit run app.py
    """)