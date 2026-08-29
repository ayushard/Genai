import streamlit as st
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NLP Analysis Studio - Annanya Kesharwani",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    :root {
        --bg-app: #0d1117;
        --bg-panel: #151b26;
        --bg-panel-alt: #1a2233;
        --border-soft: #2a3448;
        --text-primary: #e6ebf5;
        --text-secondary: #9fabc4;
        --accent: #4d8bff;
        --accent-soft: rgba(77, 139, 255, 0.14);
        --accent-border: rgba(77, 139, 255, 0.35);
    }

    .stApp {
        background: radial-gradient(circle at 12% 0%, #131b2e 0%, #0d1117 45%, #0d1117 100%);
        color: var(--text-primary);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    p, span, label, li, div {
        color: var(--text-primary);
    }

    /* -------- Hero header -------- */
    .nlp-hero {
        background: linear-gradient(120deg, #101a30 0%, #17274a 55%, #1d3a6e 100%);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }

    .nlp-hero::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(77,139,255,0.16) 0%, rgba(77,139,255,0) 70%);
        border-radius: 50%;
    }

    .nlp-hero h1 {
        color: #f5f8ff !important;
        font-weight: 800;
        font-size: 2.1rem;
        letter-spacing: -0.035em;
        margin: 0 0 0.4rem 0;
    }

    .nlp-hero p {
        color: #aebde0;
        font-size: 1.02rem;
        margin: 0;
        max-width: 640px;
    }

    .nlp-hero .badge-row {
        margin-top: 0.9rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .nlp-badge {
        display: inline-block;
        background: rgba(77, 139, 255, 0.12);
        border: 1px solid rgba(77, 139, 255, 0.3);
        color: #cfe0ff;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        letter-spacing: 0.01em;
    }

    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    h3 {
        border-left: 4px solid var(--accent);
        padding-left: 0.65rem;
        margin-top: 0.5rem !important;
        color: #dce6fb;
    }

    /* -------- Sidebar -------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1c 0%, #101a30 100%);
        border-right: 1px solid var(--border-soft);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #dbe4f7;
    }

    section[data-testid="stSidebar"] h1 {
        color: #f5f8ff;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid var(--border-soft);
        margin-bottom: 1rem;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: #6f8dc9;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    section[data-testid="stSidebar"] textarea {
        background: #0d1424;
        color: #e6ebf5;
        border: 1px solid var(--border-soft);
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.3rem;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.6rem;
        border-radius: 12px;
        border: 1px solid var(--border-soft);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.42rem 0.5rem;
        border-radius: 8px;
        font-size: 0.92rem;
        transition: background 0.15s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(77, 139, 255, 0.1);
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--border-soft);
    }

    /* -------- Buttons -------- */
    .stButton > button {
        min-height: 48px;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(90deg, #3868d6 0%, #4d8bff 100%);
        color: #ffffff;
        box-shadow: 0 8px 18px rgba(30, 70, 160, 0.35);
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #2f58bd 0%, #3f79eb 100%);
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(30, 70, 160, 0.45);
    }

    /* -------- Cards / containers -------- */
    div[data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        background: var(--bg-panel);
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stExpander"] summary {
        font-weight: 700;
        color: #dce6fb;
    }

    div[data-testid="stExpander"] summary:hover {
        color: var(--accent);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        background: var(--bg-panel-alt);
        border: 1px solid var(--border-soft);
    }

    hr {
        margin: 1.6rem 0;
        border-color: var(--border-soft);
    }

    /* -------- Token / word chips -------- */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.4rem 0 0.9rem 0;
    }

    .chip {
        display: inline-block;
        background: var(--accent-soft);
        color: #cfe0ff;
        border: 1px solid var(--accent-border);
        padding: 0.28rem 0.65rem;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .chip-muted {
        background: rgba(255, 255, 255, 0.04);
        color: #aebac9;
        border: 1px solid var(--border-soft);
    }

    .sentence-card {
        background: var(--bg-panel-alt);
        border: 1px solid var(--border-soft);
        border-left: 4px solid var(--accent);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.55rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        color: #dbe4f7;
    }

    .sentence-card b {
        color: #7fabff;
        margin-right: 0.4rem;
    }

    .noun-chunk-chip {
        display: inline-block;
        background: rgba(52, 211, 153, 0.1);
        color: #7ee9c1;
        border: 1px solid rgba(52, 211, 153, 0.32);
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 0.2rem 0.35rem 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# One-time downloads / model loading (cached so it only runs once)
# ------------------------------------------------------------------
@st.cache_resource
def load_resources():
    for pkg in ["punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    nlp_model = spacy.load("en_core_web_sm")
    return nlp_model


with st.spinner("Loading NLP models (first run only)..."):
    nlp = load_resources()

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------------
# Sidebar — text input + step selector
# ------------------------------------------------------------------
st.sidebar.title("🧠 NLP Control Panel")

default_text = (
    "Apple Inc. is planning to open a new office in Bangalore next year. "
    "Tim Cook announced this during a press conference in California."
)

st.sidebar.markdown("Input Text")
text = st.sidebar.text_area("📝 Enter Text", value=default_text, height=180, label_visibility="collapsed")

steps = [
    "1. Sentence Segmentation",
    "2. Word Tokenization",
    "3. Stop Word Removal",
    "4. Stemming",
    "5. Lemmatization",
    "6. POS Tagging",
    "7. Named Entity Recognition (NER)",
    "8. Dependency Parsing",
    "9. Noun Phrase Chunking",
    "🔁 Run All Steps",
]

st.sidebar.markdown("Pipeline Step")
selected_step = st.sidebar.radio("📌 Choose NLP Operation", steps, label_visibility="collapsed")

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
st.markdown("""
<div class="nlp-hero">
    <h1>🧠 NLP Analysis Studio</h1>
    <h2>Annanya Kesharwani - 09/08/2026
    <p>Explore a full Natural Language Processing pipeline — from tokenization to dependency
    parsing — powered by <b>NLTK</b> &amp; <b>spaCy</b>. Pick a step in the control panel, drop in your text,
    and run the analysis.</p>
    <div class="badge-row">
        <span class="nlp-badge">NLTK</span>
        <span class="nlp-badge">spaCy</span>
        <span class="nlp-badge">9 Pipeline Steps</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📄 Input Text Preview", expanded=True):
    st.write(text if text.strip() else "_No text entered yet._")


# ------------------------------------------------------------------
# Step functions
# ------------------------------------------------------------------
def step_sentence_segmentation(text):
    st.subheader("Sentence Segmentation")
    sentences = sent_tokenize(text)
    for i, sent in enumerate(sentences, 1):
        st.markdown(f'<div class="sentence-card"><b>{i}</b>{sent}</div>', unsafe_allow_html=True)
    return sentences


def step_word_tokenization(text):
    st.subheader("Word Tokenization")
    words = word_tokenize(text)
    chips = "".join(f'<span class="chip">{w}</span>' for w in words)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
    return words


def step_stopword_removal(words):
    st.subheader("After Stop Word Removal")
    filtered_words = [w for w in words if w.lower() not in stop_words]
    chips = "".join(f'<span class="chip">{w}</span>' for w in filtered_words)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
    return filtered_words


def step_stemming(filtered_words):
    st.subheader("After Stemming")
    stemmed_words = [stemmer.stem(w) for w in filtered_words]
    chips = "".join(f'<span class="chip chip-muted">{w}</span>' for w in stemmed_words)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
    return stemmed_words


def step_lemmatization(filtered_words):
    st.subheader("After Lemmatization")
    lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    chips = "".join(f'<span class="chip chip-muted">{w}</span>' for w in lemmatized_words)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
    return lemmatized_words


def step_pos_tagging(words):
    st.subheader("POS Tagging")
    pos_tags = nltk.pos_tag(words)
    st.dataframe(
        {"Word": [w for w, t in pos_tags], "POS Tag": [t for w, t in pos_tags]},
        use_container_width=True,
    )
    return pos_tags


def step_ner(doc):
    st.subheader("Named Entity Recognition (NER)")
    if doc.ents:
        st.dataframe(
            {"Entity": [ent.text for ent in doc.ents], "Label": [ent.label_ for ent in doc.ents]},
            use_container_width=True,
        )
    else:
        st.info("No named entities found.")


def step_dependency_parsing(doc):
    st.subheader("Dependency Parsing")
    st.dataframe(
        {
            "Token": [tok.text for tok in doc],
            "Dependency": [tok.dep_ for tok in doc],
            "Head": [tok.head.text for tok in doc],
        },
        use_container_width=True,
    )
    try:
        from spacy import displacy

        html = displacy.render(doc, style="dep", options={"compact": True})
        st.components.v1.html(html, height=400, scrolling=True)
    except Exception:
        pass


def step_noun_chunks(doc):
    st.subheader("Noun Phrase Chunking")
    chunks = list(doc.noun_chunks)
    if chunks:
        chips = "".join(f'<span class="noun-chunk-chip">{chunk.text}</span>' for chunk in chunks)
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.info("No noun phrase chunks found.")


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------
if run_button:
    if not text.strip():
        st.warning("Please enter some text in the sidebar first.")
    else:
        words = word_tokenize(text)
        doc = nlp(text)

        with st.container(border=True):
            if selected_step == steps[0]:
                step_sentence_segmentation(text)

            elif selected_step == steps[1]:
                step_word_tokenization(text)

            elif selected_step == steps[2]:
                step_stopword_removal(words)

            elif selected_step == steps[3]:
                filtered = [w for w in words if w.lower() not in stop_words]
                step_stemming(filtered)

            elif selected_step == steps[4]:
                filtered = [w for w in words if w.lower() not in stop_words]
                step_lemmatization(filtered)

            elif selected_step == steps[5]:
                step_pos_tagging(words)

            elif selected_step == steps[6]:
                step_ner(doc)

            elif selected_step == steps[7]:
                step_dependency_parsing(doc)

            elif selected_step == steps[8]:
                step_noun_chunks(doc)

            elif selected_step == steps[9]:  # Run All
                step_sentence_segmentation(text)
                st.divider()
                step_word_tokenization(text)
                st.divider()
                filtered = step_stopword_removal(words)
                st.divider()
                step_stemming(filtered)
                st.divider()
                step_lemmatization(filtered)
                st.divider()
                step_pos_tagging(words)
                st.divider()
                step_ner(doc)
                st.divider()
                step_dependency_parsing(doc)
                st.divider()
                step_noun_chunks(doc)
else:
    st.info("👈 Enter text and click **Run Analysis** in the control panel to see results.")

#===============commands to run==============
#pip install streamlit nltk spacy
#python -m spacy download en_core_web_sm
#streamlit run nlp_app.py
