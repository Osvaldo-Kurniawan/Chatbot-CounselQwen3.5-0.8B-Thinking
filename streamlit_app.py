"""
streamlit_app.py
Serene AI — calm, emotionally safe mental wellness chat interface.
Model inference  →  model.py
Tokenization     →  tokenizer.py
"""

import streamlit as st
import model_llama as model
from model import get_model, stream_response
from tokenizer import MODEL_NAME, get_tokenizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Serene AI",
    page_icon="🌿",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

    <style>
    /* ── CSS Variables ── */
    :root {
        --bg:           #FAF5F2;
        --bg-card:      #FFFFFF;
        --accent:       #8B6355;
        --accent-light: #C4956A;
        --sage:         #E8EDE8;
        --lavender:     #EAE8F0;
        --beige:        #F0EDE8;
        --bubble-user:  #EDE8F5;
        --bubble-ai:    #FFFFFF;
        --text-main:    #2D2D2D;
        --text-muted:   #888888;
        --text-soft:    #B0A090;
        --border:       rgba(139,99,85,0.12);
        --shadow-sm:    0 2px 12px rgba(139,99,85,0.07);
        --shadow-md:    0 6px 28px rgba(139,99,85,0.11);
        --shadow-lg:    0 16px 48px rgba(139,99,85,0.14);
        --icon-bg:      #4A5E57;
        --radius-lg:    20px;
        --radius-pill:  100px;
    }

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-main) !important;
    }

    .stApp {
        background: var(--bg) !important;
        background-image:
            radial-gradient(ellipse at 20% 0%, rgba(200,180,165,0.18) 0%, transparent 55%),
            radial-gradient(ellipse at 80% 100%, rgba(180,190,175,0.14) 0%, transparent 55%) !important;
        min-height: 100vh;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 7rem !important;
        max-width: 720px;
        margin: 0 auto;
    }

    /* ── Header ── */
    .serene-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        padding: 1.4rem 0 1.1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.25rem;
    }

    /* Wrap brand + tag */
    .serene-header .brand-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .serene-header .logo-circle { 
        width: 38px; 
        height: 38px; 
        border-radius: 50%; 
        background: linear-gradient(135deg, #C4956A 0%, #8B6355 55%, #4A5E57 100%); 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-size: 1rem; 
        box-shadow: 0 3px 10px rgba(139,99,85,0.25); 
        flex-shrink: 0; 
    }

    /* Brand */
    .serene-header .brand {
        font-family: 'Lora', serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--accent);
        letter-spacing: -0.01em;
    }

    /* Model tag under it */
    .serene-header .model-tag {
        font-size: 0.68rem;
        color: var(--text-soft);
        background: var(--beige);
        border: 1px solid var(--border);
        border-radius: var(--radius-pill);
        padding: 3px 10px;
        letter-spacing: 0.04em;
        margin-top: 4px;
    }

    /* ── Welcome / empty state ── */
    .welcome-hero {
        text-align: center;
        padding: 2.8rem 1rem 1.5rem;
    }
    .welcome-hero h2 {
        font-family: 'Lora', serif;
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-main);
        letter-spacing: -0.025em;
        line-height: 1.25;
        margin: 0 0 0.7rem;
    }
    .welcome-hero p {
        font-size: 0.92rem;
        color: var(--text-muted);
        font-weight: 300;
        line-height: 1.75;
        margin: 0 auto 2rem;
        max-width: 360px;
    }

    /* Session Cards */
    .session-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 2rem;
    }
    .session-card {
        border-radius: var(--radius-lg);
        padding: 1.1rem 1rem 1rem;
        text-align: left;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }
    .session-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
    }
    .session-card.sage  { background: var(--sage);    border: 1px solid rgba(150,180,150,0.2); }
    .session-card.lav   { background: var(--lavender); border: 1px solid rgba(160,150,200,0.2); }
    .session-card.beige { background: var(--beige);   border: 1px solid rgba(180,160,140,0.2); }

    .session-card .sc-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: var(--icon-bg);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        margin-bottom: 0.7rem;
    }
    .session-card .sc-title {
        font-family: 'Lora', serif;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 0.25rem;
    }
    .session-card .sc-desc {
        font-size: 0.76rem;
        color: var(--text-muted);
        line-height: 1.55;
        font-weight: 300;
    }

    /* Hero landscape image strip */
    .hero-image {
        width: 100%;
        height: 160px;
        border-radius: var(--radius-lg);
        background: linear-gradient(
            120deg,
            #d4a882 0%,
            #c4906a 25%,
            #a07858 45%,
            #7a9e8c 75%,
            #5a7a6e 100%
        );
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-image::before {
        content: '';
        position: absolute;
        inset: 0;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='160'%3E%3Ccircle cx='80' cy='100' r='60' fill='rgba(255,220,180,0.18)'/%3E%3Ccircle cx='200' cy='130' r='90' fill='rgba(255,200,150,0.12)'/%3E%3Ccircle cx='340' cy='80' r='70' fill='rgba(100,160,140,0.2)'/%3E%3C/svg%3E") center/cover;
    }
    .hero-image .hero-text {
        position: relative;
        z-index: 1;
        font-family: 'Lora', serif;
        font-style: italic;
        font-size: 1.05rem;
        color: rgba(255,255,255,0.9);
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
        letter-spacing: 0.01em;
    }

    /* ── Date separator ── */
    .date-sep {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 1.2rem 0 0.8rem;
        padding: 0 0.25rem;
    }
    .date-sep::before, .date-sep::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }
    .date-sep span {
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        color: var(--text-soft);
        font-weight: 500;
        text-transform: uppercase;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.2rem 0 !important;
        border: none !important;
        box-shadow: none !important;
        display: flex !important;
        justify-content: flex-start !important;
    }

    /* User bubble — right-aligned lavender */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stMarkdownContainer"] {
        background: var(--bubble-user) !important;
        border: 1px solid var(--border) !important;
        border-radius: 5px 18px 18px 18px !important;
        padding: 0.75rem 1.1rem !important;
        color: #3a3050 !important;
        font-size: 0.90rem;
        line-height: 1.7;
        font-family: 'DM Sans', sans-serif !important;
        box-shadow: var(--shadow-sm);
    }

    /* Assistant bubble — left-aligned white */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stMarkdownContainer"] {
        background: var(--bubble-ai) !important;
        border: 1px solid var(--border) !important;
        border-radius: 5px 18px 18px 18px !important;
        padding: 0.75rem 1.1rem !important;
        color: var(--text-main) !important;
        font-size: 0.90rem;
        line-height: 1.7;
        font-family: 'DM Sans', sans-serif !important;
        box-shadow: var(--shadow-sm);
    }

    /* Avatar overrides */
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #C4956A, #8B6355) !important;
        border-radius: 50% !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: var(--icon-bg) !important;
        border-radius: 50% !important;
    }

    /* Code blocks */
    [data-testid="stChatMessage"] code,
    [data-testid="stChatMessage"] pre {
        font-family: 'DM Mono', 'Fira Code', monospace !important;
        font-size: 0.82rem !important;
        background: rgba(139,99,85,0.06) !important;
        border-radius: 6px;
        padding: 2px 6px;
    }

    /* ── Suggestion chips ── */
    .chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.6rem;
        margin-left: 3rem;
    }
    .chip {
        display: inline-block;
        padding: 5px 14px;
        border-radius: var(--radius-pill);
        border: 1px solid var(--border);
        background: var(--bg-card);
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.15s, box-shadow 0.15s;
    }
    .chip:hover {
        background: var(--beige);
        box-shadow: var(--shadow-sm);
    }

    /* ── Input bar ── */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 720px;
        max-width: calc(100% - 2rem);

        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        z-index: 999;
    }
    [data-testid="stChatInput"]:focus-within > div {
        border-color: rgba(139,99,85,0.3);
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--text-soft) !important; }
    [data-testid="stChatInputSubmitButton"] svg { stroke: var(--accent) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(139,99,85,0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139,99,85,0.28); }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--accent) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="position:relative;">
        <div class="serene-header">
                <div class="brand-wrapper">
                    <div class="logo-circle">🌿</div>
                    <span class="brand">Serene AI</span>
                    <span class="model-tag">{MODEL_NAME}</span>
                </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load model & tokenizer once ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Warming up…")
def load_resources():
    get_tokenizer()
    get_model()

load_resources()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Empty / Welcome state ─────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-hero">
            <h2>How are you feeling today?</h2>
            <p>This is your calm, judgement-free space. Share what's on your mind — I'm here to listen and support you.</p>
        </div>

        <div class="session-grid">
            <div class="session-card sage">
                <div class="sc-icon">🧘</div>
                <div class="sc-title">Mindfulness Exercise</div>
                <div class="sc-desc">Gentle breathing and grounding practices for this moment.</div>
            </div>
            <div class="session-card lav">
                <div class="sc-icon">📓</div>
                <div class="sc-title">Journal Prompt</div>
                <div class="sc-desc">Reflective questions to help you explore your thoughts.</div>
            </div>
            <div class="session-card beige">
                <div class="sc-icon">🌊</div>
                <div class="sc-title">Stress Relief</div>
                <div class="sc-desc">Simple techniques to ease tension and find calm.</div>
            </div>
        </div>

        <div class="hero-image">
            <span class="hero-text">"Every breath is a new beginning."</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render chat history ───────────────────────────────────────────────────────
else:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Input & response ──────────────────────────────────────────────────────────
if user_input := st.chat_input("Share what's on your mind…"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    history = st.session_state.messages[:-1]
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in model.stream_response(
            user_input=prompt,
            conversation_history=st.session_state.messages,
            max_new_tokens=256,
        ):
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})