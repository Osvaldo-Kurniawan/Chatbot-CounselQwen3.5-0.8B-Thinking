"""
streamlit_app.py
Signal Insight — contextual biosignal notification interface.
Model inference  →  model.py
Tokenization     →  tokenizer.py
"""

import streamlit as st
from model import get_model, stream_response
from tokenizer import MODEL_NAME, get_tokenizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Signal Insight",
    page_icon="📡",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

    <style>
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

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 7rem !important;
        max-width: 720px;
        margin: 0 auto;
    }

    /* ── Header ── */
    .signal-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        padding: 1.4rem 0 1.1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.25rem;
    }

    .signal-header .brand-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .signal-header .logo-circle {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(
            135deg,
            #C4956A 0%,
            #8B6355 55%,
            #4A5E57 100%
        );
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 3px 10px rgba(139,99,85,0.25);
        flex-shrink: 0;
    }

    .signal-header .brand {
        font-family: 'Lora', serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--accent);
        letter-spacing: -0.01em;
    }

    .signal-header .model-tag {
        font-size: 0.68rem;
        color: var(--text-soft);
        background: var(--beige);
        border: 1px solid var(--border);
        border-radius: var(--radius-pill);
        padding: 3px 10px;
        letter-spacing: 0.04em;
        margin-top: 4px;
    }

    /* ── Welcome ── */
    .welcome-hero {
        text-align: center;
        padding: 4rem 1rem 2rem;
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
        max-width: 520px;
    }

    /* ── Hero Image ── */
    .hero-image {
        width: 100%;
        height: 180px;
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
        background:
            radial-gradient(circle at 20% 30%, rgba(255,255,255,0.12), transparent 30%),
            radial-gradient(circle at 80% 70%, rgba(255,255,255,0.08), transparent 35%);
    }

    .hero-image .hero-text {
        position: relative;
        z-index: 1;
        font-family: 'Lora', serif;
        font-style: italic;
        font-size: 1.05rem;
        color: rgba(255,255,255,0.92);
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
        letter-spacing: 0.01em;
    }

    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.2rem 0 !important;
        border: none !important;
        box-shadow: none !important;
        display: flex !important;
        justify-content: flex-start !important;
    }

    /* User Bubble */
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

    /* Assistant Bubble */
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

    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #C4956A, #8B6355) !important;
        border-radius: 50% !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background: var(--icon-bg) !important;
        border-radius: 50% !important;
    }

    /* ── Input ── */
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

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-soft) !important;
    }

    [data-testid="stChatInputSubmitButton"] svg {
        stroke: var(--accent) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 4px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(139,99,85,0.15);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139,99,85,0.28);
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="position:relative;">
        <div class="signal-header">
            <div class="brand-wrapper">
                <div class="logo-circle">📡</div>
                <span class="brand">Signal Insight</span>
                <span class="model-tag">{MODEL_NAME}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load model & tokenizer once ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing model...")
def load_resources():
    get_tokenizer()
    get_model()

load_resources()

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome State ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-hero">
            <h2>Bio Signal Notification Engine</h2>
            <p>
                Analyze physiological signals and generate contextual notifications
                based on detected behavioral or biometric patterns.
            </p>
        </div>

        <div class="hero-image">
            <span class="hero-text">
                "Notice:
The deployed version of this app uses a smaller model than the original demo due to Streamlit Cloud resource limitations. The full model and architecture are still supported in the codebase and can be run locally or on a server with sufficient GPU resources."
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render Chat History ───────────────────────────────────────────────────────
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Input biosignal event or notification context..."):

    # Render user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate assistant response
    history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        full_response = st.write_stream(
            stream_response(user_input, history)
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })
