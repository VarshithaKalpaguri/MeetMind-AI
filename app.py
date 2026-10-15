import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# Optional integrations are imported only when needed so the page can still
# display a helpful error if a package is missing.
APP_TITLE = "MeetMind AI"
DB_DIR = Path("data")
DB_PATH = DB_DIR / "meetings.db"
DEFAULT_MODEL = "gemma3:1b"


# ----------------------------- Database ---------------------------------

def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                transcript TEXT NOT NULL,
                analysis_json TEXT NOT NULL
            )
            """
        )


def save_meeting(title, transcript, analysis):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO meetings (title, created_at, transcript, analysis_json)
            VALUES (?, ?, ?, ?)
            """,
            (title, created_at, transcript, json.dumps(analysis, ensure_ascii=False)),
        )
        return cursor.lastrowid


def list_meetings():
    with get_connection() as connection:
        return connection.execute(
            "SELECT id, title, created_at FROM meetings ORDER BY id DESC"
        ).fetchall()


def load_meeting(meeting_id):
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()


def delete_meeting(meeting_id):
    with get_connection() as connection:
        connection.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))


# ----------------------------- AI helpers -------------------------------

def get_ollama_response(transcript, model):
    import ollama

    prompt = f"""
You are MeetMind AI, an accurate meeting-minutes assistant.
Analyze the transcript below. Do not invent names, decisions, owners, or deadlines.
If an owner or deadline is not explicitly stated, use "Not specified".
Return ONLY valid JSON, without Markdown fences, using exactly this structure:
{{
  "summary": ["short summary point", "short summary point"],
  "key_points": ["important discussion point"],
  "decisions": ["decision explicitly made"],
  "action_items": [
    {{
      "task": "specific task",
      "owner": "person's name or Not specified",
      "deadline": "deadline or Not specified",
      "status": "Pending"
    }}
  ],
  "follow_up_questions": ["unresolved question, if any"]
}}
Use empty arrays when a section has no supported information.
Keep the summary concise and preserve important details.

MEETING TRANSCRIPT:
{transcript}
"""
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce factual meeting minutes. Never fabricate details. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        format="json",
    )
    content = response.message.content.strip()

    # Be tolerant of models that still wrap JSON in code fences.
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()

    analysis = json.loads(content)
    for key in ("summary", "key_points", "decisions", "action_items", "follow_up_questions"):
        if not isinstance(analysis.get(key), list):
            analysis[key] = []
    return analysis


def transcribe_audio(uploaded_file, model_size="base"):
    from faster_whisper import WhisperModel

    suffix = Path(uploaded_file.name).suffix or ".audio"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = whisper.transcribe(temp_path, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, getattr(info, "language", None)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ----------------------------- Display ----------------------------------

def display_analysis(analysis):
    st.subheader("📝 Meeting summary")
    summary = analysis.get("summary", [])
    if summary:
        for item in summary:
            st.markdown(f"- {item}")
    else:
        st.info("No summary points were returned.")

    left, right = st.columns(2)
    with left:
        st.subheader("🔑 Key discussion points")
        points = analysis.get("key_points", [])
        if points:
            for item in points:
                st.markdown(f"- {item}")
        else:
            st.caption("No key points identified.")

    with right:
        st.subheader("✅ Decisions")
        decisions = analysis.get("decisions", [])
        if decisions:
            for item in decisions:
                st.markdown(f"- {item}")
        else:
            st.caption("No explicit decisions identified.")

    st.subheader("📌 Action items")
    actions = analysis.get("action_items", [])
    if actions:
        st.dataframe(
            [
                {
                    "Task": item.get("task", "Not specified"),
                    "Owner": item.get("owner", "Not specified"),
                    "Deadline": item.get("deadline", "Not specified"),
                    "Status": item.get("status", "Pending"),
                }
                for item in actions
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No action items identified.")

    st.subheader("❓ Follow-up questions")
    questions = analysis.get("follow_up_questions", [])
    if questions:
        for question in questions:
            st.markdown(f"- {question}")
    else:
        st.caption("No follow-up questions identified.")


def make_minutes_markdown(title, created_at, transcript, analysis):
    lines = [
        f"# Meeting Minutes: {title}",
        "",
        f"**Created:** {created_at}",
        "",
        "## Summary",
    ]
    lines.extend([f"- {x}" for x in analysis.get("summary", [])] or ["- No summary available."])
    lines += ["", "## Key Discussion Points"]
    lines.extend([f"- {x}" for x in analysis.get("key_points", [])] or ["- None identified."])
    lines += ["", "## Decisions"]
    lines.extend([f"- {x}" for x in analysis.get("decisions", [])] or ["- None identified."])
    lines += ["", "## Action Items"]
    actions = analysis.get("action_items", [])
    if actions:
        lines += [
            "| Task | Owner | Deadline | Status |",
            "|---|---|---|---|",
        ]
        for item in actions:
            task = str(item.get("task", "Not specified")).replace("|", "\\|")
            owner = str(item.get("owner", "Not specified")).replace("|", "\\|")
            deadline = str(item.get("deadline", "Not specified")).replace("|", "\\|")
            status = str(item.get("status", "Pending")).replace("|", "\\|")
            lines.append(f"| {task} | {owner} | {deadline} | {status} |")
    else:
        lines.append("- None identified.")
    lines += ["", "## Follow-up Questions"]
    lines.extend(
        [f"- {x}" for x in analysis.get("follow_up_questions", [])]
        or ["- None identified."]
    )
    lines += ["", "## Original Transcript", "", transcript]
    return "\n".join(lines)


# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
initialize_database()

st.markdown(
    """
    <style>
      .stApp { background: #0b1220; }
      [data-testid="stSidebar"] { background: #111a29; }
      .main-title { font-size: 2.35rem; font-weight: 800; margin-bottom: 0.15rem; }
      .subtitle { color: #cbd5e1; font-size: 1rem; margin-bottom: 1.3rem; }
      div[data-testid="stMetric"] {
        background: #111a29; border: 1px solid #26354a;
        padding: 14px; border-radius: 12px;
      }
      .stButton > button { border-radius: 9px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_installed_ollama_models():
    """Return model names currently installed in the local Ollama instance."""
    try:
        import ollama
        response = ollama.list()
        raw_models = (
            response.get("models", [])
            if isinstance(response, dict)
            else getattr(response, "models", [])
        )
        model_names = []
        for item in raw_models:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
            else:
                name = (
                    getattr(item, "model", None)
                    or getattr(item, "name", None)
                    or getattr(item, "model_name", None)
                )
            if name:
                model_names.append(str(name))
        return sorted(set(model_names)), None
    except Exception as error:
        return [], str(error)


# These are choices to make visible in the selector. Models not installed locally
# must be downloaded with `ollama pull <model-name>` before they can be used.
RECOMMENDED_OLLAMA_MODELS = [
    "phi3:latest",
    "qwen2.5:3b",
    "qwen2.5:7b",
    "mistral:7b",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "gemma3:1b",
    "llama3.2:latest",
]

# Load the installed model names before the sidebar uses them.
installed_models, model_load_error = get_installed_ollama_models()

with st.sidebar:
    st.markdown("## 🎙️ MeetMind AI")
    st.caption("Smart Meeting Intelligence")

    # Show installed models first, plus popular Ollama models the user can install.
    model_options = list(dict.fromkeys(installed_models + RECOMMENDED_OLLAMA_MODELS))
    default_index = (
        model_options.index(DEFAULT_MODEL)
        if DEFAULT_MODEL in model_options
        else 0
    )
    model_name = st.selectbox(
        "Choose an AI model (Ollama)",
        options=model_options,
        index=default_index,
        help="Installed models can be used immediately. Models marked as not installed must be downloaded first.",
    )

    if model_name in installed_models:
        st.success("Selected model is installed and ready.")
    else:
        st.warning(f"{model_name} is not installed on this computer yet.")
        st.code(f"ollama pull {model_name}", language="powershell")
        st.caption("Run that command in a separate terminal, wait for the download to finish, then click Refresh model list.")

    if not installed_models and model_load_error:
        st.error("Could not retrieve the installed model list. Make sure Ollama is running.")
        with st.expander("Model-list error"):
            st.code(model_load_error)

    if st.button("🔄 Refresh model list", use_container_width=True):
        st.rerun()

    st.divider()
    st.markdown("**Workflow**")
    st.markdown("1. Upload audio or paste text")
    st.markdown("2. Transcribe if needed")
    st.markdown("3. Generate meeting minutes")
    st.markdown("4. Save and export results")
    st.divider()
    st.caption("Your meeting data is stored locally in data/meetings.db.")

st.markdown('<div class="main-title">🎙️ MeetMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Turn conversations into clear summaries, decisions, and trackable action items.</div>',
    unsafe_allow_html=True,
)

tab_new, tab_history, tab_about = st.tabs(["✨ New meeting", "🗂️ Meeting history", "ℹ️ About"])

with tab_new:
    col1, col2 = st.columns([1.05, 0.95], gap="large")

    with col1:
        st.subheader("1. Meeting details")
        meeting_title = st.text_input(
            "Meeting title",
            placeholder="e.g. Project team meeting",
        )
        input_method = st.radio(
            "Choose input type",
            ["Paste transcript / notes", "Upload audio recording"],
            horizontal=True,
        )

        transcript = ""
        detected_language = None

        if input_method == "Paste transcript / notes":
            transcript = st.text_area(
                "Meeting transcript",
                height=300,
                placeholder=(
                    "Paste the transcript here...\n\n"
                    "Example: Bhavana will complete the UI by Friday. "
                    "Varshitha will develop the login module by Sunday."
                ),
            )
        else:
            audio_file = st.file_uploader(
                "Upload meeting audio",
                type=["mp3", "wav", "m4a", "webm", "mp4", "mpeg", "mpga", "ogg", "flac"],
                help="Audio transcription runs locally with faster-whisper. The first run may download the Whisper model.",
            )
            whisper_size = st.selectbox(
                "Transcription model",
                ["tiny", "base", "small"],
                index=1,
                help="tiny is faster; base is a balanced starting point; small may be more accurate but slower.",
            )
            if audio_file is not None:
                st.audio(audio_file)
                if st.button("🎧 Transcribe audio", use_container_width=True):
                    try:
                        with st.spinner("Transcribing audio... this can take a while on CPU."):
                            transcript, detected_language = transcribe_audio(audio_file, whisper_size)
                        st.session_state["transcribed_text"] = transcript
                        st.session_state["transcribed_language"] = detected_language or "unknown"
                        if transcript:
                            st.success("Transcription complete.")
                        else:
                            st.warning("No speech was detected. Try a clearer recording.")
                    except Exception as error:
                        st.error(f"Audio transcription failed: {error}")
                        st.info(
                            "Check that faster-whisper is installed and that the audio file is supported. "
                            "The first transcription can take longer while the model downloads."
                        )

            transcript = st.text_area(
                "Transcript (you can edit this after transcription)",
                value=st.session_state.get("transcribed_text", ""),
                height=220,
                key="audio_transcript_editor",
                placeholder="Transcribed text will appear here. You can also paste text manually.",
            )
            detected_language = st.session_state.get("transcribed_language")

        if detected_language:
            st.caption(f"Detected audio language: {detected_language}")

    with col2:
        st.subheader("2. Analyze the meeting")
        st.write(
            "MeetMind AI extracts the meeting summary, key points, explicit decisions, "
            "action items, owners, deadlines, and follow-up questions."
        )
        st.info(
            "Accuracy note: AI may miss context or mishear audio. Review the generated minutes "
            "before sharing them, especially owners and deadlines."
        )
        analyze_clicked = st.button(
            "✨ Analyze meeting",
            type="primary",
            use_container_width=True,
        )

        if analyze_clicked:
            if not meeting_title.strip():
                st.warning("Please enter a meeting title.")
            elif not transcript.strip():
                st.warning("Please paste a transcript or transcribe an audio recording first.")
            else:
                try:
                    with st.spinner(f"Analyzing with {model_name}..."):
                        analysis = get_ollama_response(transcript.strip(), model_name.strip())
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    meeting_id = save_meeting(meeting_title.strip(), transcript.strip(), analysis)
                    st.session_state["latest_result"] = {
                        "id": meeting_id,
                        "title": meeting_title.strip(),
                        "created_at": created_at,
                        "transcript": transcript.strip(),
                        "analysis": analysis,
                    }
                    st.success("Meeting analyzed and saved!")
                except json.JSONDecodeError:
                    st.error("The AI returned an unreadable response. Click Analyze meeting to try again.")
                except Exception as error:
                    st.error(f"Analysis failed: {error}")
                    st.markdown(
                        "Check that Ollama is running and that the selected model is installed. "
                        "In a terminal, run `ollama list` and confirm the model name."
                    )

    result = st.session_state.get("latest_result")
    if result:
        st.divider()
        st.markdown(f"## Results: {result['title']}")
        st.caption(f"Saved on {result['created_at']} · Meeting ID {result['id']}")
        display_analysis(result["analysis"])
        markdown_minutes = make_minutes_markdown(
            result["title"], result["created_at"], result["transcript"], result["analysis"]
        )
        st.download_button(
            "⬇️ Download meeting minutes (.md)",
            data=markdown_minutes,
            file_name=f"meeting_minutes_{result['id']}.md",
            mime="text/markdown",
        )
        with st.expander("View original transcript"):
            st.text(result["transcript"])

with tab_history:
    st.subheader("Saved meetings")
    meetings = list_meetings()
    if not meetings:
        st.info("No meetings saved yet. Analyze a meeting in the New meeting tab first.")
    else:
        for meeting in meetings:
            with st.expander(f"📄 {meeting['title']} — {meeting['created_at']} (ID {meeting['id']})"):
                row = load_meeting(meeting["id"])
                if row:
                    try:
                        analysis = json.loads(row["analysis_json"])
                    except json.JSONDecodeError:
                        st.error("This saved analysis could not be read.")
                        continue
                    display_analysis(analysis)
                    markdown_minutes = make_minutes_markdown(
                        row["title"], row["created_at"], row["transcript"], analysis
                    )
                    st.download_button(
                        "⬇️ Download these minutes",
                        data=markdown_minutes,
                        file_name=f"meeting_minutes_{meeting['id']}.md",
                        mime="text/markdown",
                        key=f"download_{meeting['id']}",
                    )
                    with st.expander("View transcript"):
                        st.text(row["transcript"])
                    if st.button("Delete this meeting", key=f"delete_{meeting['id']}"):
                        delete_meeting(meeting["id"])
                        st.rerun()

with tab_about:
    st.subheader("About MeetMind AI")
    st.write(
        "MeetMind AI is a local-first smart meeting assistant for student teams and project groups. "
        "It turns meeting recordings or pasted transcripts into structured minutes."
    )
    st.markdown(
        """
        **Technology stack**
        - **Streamlit:** web interface
        - **Ollama + Gemma 3:1B:** meeting analysis
        - **faster-whisper:** audio transcription
        - **SQLite:** local meeting history

        **Privacy:** the app stores meeting text and analysis in a local SQLite database. Audio is
        processed locally by faster-whisper after its model is available. Ollama runs the selected
        language model locally. Do not upload recordings unless you have permission from the people
        recorded.

        **Current limitation:** action-item status is initially set to Pending; this version does not
        automatically sync with calendars or project-management tools.
        """
    )
