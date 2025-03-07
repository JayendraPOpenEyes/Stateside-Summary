import streamlit as st
from summary import process_input
import logging
import time
import firebase_admin
from firebase_admin import credentials, firestore, auth
from firebase_admin.auth import EmailAlreadyExistsError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)
# Use 'database_id' to specify the stateside-summary database
db = firestore.client(database_id="statside-summary")


def typewriter_effect(text, placeholder, delay=0.005):
    """Simulate a typewriter effect by displaying text character by character."""
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(display_text)
        time.sleep(delay)


def display_summary(summary, identifier, use_typewriter=False):
    """Display the summary in a structured format."""
    with st.expander(f"Summary for {identifier}", expanded=True):
        st.subheader("Summary")
        if "Error" in summary["summary"]:
            st.warning(f"Could not generate summary: {summary['summary']}")
        elif use_typewriter:
            placeholder = st.empty()
            typewriter_effect(summary["summary"], placeholder)
        else:
            st.markdown(summary["summary"])
        st.write("---")


def login():
    """Handle user login and registration."""
    st.sidebar.subheader("Login / Register")
    email = st.sidebar.text_input("Email", placeholder="Enter email")
    password = st.sidebar.text_input("Password", type="password", placeholder="Enter password")
    display_name = st.sidebar.text_input("Display Name", placeholder="Enter your name")  # New field for registration
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Login"):
            try:
                user = auth.get_user_by_email(email)
                st.session_state["user"] = user.uid
                st.session_state["display_name"] = db.collection("users").document(user.uid).get().to_dict().get("display_name", email.split("@")[0])
                st.success("Logged in successfully!")
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
    with col2:
        if st.button("Register"):
            try:
                user = auth.create_user(email=email, password=password)
                db.collection("users").document(user.uid).set({"email": email, "display_name": display_name or email.split("@")[0]})
                st.session_state["user"] = user.uid
                st.session_state["display_name"] = display_name or email.split("@")[0]
                st.success("Registered and logged in successfully!")
            except EmailAlreadyExistsError:
                st.error("Email already exists.")
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")


def logout():
    """Handle user logout."""
    if st.sidebar.button("Logout"):
        del st.session_state["user"]
        del st.session_state["display_name"]
        st.success("Logged out successfully!")


def main():
    st.set_page_config(layout="centered")

    # CSS styling (unchanged)
    st.markdown(
        """
        <style>
            .stFileUploader {
                margin-top: 0 !important;
                margin-bottom: 0 !important;
            }
            .stAlert {
                margin-top: 0 !important;
                margin-bottom: 0 !important;
            }
            .custom-prompt-box {
                margin-top: 0 !important;
                margin-bottom: 0 !important;
                padding: 0 !important;
            }
            .title-container {
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 10px;
            }
            .logo-img {
                border-radius: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                max-width: 100%;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 12px 12px;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
            .stSelectbox, .stTextInput, .stFileUploader, .stTextArea {
                border-radius: 8px;
                padding: 10px;
            }
            .stFileUploader label {
                color: #333;
                font-weight: normal;
            }
            .stFileUploader .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            .stFileUploader .stButton>button:hover {
                background-color: #45a049;
            }
            .stSuccess {
                background-color: #e6ffe6;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 0;
            }
            .stTextArea {
                border: 1px solid #e0e0e0;
                background-color: #f9f9f9;
                border-radius: 3px;
                padding: 1px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                white-space: pre-wrap;
                margin-top: 0;
                margin-bottom: 0;
            }
            .stTextArea::placeholder {
                color: #666;
                opacity: 0.8;
            }
            [data-testid="stMarkdownContainer"] {
                margin: 0 !important;
                padding: 0 !important;
            }
            .stMarkdown p {
                font-size: 16px;
                color: #666;
                margin-top: 5px;
                margin-bottom: 15px;
                font-style: italic;
            }
            [data-testid="stWidgetLabel"] {
                display: none !important;
            }
            .custom-prompt-box .stTextArea {
                margin: 0 !important;
                padding: 0 !important;
            }
            .custom-prompt-box .stColumns > div {
                padding: 0 !important;
                margin: 0 !important;
            }
            .custom-prompt-box .stMarkdown h4 {
                margin: 0 !important;
                padding: 0 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Authentication handling
    if "user" not in st.session_state:
        login()
        return
    else:
        display_name = st.session_state.get("display_name", auth.get_user(st.session_state["user"]).email.split("@")[0])
        st.sidebar.write(f"Logged in as: {display_name} ({auth.get_user(st.session_state['user']).email})")
        logout()

    # UI Layout
    col1, col2 = st.columns([1.5, 4.5])
    with col1:
        st.image("logo.jpg", width=100, output_format="PNG", use_container_width=True)
    with col2:
        st.title("Bill Summarization")

    st.markdown(
        "Select a Stateside bill type, enter a URL, or upload a PDF file to generate a summary."
    )

    input_type = st.selectbox("Select input type:", ["Upload PDF", "Enter URL"])
    input_data = None
    identifier = ""
    if input_type == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_pdf:
            input_data = uploaded_pdf
            identifier = uploaded_pdf.name
            st.success("PDF uploaded successfully!")
    else:
        url = st.text_input("Enter the URL of a PDF:", placeholder="Enter bill URL")
        if url:
            input_data = url
            identifier = url

    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = ""

    # Custom Prompt Section
    st.markdown('<div class="custom-prompt-box">', unsafe_allow_html=True)
    label_col, spacer_col, button_col = st.columns([6.5, 1, 2])
    with label_col:
        st.markdown("#### Enter your custom prompt:")
    with button_col:
        sample_prompt = """
            Summarize the provided legislative document, focusing only on new provisions introduced by this bill.
            Start each summary with 'This measure...'.
            The summary must be at least one paragraph long (minimum 4-6 sentences) and no longer than a full page,
            detailing key changes such as definitions, rules, or exemptions, without including opinions, current laws,
            or repetitive statements.
            Do not add a title, introduction, or conclusion (e.g., 'in summary'); the entire text should be the summary.
            If the document specifies an effective date, end the summary with 'This measure has an effective date of: [date]',
            using the exact date provided; otherwise, do not include any effective date statement.
        """
        sample_prompt = "\n".join(
            line.strip() for line in sample_prompt.splitlines() if line.strip()
        )
        if st.button("Sample Prompt", key="sample_prompt", help="Click to insert a sample prompt"):
            st.session_state.custom_prompt = sample_prompt

    custom_prompt = st.text_area(
        "",
        height=150,
        placeholder="Please enter a prompt or click 'Sample Prompt' to use the default.",
        value=st.session_state.custom_prompt,
        label_visibility="hidden",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    model_options = ["OpenAI (GPT-4o-mini)", "TogetherAI (LLaMA)"]
    model = st.selectbox("Select model:", model_options)
    model_key = "openai" if "OpenAI" in model else "togetherai"

    # Summarize Button Logic
    if st.button("Summarize"):
        if input_data:
            if not custom_prompt or custom_prompt.strip() == "":
                st.error("Please enter a prompt or click 'Sample Prompt' to generate a summary.")
            else:
                with st.spinner("Processing..."):
                    result = process_input(input_data, model=model_key, custom_prompt=custom_prompt, user_id=st.session_state["user"], display_name=st.session_state.get("display_name", "Unknown"))
                    if "error" in result:
                        st.warning(f"Failed to process: {result['error']}")
                    else:
                        st.success("Summarization complete!")
                        st.session_state["last_processed"] = identifier
                        display_summary(result, identifier, use_typewriter=True)
        else:
            st.error("Please provide an input (PDF or URL).")

    # Display Previous Summaries from Firestore
    st.subheader("Previous Summaries")
    user_id = st.session_state["user"]
    summaries_ref = db.collection("users").document(user_id).collection("summaries")
    docs = summaries_ref.stream()
    for doc in docs:
        summary_data = doc.to_dict()
        identifier = summary_data["identifier"]
        model_used = summary_data["model"]
        display_name = summary_data.get("display_name", "Unknown")
        # Use display_name as the primary identifier in the UI
        unique_identifier = f"{display_name}'s summary for {identifier} ({model_used})"
        if identifier != st.session_state.get("last_processed", ""):
            display_summary({"summary": summary_data["summary"]}, unique_identifier)


if __name__ == "__main__":
    main()