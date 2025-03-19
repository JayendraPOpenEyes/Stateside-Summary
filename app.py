import streamlit as st
from summary import process_input
import logging
import time
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import timedelta
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Firebase using Streamlit Cloud secrets or fallback credentials
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            firebase_creds = {
                "type": st.secrets["firebase"]["type"],
                "project_id": st.secrets["firebase"]["project_id"],
                "private_key_id": st.secrets["firebase"]["private_key_id"],
                "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
                "client_email": st.secrets["firebase"]["client_email"],
                "client_id": st.secrets["firebase"]["client_id"],
                "auth_uri": st.secrets["firebase"]["auth_uri"],
                "token_uri": st.secrets["firebase"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
                "universe_domain": st.secrets["firebase"]["universe_domain"]
            }
            bucket_name = f"{st.secrets['firebase']['project_id']}.appspot.com"
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
            logging.info(f"Firebase initialized with storage bucket: {bucket_name}")
        else:
            # Fallback credentials (update for production)
            bucket_name = "project-astra-438804.appspot.com"
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": "project-astra-438804",
                "private_key_id": "your-private-key-id",
                "private_key": "your-private-key",
                "client_email": "your-client-email",
                "client_id": "your-client-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "your-client-x509-cert-url",
                "universe_domain": "googleapis.com"
            })
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
            logging.warning("Streamlit secrets missing. Using hardcoded credentials (update for production).")
    except Exception as e:
        logging.error(f"Failed to initialize Firebase: {str(e)}")
        raise

# Firestore (for saving summaries) and Storage clients
db = firestore.client(database_id="statside-summary")
bucket = storage.bucket(name="project-astra-438804.appspot.com")  # Your bucket name
    
def typewriter_effect(text, placeholder, delay=0.005):
    """Simulate a typewriter effect by displaying text character by character."""
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(display_text)
        time.sleep(delay)

def display_summary(summary, identifier, use_typewriter=False):
    """Display the generated summary."""
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

def list_uploaded_files(user_id):
    """
    List PDF files for a user from Firebase Storage.
    Generates a signed URL for each file valid for 1 hour.
    Returns a dictionary with file names as keys and signed URLs as values.
    """
    prefix = f"users/{user_id}/"
    blobs = bucket.list_blobs(prefix=prefix)
    files = {}
    for blob in blobs:
        # Remove the prefix to get just the file name.
        file_name = blob.name.replace(prefix, "", 1)
        if file_name:
            # Generate a signed URL valid for 1 hour.
            file_url = blob.generate_signed_url(expiration=timedelta(hours=1))
            files[file_name] = file_url
    return files

def upload_to_storage(file, file_name, user_id):
    """
    Upload a PDF to Firebase Storage under the user's folder and
    return a signed URL valid for 1 hour.
    """
    try:
        blob = bucket.blob(f"users/{user_id}/{file_name}")
        file.seek(0)
        blob.upload_from_file(file, content_type="application/pdf")
        download_url = blob.generate_signed_url(expiration=timedelta(hours=1))
        logging.info(f"Uploaded {file_name} to Firebase Storage: {download_url}")
        return download_url
    except Exception as e:
        logging.error(f"Failed to upload {file_name} to Firebase Storage: {str(e)}")
        raise

def main():
    st.set_page_config(layout="wide")

    # CSS styling (adjust as needed)
    st.markdown(
        """
        <style>
            .stFileUploader { margin-top: 0 !important; margin-bottom: 0 !important; }
            .stAlert { margin-top: 0 !important; margin-bottom: 0 !important; }
            .custom-prompt-box { margin-top: 0 !important; margin-bottom: 0 !important; padding: 0 !important; }
            .title-container { display: flex; align-items: center; gap: 20px; margin-bottom: 10px; }
            .logo-img { border-radius: 20px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); max-width: 100%; }
            .stButton button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 12px 12px; }
            .stButton button:hover { background-color: #45a049; }
            .stSelectbox, .stTextInput, .stFileUploader, .stTextArea { border-radius: 8px; padding: 10px; }
            .stFileUploader label { color: #333; font-weight: normal; }
            .stFileUploader .stButton>button { background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 8px 16px; }
            .stFileUploader .stButton>button:hover { background-color: #45a049; }
            .stSuccess { background-color: #e6ffe6; border-radius: 5px; padding: 10px; margin-bottom: 0; }
            .stTextArea { border: 1px solid #e0e0e0; background-color: #f9f9f9; border-radius: 3px; padding: 1px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); white-space: pre-wrap; margin-top: 0; margin-bottom: 0; }
            .stTextArea::placeholder { color: #666; opacity: 0.8; }
            [data-testid="stMarkdownContainer"] { margin: 0 !important; padding: 0 !important; }
            .stMarkdown p { font-size: 16px; color: #666; margin-top: 5px; margin-bottom: 15px; font-style: italic; }
            [data-testid="stWidgetLabel"] { display: none !important; }
            .custom-prompt-box .stTextArea { margin: 0 !important; padding: 0 !important; }
            .custom-prompt-box .stColumns > div { padding: 0 !important; margin: 0 !important; }
            .custom-prompt-box .stMarkdown h4 { margin: 0 !important; padding: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Set up user and session state
    display_name = "Guest"
    user_id = "guest_user"
    st.session_state.user_id = user_id

    # Header layout
    col1, col2 = st.columns([1.5, 4.5])
    with col1:
        st.image("logo.jpg", width=100, output_format="PNG", use_container_width=True)
    with col2:
        st.title("Bill Summarization")

    st.markdown(
        "Select a Stateside bill type, enter a URL, upload a PDF, or choose a previously uploaded file to generate a summary."
    )

    # List previously uploaded PDFs from Firebase Storage
    uploaded_files = list_uploaded_files(user_id)

    # Input type selection: Upload PDF, Enter URL, or Choose previously uploaded file.
    input_type = st.selectbox("Select input type:", 
                              options=["Upload PDF", "Enter URL", "Choose file"])

    input_data = None
    identifier = ""
    file_url = None

    if input_type == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_pdf:
            identifier = uploaded_pdf.name
            # If not already in storage, use the uploaded file to generate a signed URL.
            if identifier not in uploaded_files:
                input_data = uploaded_pdf
                st.success("PDF uploaded successfully!")
            else:
                st.info(f"File '{identifier}' already exists. It will be used from your previous uploads.")
                file_url = uploaded_files.get(identifier, "")
                input_data = file_url  # Process as a URL.
    elif input_type == "Enter URL":
        url = st.text_input("Enter the URL of a PDF:", 
                            placeholder="Enter bill URL", 
                            value=st.session_state.get("selected_url", ""))
        if url:
            input_data = url
            identifier = url
    elif input_type == "Choose file":
        if uploaded_files:
            choice = st.selectbox("Select a previously uploaded file:", list(uploaded_files.keys()))
            if choice:
                identifier = choice
                file_url = uploaded_files.get(choice, "")
                input_data = file_url  # Use the stored signed URL.
                st.info(f"Selected file: {choice}")
        else:
            st.warning("No previously uploaded files available. Please upload a PDF first.")

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

    # Summarize Button Logic
    if st.button("Summarize"):
        if input_data:
            if not custom_prompt or custom_prompt.strip() == "":
                st.error("Please enter a prompt or click 'Sample Prompt' to generate a summary.")
            else:
                with st.spinner("Processing..."):
                    # If a file was uploaded, generate its signed URL.
                    if input_type == "Upload PDF" and hasattr(input_data, "read"):
                        file_url = upload_to_storage(input_data, identifier, user_id)
                        input_data.seek(0)
                    
                    # Generate summary using OpenAI (GPT-4o-mini)
                    result_openai = process_input(
                        input_data,
                        model="openai",
                        custom_prompt=custom_prompt,
                        user_id=user_id,
                        display_name=display_name,
                        file_url=file_url
                    )
                    
                    # Reset pointer if necessary.
                    if input_type == "Upload PDF" and hasattr(input_data, "seek"):
                        input_data.seek(0)
                    
                    # Generate summary using TogetherAI (LLaMA)
                    result_togetherai = process_input(
                        input_data,
                        model="togetherai",
                        custom_prompt=custom_prompt,
                        user_id=user_id,
                        display_name=display_name,
                        file_url=file_url
                    )
                    
                    # Display errors if any.
                    if "error" in result_openai:
                        st.warning(f"OpenAI failed: {result_openai['error']}")
                    if "error" in result_togetherai:
                        st.warning(f"TogetherAI failed: {result_togetherai['error']}")
                    
                    # If at least one summary succeeded, show both side by side.
                    if "error" not in result_openai or "error" not in result_togetherai:
                        st.success("Summarization complete!")
                        st.session_state["last_processed"] = identifier
                        cols = st.columns(2)
                        with cols[0]:
                            st.header("Model 1")
                            display_summary(result_openai, identifier, use_typewriter=True)
                        with cols[1]:
                            st.header("Model 2")
                            display_summary(result_togetherai, identifier, use_typewriter=True)
                        
                        # Save selected input for future sessions.
                        if input_type == "Enter URL":
                            st.session_state.selected_url = input_data
                        elif input_type == "Upload PDF":
                            st.session_state.selected_pdf = identifier
        else:
            st.error("Please provide an input (PDF, URL, or choose file).")

if __name__ == "__main__":
    main()
