# main.py
import streamlit as st
from summary import process_input
import logging
import time
import firebase_admin
from firebase_admin import credentials, firestore, storage
import json
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Firebase (unchanged)
if not firebase_admin._apps:
    # ... [Keep your existing Firebase initialization code] ...
    pass

# Firestore and Storage clients
db = firestore.client(database_id="statside-summary")
bucket = storage.bucket(name="project-astra-438804.appspot.com")

def typewriter_effect(text, placeholder, delay=0.005):
    """Simulate a typewriter effect by displaying text character by character."""
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(display_text)
        time.sleep(delay)

def display_summaries(summary_openai, summary_together, identifier, use_typewriter=False, file_url=None):
    """Display both summaries side by side with an optional download link."""
    with st.expander(f"Summaries for {identifier}", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("OpenAI (GPT-4o-mini) Summary")
            if "Error" in summary_openai["summary"]:
                st.warning(f"Could not generate OpenAI summary: {summary_openai['summary']}")
            elif use_typewriter:
                placeholder = st.empty()
                typewriter_effect(summary_openai["summary"], placeholder)
            else:
                st.markdown(summary_openai["summary"])
        with col2:
            st.subheader("TogetherAI (LLaMA) Summary")
            if "Error" in summary_together["summary"]:
                st.warning(f"Could not generate TogetherAI summary: {summary_together['summary']}")
            elif use_typewriter:
                placeholder = st.empty()
                typewriter_effect(summary_together["summary"], placeholder)
            else:
                st.markdown(summary_together["summary"])
        
        if file_url and not identifier.startswith(("http://", "https://")):
            blob = bucket.blob(f"users/{st.session_state.get('user_id', 'guest_user')}/{identifier}")
            try:
                file_bytes = blob.download_as_bytes()
                st.download_button(
                    label="Download Original PDF",
                    data=file_bytes,
                    file_name=identifier,
                    mime="application/pdf"
                )
            except Exception as e:
                st.warning(f"Could not retrieve PDF: {str(e)}")
        st.write("---")

def get_uploaded_files(user_id):
    """Fetch previously uploaded files/URLs and their details from Firestore."""
    # ... [Keep your existing get_uploaded_files function] ...
    pass

def upload_to_storage(file, file_name, user_id):
    """Upload a file to Firebase Storage and return its download URL."""
    # ... [Keep your existing upload_to_storage function] ...
    pass

def main():
    st.set_page_config(layout="wide")  # Changed to wide layout for side-by-side display

    # CSS styling (updated for side-by-side layout)
    st.markdown(
        """
        <style>
            /* ... [Keep your existing CSS] ... */
            .stColumn {
                padding: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # UI Layout
    display_name = "Guest"
    user_id = "guest_user"
    st.session_state.user_id = user_id

    col1, col2 = st.columns([1.5, 4.5])
    with col1:
        st.image("logo.jpg", width=100, output_format="PNG", use_container_width=True)
    with col2:
        st.title("Bill Summarization")

    st.markdown(
        "Select a Stateside bill type, enter a URL, or upload a PDF file to generate summaries."
    )

    # Fetch previously uploaded files/URLs with details
    uploaded_files = get_uploaded_files(user_id)
    if uploaded_files:
        st.sidebar.markdown("### Previous Uploads")
        selected_file = st.sidebar.selectbox("Select a previous file or URL:", ["New Input"] + list(uploaded_files.keys()))
        if selected_file != "New Input":
            selected_data = uploaded_files[selected_file]
            input_data = selected_data["input_data"]
            file_url = selected_data["file_url"]
            if input_data.startswith(("http://", "https://")):
                st.session_state.selected_url = input_data
                input_type_default = "Enter URL"
            else:
                st.session_state.selected_pdf = input_data
                input_type_default = "Upload PDF"
            # Display existing summaries if available
            display_summaries(
                {"summary": selected_data["summary"] if selected_data["model"] == "gpt-4o-mini" else "Not generated yet"},
                {"summary": selected_data["summary"] if selected_data["model"] != "gpt-4o-mini" else "Not generated yet"},
                selected_file,
                use_typewriter=False,
                file_url=file_url
            )
            st.session_state.custom_prompt = selected_data["custom_prompt"]
        else:
            input_type_default = "Upload PDF"
    else:
        input_type_default = "Upload PDF"

    input_type = st.selectbox("Select input type:", ["Upload PDF", "Enter URL"], index=0 if input_type_default == "Upload PDF" else 1)
    input_data = None
    identifier = ""
    file_url = None
    if input_type == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_pdf:
            if uploaded_pdf.name not in uploaded_files:
                input_data = uploaded_pdf
                identifier = uploaded_pdf.name
                st.success("PDF uploaded successfully!")
            else:
                st.info(f"File '{uploaded_pdf.name}' already exists. Select it from the sidebar.")
                identifier = uploaded_pdf.name
        elif "selected_pdf" in st.session_state and selected_file == "New Input":
            st.info(f"Selected PDF: {st.session_state.selected_pdf}")
            identifier = st.session_state.selected_pdf
    else:
        url = st.text_input("Enter the URL of a PDF:", placeholder="Enter bill URL", value=st.session_state.get("selected_url", ""))
        if url:
            input_data = url
            identifier = url

    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = ""

    # Custom Prompt Section (unchanged)
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
        if input_data or (selected_file != "New Input" and selected_file):
            if not custom_prompt or custom_prompt.strip() == "":
                st.error("Please enter a prompt or click 'Sample Prompt' to generate a summary.")
            else:
                with st.spinner("Processing..."):
                    if input_type == "Upload PDF" and hasattr(input_data, "read"):
                        file_url = upload_to_storage(input_data, identifier, user_id)
                        input_data.seek(0)
                    elif selected_file != "New Input":
                        input_data = uploaded_files[selected_file]["input_data"]
                        file_url = uploaded_files[selected_file]["file_url"]
                        identifier = selected_file

                    # Generate summaries from both models
                    result_openai = process_input(
                        input_data,
                        model="openai",
                        custom_prompt=custom_prompt,
                        user_id=user_id,
                        display_name=display_name,
                        file_url=file_url
                    )
                    result_together = process_input(
                        input_data,
                        model="togetherai",
                        custom_prompt=custom_prompt,
                        user_id=user_id,
                        display_name=display_name,
                        file_url=file_url
                    )

                    if "error" in result_openai or "error" in result_together:
                        st.warning(f"Errors: OpenAI - {result_openai.get('error', 'None')}, TogetherAI - {result_together.get('error', 'None')}")
                    else:
                        st.success("Summarization complete!")
                        st.session_state["last_processed"] = identifier
                        display_summaries(result_openai, result_together, identifier, use_typewriter=True, file_url=file_url)
                        if input_type == "Enter URL":
                            st.session_state.selected_url = input_data
                        else:
                            st.session_state.selected_pdf = identifier
        else:
            st.error("Please provide an input (PDF or URL) or select a file from the sidebar.")

if __name__ == "__main__":
    main()