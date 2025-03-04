import streamlit as st
from summary import process_input
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def main():
    st.set_page_config(layout="centered")

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
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                white-space: pre-wrap;
                margin-top: 0;
            }
            .stTextArea::placeholder {
                color: #666;
                opacity: 0.8;
            }
            [data-testid="stMarkdownContainer"] {
                margin: 0;
                padding: 0;
            }
            .stMarkdown p {
                font-size: 16px;
                color: #666;
                margin-top: 5px;
                margin-bottom: 15px;
                font-style: italic;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
        url = st.text_input("Enter the URL of a PDF:")
        if url:
            input_data = url
            identifier = url

    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = ""

    st.markdown('<div class="custom-prompt-box">', unsafe_allow_html=True)
    label_col, button_col = st.columns([0.7, 0.3])
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
    )
    st.markdown("</div>", unsafe_allow_html=True)

    model_options = ["OpenAI (GPT-4o-mini)", "TogetherAI (LLaMA)"]
    model = st.selectbox("Select model:", model_options)
    model_key = "openai" if "OpenAI" in model else "togetherai"

    if st.button("Summarize"):
        if input_data:
            if not custom_prompt or custom_prompt.strip() == "":
                st.error("Please enter a prompt or click 'Sample Prompt' to generate a summary.")
            else:
                with st.spinner("Processing..."):
                    result = process_input(input_data, model=model_key, custom_prompt=custom_prompt)
                    if "error" in result:
                        st.warning(f"Failed to process: {result['error']}")
                    else:
                        st.success("Summarization complete!")
                        st.session_state.all_summaries = st.session_state.get("all_summaries", {})
                        st.session_state.all_summaries[identifier] = result
                        st.session_state.last_processed = identifier
                        display_summary(result, identifier, use_typewriter=True)
        else:
            st.error("Please provide an input (PDF or URL).")

    if "all_summaries" in st.session_state and st.session_state.all_summaries:
        st.subheader("Previous Summaries")
        for iden, summary in st.session_state.all_summaries.items():
            if iden != st.session_state.get("last_processed", ""):
                display_summary(summary, iden)


if __name__ == "__main__":
    main()