import os
import re
import io
import json
import time
import hashlib
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import requests
import pdfkit
import tiktoken
from bs4 import BeautifulSoup
from PIL import Image
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import pytesseract
import fitz
import openai

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Base directory to save all data
SAVE_DIR = "saved_data"
os.makedirs(SAVE_DIR, exist_ok=True)

class TextProcessor:
    def __init__(self, model):
        # Hardcode the model names for consistency on the backend
        if model.lower() == "openai":
            self.model = "gpt-4o-mini"  # Use gpt-4o-mini as the OpenAI model
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key or self.openai_api_key.strip() == "":
                raise ValueError("OpenAI API key is missing or empty. Set OPENAI_API_KEY in the .env file.")
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)  # Initialize OpenAI client
        elif model.lower() == "togetherai":
            self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Use a specific LLaMA model for TogetherAI
            self.together_api_key = os.getenv('TOGETHERAI_API_KEY')
            if not self.together_api_key or self.together_api_key.strip() == "":
                raise ValueError("TogetherAI API key is missing or empty. Set TOGETHERAI_API_KEY in the .env file.")
        else:
            raise ValueError(f"Unsupported model selection: {model}. Use 'openai' or 'togetherai'.")

        # Log the API keys (masked for security) to debug
        logging.info(f"OpenAI API Key: {'Set' if hasattr(self, 'openai_api_key') and self.openai_api_key else 'Not Set'}")
        logging.info(f"TogetherAI API Key: {'Set' if hasattr(self, 'together_api_key') and self.together_api_key else 'Not Set'}")

    def get_save_directory(self, base_name):
        if not base_name or not base_name.strip():
            raise ValueError("Cache name must be specified")
        folder_path = os.path.join(SAVE_DIR, base_name)
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directory {folder_path}: {str(e)}")
            raise
        return folder_path

    def get_base_name_from_link(self, link):
        parts = link.split('/')
        meaningful_parts = [part for part in parts[-4:] if part and part.lower() not in ['pdf', 'html', 'htm']]
        base_name = '_'.join(meaningful_parts) or '_'.join(parts)
        base_name = re.sub(r"\.(htm|html|pdf)$", "", base_name, flags=re.IGNORECASE)
        base_name = re.sub(r"[^\w\-_\. ]", "_", base_name)
        logging.info(f"Generated base_name: {base_name}")
        return base_name or "default_name"

    def is_google_cache_link(self, link):
        return "webcache.googleusercontent.com" in link

    def is_blank_text(self, text):
        clean_text = re.sub(r"\s+", "", text).strip()
        return len(clean_text) < 100

    def process_image_with_tesseract(self, image_path):
        try:
            return pytesseract.image_to_string(Image.open(image_path))
        except Exception as e:
            logging.error(f"Error processing image with Tesseract: {str(e)}")
            return ""

    def extract_text_from_pdf_native(self, pdf_bytes):
        """Try to extract text natively using PyMuPDF, matching the old behavior."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"  # Use simple get_text() as in the old code
            return text.strip()
        except Exception as e:
            logging.error(f"Error in native PDF extraction: {str(e)}")
            return ""

    def extract_text_from_pdf(self, pdf_content, link):
        base_name = self.get_base_name_from_link(link)
        folder = self.get_save_directory(base_name)
        pdf_bytes = pdf_content.read()
        native_text = self.extract_text_from_pdf_native(pdf_bytes)
        if native_text and not self.is_blank_text(native_text):
            logging.info("Native PDF text extraction succeeded.")
            return native_text
        images = convert_from_bytes(pdf_bytes)
        logging.info(f"OCR fallback: converting {len(images)} pages to images.")
        combined_text = ""

        def process_page(i, img):
            img_filename = f"{base_name}_page_{i+1}.png"
            img_path = os.path.join(folder, img_filename)
            try:
                img.save(img_path, 'PNG')
                logging.info(f"Saved image: {img_path}")
            except Exception as e:
                logging.error(f"Failed to save image {img_path}: {str(e)}")
                return ""
            return self.process_image_with_tesseract(img_path)

        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda x: process_page(x[0], x[1]), enumerate(images))
            for text in results:
                combined_text += text + "\n"
        if self.is_blank_text(combined_text):
            return ""
        return combined_text

    def extract_text_from_html(self, html_content):
        """Extract text from HTML content using BeautifulSoup, matching the old behavior."""
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ').strip()

    async def async_extract_text_from_url(self, url: str) -> dict:
        try:
            if self.is_google_cache_link(url):
                return {"text": "", "content_type": None, "error": "google_cache"}
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url) as response:
                        if response.status != 200:
                            return {"text": "", "content_type": None, "error": f"HTTP error {response.status}"}
                        content_type = response.headers.get('Content-Type', '').lower()
                        content = await response.read()
                        base_name = self.get_base_name_from_link(url)
                        folder = self.get_save_directory(base_name)
                        if url.lower().endswith('.pdf') or 'application/pdf' in content_type:
                            pdf_path = os.path.join(folder, f"{base_name}.pdf")
                            try:
                                with open(pdf_path, 'wb') as f:
                                    f.write(content)
                                logging.info(f"Saved PDF: {pdf_path}")
                            except Exception as e:
                                logging.error(f"Failed to save PDF {pdf_path}: {str(e)}")
                                raise
                            text = self.extract_text_from_pdf(io.BytesIO(content), url)
                            if self.is_blank_text(text):
                                return {"text": "", "content_type": "pdf", "error": "blank_pdf"}
                            return {"text": text, "content_type": "pdf", "error": None}
                        elif url.lower().endswith(('.htm', '.html')) or 'text/html' in content_type:
                            html_path = os.path.join(folder, f"{base_name}.html")
                            try:
                                with open(html_path, 'wb') as f:
                                    f.write(content)
                                logging.info(f"Saved HTML: {html_path}")
                            except Exception as e:
                                logging.error(f"Failed to save HTML {html_path}: {str(e)}")
                                raise
                            text = self.extract_text_from_html(content)
                            if self.is_blank_text(text):
                                return {"text": "", "content_type": "html", "error": "blank_html"}
                            return {"text": text, "content_type": "html", "error": None}
                        elif 'text/plain' in content_type:
                            text = content.decode('utf-8', errors='ignore').strip()
                            if self.is_blank_text(text):
                                return {"text": "", "content_type": "text", "error": "blank_text"}
                            return {"text": text, "content_type": "text", "error": None}
                        else:
                            return {"text": "", "content_type": None, "error": "unsupported_type"}
                except aiohttp.ClientError as ce:
                    logging.error(f"Network error fetching URL {url}: {str(ce)}")
                    return {"text": "", "content_type": None, "error": f"Network error: {str(ce)}"}
                except Exception as e:
                    logging.error(f"Unexpected error fetching URL {url}: {str(e)}")
                    return {"text": "", "content_type": None, "error": f"Unexpected error: {str(e)}"}
        except Exception as e:
            logging.error(f"Error in async operation for URL {url}: {str(e)}")
            return {"text": "", "content_type": None, "error": f"Async operation error: {str(e)}"}

    def process_uploaded_pdf(self, pdf_file, base_name="uploaded_pdf"):
        try:
            folder = self.get_save_directory(base_name)
            pdf_path = os.path.join(folder, f"{base_name}.pdf")
            pdf_bytes = pdf_file.read()
            if not pdf_bytes:
                return {"text": "", "content_type": "pdf", "error": "Empty PDF file"}
            try:
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_bytes)
                logging.info(f"Saved uploaded PDF: {pdf_path}")
            except Exception as e:
                logging.error(f"Failed to save PDF {pdf_path}: {str(e)}")
                raise
            
            # Step 1: Try native PDF text extraction first for speed
            native_text = self.extract_text_from_pdf_native(pdf_bytes)
            if native_text and not self.is_blank_text(native_text):
                logging.info("Native PDF text extraction succeeded.")
                return {"text": native_text, "content_type": "pdf", "error": None}
            
            # Step 2: If native extraction fails, use OCR as fallback
            logging.info("Native PDF extraction failed. Falling back to OCR.")
            images = convert_from_bytes(pdf_bytes)
            logging.info(f"OCR fallback: converting {len(images)} page(s) to images.")
            combined_text = ""

            def process_page(i, img):
                img_filename = f"{base_name}_page_{i+1}.png"
                img_path = os.path.join(folder, img_filename)
                try:
                    img.save(img_path, 'PNG')
                    logging.info(f"Saved image: {img_path}")
                except Exception as e:
                    logging.error(f"Failed to save image {img_path}: {str(e)}")
                    return ""
                return self.process_image_with_tesseract(img_path)

            with ThreadPoolExecutor() as executor:
                results = executor.map(lambda x: process_page(x[0], x[1]), enumerate(images))
                for text in results:
                    combined_text += text + "\n"
            if self.is_blank_text(combined_text):
                return {"text": "", "content_type": "pdf", "error": "blank_pdf"}
            return {"text": combined_text, "content_type": "pdf", "error": None}
        except Exception as e:
            logging.error(f"Error processing uploaded PDF: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def process_uploaded_html(self, html_file, base_name="uploaded_html"):
        try:
            folder = self.get_save_directory(base_name)
            html_path = os.path.join(folder, f"{base_name}.html")
            html_bytes = html_file.read()
            if not html_bytes:
                return {"text": "", "content_type": "html", "error": "Empty HTML file"}
            try:
                with open(html_path, 'wb') as f:
                    f.write(html_bytes)
                logging.info(f"Saved uploaded HTML: {html_path}")
            except Exception as e:
                logging.error(f"Failed to save HTML {html_path}: {str(e)}")
                raise
            text = self.extract_text_from_html(html_bytes)
            if self.is_blank_text(text):
                return {"text": "", "content_type": "html", "error": "blank_html"}
            return {"text": text, "content_type": "html", "error": None}
        except Exception as e:
            logging.error(f"Error processing uploaded HTML: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def preprocess_text(self, text):
        text = re.sub(r"[\r\n]{2,}", "\n", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def generate_structured_json(self, text):
        """Streamlined generation of JSON structure directly from text, matching old behavior."""
        paragraphs = text.split('\n')
        json_data = {"h1": [], "p": []}
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para.split()) > 10:
                json_data["p"].append(para)
            else:
                json_data["h1"].append(para)
        return json_data

    def process_full_text_to_json(self, text, base_name):
        json_data = self.generate_structured_json(text)
        base_folder = self.get_save_directory(base_name)
        json_path = os.path.join(base_folder, f"{base_name}.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, indent=4)
            logging.info(f"Saved JSON: {json_path}")
        except Exception as e:
            logging.error(f"Error saving JSON: {str(e)}")
        return text  # Return text instead of JSON for consistency with summary generation

    def truncate_text(self, text, max_tokens=3000):
        encoding = tiktoken.get_encoding("gpt2")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return encoding.decode(tokens)

    def generate_summary_openai(self, text, custom_prompt):
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is not set.")
        if not custom_prompt or custom_prompt.strip() == "":
            raise ValueError("Please enter a prompt or generate it through the sample prompt in the web app.")
        text = self.truncate_text(text, max_tokens=4000)
        prompt = custom_prompt + "\n\nText to summarize:\n" + text
        
        logging.info(f"Sending prompt to OpenAI: {prompt[:100]}...")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500,
            )
            summary = response.choices[0].message.content.strip()
            logging.info(f"Received summary: {summary[:100]}...")
            return {"summary": summary}
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return {"summary": f"Error generating summary: {str(e)}"}

    def generate_summary_togetherai(self, text, custom_prompt):
        if not self.together_api_key:
            raise ValueError("TogetherAI API key is not set.")
        if not custom_prompt or custom_prompt.strip() == "":
            raise ValueError("Please enter a prompt or generate it through the sample prompt in the web app.")
        text = self.truncate_text(text, max_tokens=4000)
        prompt = custom_prompt + "\n\nText to summarize:\n" + text
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 1500,
            }
            response = requests.post("https://api.together.ai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"].strip()
            logging.info(f"Received TogetherAI summary: {summary[:100]}...")
            return {"summary": summary}
        except Exception as e:
            logging.error(f"Error generating TogetherAI summary: {str(e)}")
            return {"summary": f"Error: {str(e)}"}

    def get_hash(self, text, custom_prompt):
        # Combine text, custom_prompt, and model for the hash
        combined_input = f"{text}{custom_prompt}{self.model}"
        return hashlib.md5(combined_input.encode('utf-8')).hexdigest()

    def get_cache_file_path(self, base_name, text_hash):
        folder = self.get_save_directory(base_name)
        return os.path.join(folder, f"{base_name}_{text_hash}_cache.json")

    def get_cached_summary(self, text, base_name, custom_prompt, cache_expiry=3600):
        text_hash = self.get_hash(text, custom_prompt)
        cache_file = self.get_cache_file_path(base_name, text_hash)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                if time.time() - cache.get("timestamp", 0) < cache_expiry and cache.get("text_hash") == text_hash:
                    logging.info(f"Returning cached summary from {cache_file}")
                    return {"summary": cache.get("summary")}
            except Exception as e:
                logging.error(f"Error reading cache file {cache_file}: {str(e)}")
        return None

    def update_cached_summary(self, text, summary, base_name, custom_prompt):
        text_hash = self.get_hash(text, custom_prompt)
        cache_file = self.get_cache_file_path(base_name, text_hash)
        cache = {
            "text_hash": text_hash,
            "summary": summary["summary"],
            "timestamp": time.time(),
            "model": self.model
        }
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
            logging.info(f"Cache updated at {cache_file}")
        except Exception as e:
            logging.error(f"Error writing cache file {cache_file}: {str(e)}")

    def generate_summary(self, text, base_name, custom_prompt):
        if not custom_prompt or custom_prompt.strip() == "":
            raise ValueError("Please enter a prompt or generate it through the sample prompt in the web app.")
        
        cached_summary = self.get_cached_summary(text, base_name, custom_prompt)
        if cached_summary:
            return cached_summary

        if "gpt" in self.model.lower():
            summary = self.generate_summary_openai(text, custom_prompt)
        else:
            summary = self.generate_summary_togetherai(text, custom_prompt)

        if "Error" not in summary["summary"]:
            effective_date_pattern = r"(effective\s+(?:date\s*(?:is|of|:)?|on)|takes\s+effect\s+(?:on)?)\s*[:\s]*(?:(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?[,\s]+\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})"
            match = re.search(effective_date_pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(0).split(":", 1)[-1].strip() if ":" in match.group(0) else match.group(0).split("on", 1)[-1].strip() if "on" in match.group(0) else match.group(0).split("effect", 1)[-1].strip()
                summary["summary"] += f"\nThis measure has an effective date of: {date_str}"

        self.update_cached_summary(text, summary, base_name, custom_prompt)
        self.process_full_text_to_json(text, base_name)
        return summary

def process_input(input_data, model, custom_prompt):
    try:
        processor = TextProcessor(model=model)
        if hasattr(input_data, "read"):
            base_name = input_data.name if hasattr(input_data, "name") else "uploaded_file"
            logging.info(f"Processing uploaded file: {base_name}")
            _, ext = os.path.splitext(base_name)
            ext = ext.lower()
            if ext == ".pdf":
                result = processor.process_uploaded_pdf(input_data, base_name=base_name)
            elif ext in [".htm", ".html"]:
                result = processor.process_uploaded_html(input_data, base_name=base_name)
            else:
                return {"error": "Unsupported file type. Please upload a PDF or HTML file.", "model": model}
            if result["error"]:
                return {"error": result["error"], "model": model}
            clean_text = processor.preprocess_text(result["text"])
        elif isinstance(input_data, str) and input_data.startswith(("http://", "https://")):
            result = asyncio.run(processor.async_extract_text_from_url(input_data))
            if result["error"]:
                return {"error": result["error"], "model": model}
            clean_text = processor.preprocess_text(result["text"])
            base_name = processor.get_base_name_from_link(input_data)
        else:
            return {"error": "Invalid input type. Expected URL, PDF, or HTML file.", "model": model}

        summary = processor.generate_summary(clean_text, base_name, custom_prompt)
        return {"model": model, "summary": summary["summary"]}
    except ValueError as ve:
        logging.error(f"ValueError processing input: {str(ve)}")
        return {"error": str(ve), "model": model}
    except Exception as e:
        logging.error(f"Error processing input: {str(e)}")
        return {"error": f"An error occurred: {str(e)}", "model": model}