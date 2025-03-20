import re
import json
import zipfile
import difflib
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import tempfile
import os
import shutil
from io import BytesIO
from odf.opendocument import load, OpenDocumentText
from odf.text import P
from PIL import Image
import streamlit as st
import google.generativeai as genai
import time
import traceback
import uuid
import pandas as pd

# # Configure Gemini API key
# genai.configure(api_key="AIzaSyCeAvrbYri0h9rz8ICyo6yWhv10M9iaGE0")  # Replace with your API key

# Initialize session state if it doesn't exist
if 'file_chunks' not in st.session_state:
    st.session_state.file_chunks = []
if 'processed_chunks' not in st.session_state:
    st.session_state.processed_chunks = {}
if 'all_images' not in st.session_state:
    st.session_state.all_images = []

############################################
# Helper Functions
############################################

def natural_keys(text):
    """Sort helper for natural sort."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split('(\d+)', text)]

def remove_noise(text):
    """
    Removes trailing noise from question descriptions,
    but preserves References section for proper extraction later.
    Also removes the redundant "QUESTION NO: X TYPE" from the beginning of the description.
    """
    if not text:
        return ""
    
    # Remove the "QUESTION NO: X TYPE" from the beginning of the description
    text = re.sub(r'^QUESTION NO:\s*\d+\s*(?:[A-Z]+\s*)?', '', text, flags=re.IGNORECASE)
    
    # Remove any <map> tag and following content
    text = re.sub(r'<map>.*$', '', text, flags=re.DOTALL)
    
    # We want to keep the References section, so we'll handle it separately
    references_match = re.search(r'(References:[\s\S]*?)(?=QUESTION NO:|$)', text, re.IGNORECASE)
    references_text = ""
    if references_match:
        references_text = references_match.group(1).strip()
    
    # Remove trailing Answer: and Explanation: sections
    text = re.sub(r'\s*(Answer:[\s\S]*?|Explanation:[\s\S]*?)(?=References:|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # If we had references, add them back
    if references_text:
        # Make sure we don't have duplicated References sections
        text = re.sub(r'\s*References:[\s\S]*$', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = text.strip() + "\n\n" + references_text
    
    return text.strip()

# def remove_noise(text):
#     """
#     Removes trailing noise from question descriptions,
#     but preserves References section for proper extraction later.
#     """
#     if not text:
#         return ""
#     # Remove any <map> tag and following content
#     text = re.sub(r'<map>.*$', '', text, flags=re.DOTALL)
    
#     # We want to keep the References section, so we'll handle it separately
#     references_match = re.search(r'(References:[\s\S]*?)(?=QUESTION NO:|$)', text, re.IGNORECASE)
#     references_text = ""
#     if references_match:
#         references_text = references_match.group(1).strip()
    
#     # Remove trailing Answer: and Explanation: sections
#     text = re.sub(r'\s*(Answer:[\s\S]*?|Explanation:[\s\S]*?)(?=References:|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    
#     # If we had references, add them back
#     if references_text:
#         # Make sure we don't have duplicated References sections
#         text = re.sub(r'\s*References:[\s\S]*$', '', text, flags=re.IGNORECASE | re.DOTALL)
#         text = text.strip() + "\n\n" + references_text
    
#     return text.strip()

def extract_references(block):
    """
    Extract references from a question block consistently.
    """
    references = []
    refs_match = re.search(r'References:([\s\S]+?)(?=QUESTION NO:|$)', block, re.IGNORECASE)
    if refs_match:
        refs_section = refs_match.group(1).strip()
        references = re.findall(r'(https?://\S+)', refs_section)
    return references

def clean_text(text):
    """
    Removes ODF markup (tags) from the text.
    """
    return re.sub(r'<[^>]+>', '', text).strip()

def is_valid_image(image_bytes):
    """Check if image data is valid."""
    try:
        Image.open(BytesIO(image_bytes)).verify()
        return True
    except:
        return False
    
############################################
# ODT Text Extraction Functions
############################################

def extract_text_from_odt(odt_file):
    """
    Uses the ODF library to extract and concatenate all paragraph texts from the ODT file.
    """
    doc = load(odt_file)
    paragraphs = doc.getElementsByType(P)
    all_text = []
    for p in paragraphs:
        para_text = ""
        for child in p.childNodes:
            if hasattr(child, "data") and child.data:
                para_text += child.data
        all_text.append(para_text)
    return "\n".join(all_text)

def split_questions(full_text):
    """
    Splits the full text into individual question blocks based on "QUESTION NO:" at the beginning of a line.
    """
    lines = full_text.splitlines()
    questions = []
    current = []
    for line in lines:
        if re.match(r'QUESTION NO:\s*\d+', line, re.IGNORECASE):
            if current:
                questions.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        questions.append("\n".join(current).strip())
    return questions

def get_question_number(block):
    """
    Extract question number from a question block.
    """
    match = re.match(r'QUESTION NO:\s*(\d+)', block, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 9999

def extract_images_from_odt(odt_bytes):
    """
    Extracts all images from the ODT file.
    """
    try:
        with zipfile.ZipFile(BytesIO(odt_bytes)) as z:
            image_files = sorted([f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                                key=natural_keys)
            images = [z.read(f) for f in image_files]
            return images, image_files
    except Exception as e:
        st.error(f"Error extracting images from ODT file: {str(e)}")
        return [], []

def split_odt_into_chunks(odt_file, batch_size= 100):
    """
    Split a large ODT file into text chunks and image sets, each containing batch_size questions.
    """
    # Extract all text
    full_text = extract_text_from_odt(odt_file)
    question_blocks = split_questions(full_text)
    
    # Sort by question number
    question_blocks.sort(key=get_question_number)
    
    # Extract all images
    odt_file.seek(0)
    odt_bytes = odt_file.read()
    images, image_files = extract_images_from_odt(odt_bytes)
    
    # Split into batches
    batches = []
    for i in range(0, len(question_blocks), batch_size):
        end = min(i + batch_size, len(question_blocks))
        batches.append(question_blocks[i:end])
    
    # Create chunk info for each batch
    chunks = []
    img_index_start = 0
    
    for batch_idx, batch in enumerate(batches):
        first_q = get_question_number(batch[0])
        last_q = get_question_number(batch[-1])
        
        # Count image-based questions in this batch
        img_count = 0
        for block in batch:
            question_info = analyze_question_block(block)
            if question_info and question_info["type"] in ["HOTSPOT", "DRAGDROP", "DROPDOWN"]:
                img_count += 1
        
        # Calculate image indices for this batch
        image_indices = []
        for i in range(img_index_start, img_index_start + img_count):
            start_idx = i * 2
            end_idx = start_idx + 2
            if start_idx < len(images) and end_idx <= len(images):
                image_indices.extend([start_idx, end_idx - 1])
        
        # Create chunk info
        chunk_id = str(uuid.uuid4())
        chunk = {
            "id": chunk_id,
            "batch_idx": batch_idx + 1,
            "first_q": first_q,
            "last_q": last_q,
            "text": "\n\n".join(batch),
            "question_count": len(batch),
            "image_count": img_count * 2,  # Two images per question (Q&A)
            "image_indices": image_indices,
            "processed": False
        }
        
        chunks.append(chunk)
        
        # Update image index for next batch
        img_index_start += img_count
    
    return chunks, images

############################################
# Question Analysis Functions
############################################

def analyze_question_block(block):
    """
    Analyzes a question block to determine its type reliably.
    Returns a dictionary with number and type keys.
    """
    # Extract the question number and base type
    # header = re.match(r'QUESTION NO:\s*(\d+)(?:\s+([\w\-]+))?', block, re.IGNORECASE)
    header = re.match(r'QUESTION NO:\s*(\d+)\s*(\w+)', block, re.IGNORECASE)

    if not header:
        return None
        
    q_number = header.group(1)
    declared_type = header.group(2) or ""
    
    # FIRST PRIORITY: Explicitly declared type takes precedence if present
    if declared_type and not declared_type.isdigit():
        if declared_type.upper() in ["HOTSPOT", "DRAGDROP", "DROPDOWN", "RADIOBUTTON", "MULTIPLECHOICE"]:
            final_type = declared_type.upper()
            return {
                "number": q_number,
                "type": final_type
            }
    
    # SECOND PRIORITY: Clear option markers and answer line indicate a text-based question
    options = re.findall(r'^[A-Z]\.\s+', block, re.MULTILINE)
    answer_line = re.search(r'Answer:\s*([A-Z](?:,\s*[A-Z])*)', block)
    
    if options and answer_line:
        answer_text = answer_line.group(1)
        answers = [a.strip() for a in answer_text.split(',') if a.strip()]
        if len(answers) > 1:
            final_type = "MULTIPLECHOICE"
        else:
            final_type = "RADIOBUTTON"
        return {
            "number": q_number,
            "type": final_type
        }
    
    # THIRD PRIORITY: Keyword detection for image-based questions
    
    # DRAGDROP detection
    dragdrop_patterns = [
        r'\bdrag\b.*\bdrop\b', 
        r'\bmatch\b.*\bitem\b',
        r'\bdrag\b.*\bappropriate\b',
        r'\bmatch\b\s+\beach\b',
        r'\bdrag\b\s+\bthem\b',
        r'\bcorrect\b\s+\bmatch\b'
    ]
    
    is_dragdrop = any(re.search(pattern, block, re.IGNORECASE) for pattern in dragdrop_patterns)
    
    # HOTSPOT detection
    hotspot_patterns = [
        r'\bhot\s*spot\b',
        r'\bselect\b.*\byes\b.*\bif\b',
        r'\bselect\b.*\bno\b.*\bif\b',
        r'\bselect\b.*\bstatement\b.*\btrue\b'
    ]
    
    is_hotspot = any(re.search(pattern, block, re.IGNORECASE) for pattern in hotspot_patterns)
    
    # DROPDOWN detection
    dropdown_patterns = [
        r'\bdrop\s*down\b',
        r'\bfrom\b.*\bdrop-?down\b',
        r'\bselect\b.*\bfrom\b.*\bmenu\b', 
        r'\bchoose\b.*\bfrom\b.*\bdropdown\b',
        r'\bcorrectly completes the sentence\b',  # example pattern
        r'\bselect the appropriate option in the answer area\b',
        r'\bselect the appropriate options? in the answer area\b'

    ]
    
    is_dropdown = any(re.search(pattern, block, re.IGNORECASE) for pattern in dropdown_patterns)
    
    # Final type determination based on detected patterns
    if is_dragdrop:
        final_type = "DRAGDROP"
    elif is_hotspot:
        final_type = "HOTSPOT"
    elif is_dropdown:
        final_type = "DROPDOWN"
    else:
        # Default to RADIOBUTTON if no clear pattern was detected
        final_type = "RADIOBUTTON"
    
    return {
        "number": q_number,
        "type": final_type
    }

############################################
# API Functions with Error Handling
############################################

def call_gemini_with_retry(model_name, prompt, image_data=None, max_retries=3, retry_delay=2):
    """
    Calls the Gemini API with automatic retry on failure.
    """
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            model = genai.GenerativeModel(model_name)
            if image_data:
                image_parts = [{"mime_type": "image/png", "data": image_data}]
                response = model.generate_content([prompt, image_parts[0]])
            else:
                response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            last_exception = e
            retries += 1
            time.sleep(retry_delay * retries)  # Exponential backoff
    
    # If we've exhausted retries, log the error and return a fallback value
    st.error(f"Error after {max_retries} retries: {str(last_exception)}")
    return "[]" if "JSON" in prompt else ""

def gemini_extract_text_from_image(image_bytes):
    """
    Uses Gemini to extract raw text from an image with retry logic.
    """
    try:
        Image.open(BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Invalid image data: {str(e)}")
        return ""
    
    prompt = """
    You are an expert in analyzing images.
    Extract and return all the text from the provided image.
    Output only the extracted text with no additional commentary.
    """
    
    return call_gemini_with_retry('gemini-2.0-flash', prompt, image_bytes)

def parse_question_options(extracted_text):
    """
    Parses the extracted text from a hotspot question image into a list of tuples,
    where each tuple is (statement, available_options).
    Assumes each statement has "Yes" and "No" as options.
    """
    lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
    # If the text begins with a header like "statement", "yes", "no", skip it.
    if len(lines) >= 3 and lines[0].lower() == "statement" and lines[1].lower() == "yes" and lines[2].lower() == "no":
        lines = lines[3:]
    combined_lines = []
    buffer = ""
    for line in lines:
        if not buffer:
            buffer = line
        else:
            if line and line[0].islower():
                buffer += " " + line
            else:
                combined_lines.append(buffer)
                buffer = line
    if buffer:
        combined_lines.append(buffer)
    # Only include statements with more than 20 characters
    statements = [l for l in combined_lines if len(l) > 20]
    return [(stmt, ["Yes", "No"]) for stmt in statements]

def gemini_extract_answers_from_image(image_bytes):
    """
    For HOTSPOT questions:
    Uses Gemini to extract answers from the answer image.
    """
    prompt = """
    You are an expert in analyzing images of multiple-choice questions.
    In the provided image, a grey-colored rectangle indicates the selected (correct) answer.
    Extract the complete text of each statement along with its selected answer.
    Output a JSON array of objects with exactly two keys:
      "statement": the full text of the statement,
      "answer": either "Yes" or "No".
    Output ONLY valid JSON with no extra commentary.
    """
    
    raw_text = call_gemini_with_retry('gemini-2.0-flash', prompt, image_bytes)
    
    # Clean up the response
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = re.sub(r'^json\s*', '', raw_text, flags=re.IGNORECASE).strip()
    
    try:
        return json.loads(raw_text)
    except Exception as e:
        st.error(f"Error parsing Gemini answer response: {str(e)}")
        st.error(f"Raw response: {raw_text}")
        return []
    
def gemini_extract_columns_dynamic(image_bytes):
    """
    For DRAGDROP questions:
    Uses Gemini to extract dynamic columns from the question image.
    """
    try:
        Image.open(BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Invalid image data: {str(e)}")
        return {"columns": []}
    
    prompt = """
    You are an expert in analyzing images of a drag-and-drop question.
    This question image has multiple columns (possibly 3 or more), each with a heading (like "Applications", "Feature", "Service", etc.).
    IMPORTANT: Extract ALL columns and their items completely, even if they appear in separate sections or different parts of the image.
    Pay special attention to ensuring you capture ALL columns shown in the image.
    
    Return the data in JSON with this structure:
    {
      "columns": [
        {"heading": "<column heading>", "items": ["item1", "item2", "..."]},
        ...
      ]
    }
    Output ONLY valid JSON, no extra commentary.
    """
    
    raw_text = call_gemini_with_retry('gemini-2.0-flash', prompt, image_bytes)
    
    # Clean up the response
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = re.sub(r'^json\s*', '', raw_text, flags=re.IGNORECASE).strip()
    
    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict) or "columns" not in data:
            return {"columns": []}
        return data
    except Exception as e:
        st.error(f"Error parsing dynamic columns JSON: {str(e)}")
        st.error(f"Raw response: {raw_text}")
        return {"columns": []}

def gemini_extract_pairs_dynamic(image_bytes):
    """
    For DRAGDROP questions:
    Uses Gemini to extract matched pairs from the answer image.
    """
    try:
        Image.open(BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Invalid image data: {str(e)}")
        return []
    
    prompt = """
    You are an expert in analyzing images of a drag-and-drop question.
    This image shows the final matched pairs (or triplets) of the columns that appeared in the question image.
    Each row includes all relevant columns, matched to the correct items.
    Output a JSON array where each object has keys corresponding to the column headings.
    Output ONLY valid JSON, no extra commentary.
    """
    
    raw_text = call_gemini_with_retry('gemini-2.0-flash', prompt, image_bytes)
    
    # Clean up the response
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = re.sub(r'^json\s*', '', raw_text, flags=re.IGNORECASE).strip()
    
    try:
        data = json.loads(raw_text)
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        st.error(f"Error parsing dynamic pairs JSON: {str(e)}")
        st.error(f"Raw response: {raw_text}")
        return []
    
def gemini_extract_dropdown_questions_from_image(image_bytes):
    """
    Uses Gemini to parse the question image for DROPDOWN questions.
    """
    try:
        Image.open(BytesIO(image_bytes))
    except Exception as e:
        st.error(f"The provided image bytes are not valid image data: {str(e)}")
        return []
    
    prompt = """
    You are an expert in analyzing an image that shows two columns with headers.
    One column (the left) contains the statement header and statement text,
    and the other column (the right) contains the options header and the dropdown options.
    For each row in the image, extract:
      - "statement_header": the header for the statement column,
      - "statement": the statement text,
      - "options_header": the header for the options column,
      - "options": an array of the dropdown options.
    Return a JSON array of objects with these four keys.
    Output ONLY valid JSON, with no extra commentary.
    """
    
    raw_text = call_gemini_with_retry('gemini-2.0-flash', prompt, image_bytes)
    
    # Clean up the response
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = re.sub(r'^json\s*', '', raw_text, flags=re.IGNORECASE).strip()
    
    start_idx = raw_text.find('[')
    end_idx = raw_text.rfind(']')
    if start_idx != -1 and end_idx != -1:
        raw_text = raw_text[start_idx:end_idx+1]
    
    try:
        data = json.loads(raw_text)
        return data
    except Exception as e:
        st.error(f"Error parsing dropdown question JSON: {str(e)}")
        st.error(f"Raw response: {raw_text}")
        return []

def gemini_extract_dropdown_answers_from_image(image_bytes):
    """
    Uses Gemini to parse the answer image for DROPDOWN questions.
    """
    try:
        Image.open(BytesIO(image_bytes))
    except Exception as e:
        st.error(f"The provided image bytes are not valid image data: {str(e)}")
        return []
    
    prompt = """
    You are an expert in analyzing an image that shows two columns with headers.
    One column contains the statement header and statement text,
    and the other column contains the answer header and the highlighted answer.
    For each row, extract:
      - "statement_header": the header for the statement column,
      - "statement": the full text of the statement,
      - "answer_header": the header for the answer column,
      - "answer": the highlighted option text.
    Return a JSON array of objects with these four keys.
    Output ONLY valid JSON, with no extra commentary.
    """
    
    raw_text = call_gemini_with_retry('gemini-2.0-flash', prompt, image_bytes)
    
    # Clean up the response
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = re.sub(r'^json\s*', '', raw_text, flags=re.IGNORECASE).strip()
    
    start_idx = raw_text.find('[')
    end_idx = raw_text.rfind(']')
    if start_idx != -1 and end_idx != -1:
        raw_text = raw_text[start_idx:end_idx+1]
    
    try:
        data = json.loads(raw_text)
        return data
    except Exception as e:
        st.error(f"Error parsing dropdown answer JSON: {str(e)}")
        st.error(f"Raw response: {raw_text}")
        return []

############################################
# Text-based Question Processing
############################################

def parse_question_text_textbased(text):
    """
    Parses a plain text question (RadioButton/MultipleChoice) into a structured dictionary.
    """
    text = clean_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    m = re.match(r'QUESTION NO:\s*(\d+)', lines[0], re.IGNORECASE)
    q_number = m.group(1) if m else "Unknown"
    desc_lines = []
    opt_start = None
    for i, line in enumerate(lines[1:], start=1):
        if re.match(r'^[A-Z]\.\s+', line):
            opt_start = i
            break
        else:
            desc_lines.append(line)
    description = "\n".join(desc_lines)
    options = []
    opt_index = opt_start if opt_start is not None else len(lines)
    while opt_index < len(lines) and re.match(r'^[A-Z]\.\s+', lines[opt_index]):
        match = re.match(r'^([A-Z])\.\s+(.*)', lines[opt_index])
        if match:
            options.append((match.group(1), match.group(2)))
        opt_index += 1
    answer_line = ""
    ans_index = None
    for i in range(opt_index, len(lines)):
        if lines[i].lower().startswith("answer:"):
            answer_line = lines[i]
            ans_index = i
            break
    correct_answers = []
    if answer_line:
        ans_text = answer_line.split(":", 1)[1]
        correct_answers = [x.strip() for x in ans_text.split(",") if x.strip()]
    explanation = ""
    expl_index = None
    if ans_index is not None:
        for i in range(ans_index+1, len(lines)):
            if lines[i].lower().startswith("explanation"):
                expl_index = i
                break
    if expl_index is not None:
        explanation = "\n".join(lines[expl_index+1:])
    ref_links = re.findall(r'(https?://\S+)', explanation)
    contents = [line for line in description.splitlines() if line]
    kind = "RadioButton" if len(correct_answers) == 1 else "MultipleChoice"
    return {
        "number": q_number,
        "id": "",
        "kind": kind,
        "display_kind": kind,
        "description": description,
        "question_contents": contents,
        "options": options,
        "correct_answers": correct_answers,
        "explanation": explanation,
        "ref_links": ref_links
    }

def build_xml_from_data_textbased(data):
    """
    Builds an XML element for a text-based question (RadioButton/MultipleChoice).
    """
    root = ET.Element("Questions")
    ET.SubElement(root, "Kind").text = data["kind"]
    ET.SubElement(root, "DisplayKind").text = data["display_kind"]
    ET.SubElement(root, "QuestionNo").text = data["number"]
    # Clean description before output
    desc = remove_noise(data["description"])
    ET.SubElement(root, "QuestionDescription").text = desc
    for letter, opt_text in data["options"]:
        choice = ET.SubElement(root, "Choices")
        ET.SubElement(choice, "Number").text = letter
        contents = ET.SubElement(choice, "Contents")
        ET.SubElement(contents, "ContentType").text = "Text"
        ET.SubElement(contents, "Text").text = opt_text
    ET.SubElement(root, "Id").text = data["id"]
    answer = ET.SubElement(root, "Answer")
    for ans in data["correct_answers"]:
        ET.SubElement(answer, "Choices").text = ans
    explanation = ET.SubElement(root, "Explanation")
    if data.get("ref_links") and data.get("correct_answers"):
        refs = ET.SubElement(explanation, "References")
        for idx, link in enumerate(data["ref_links"]):
            prefix = data["correct_answers"][idx] if idx < len(data["correct_answers"]) else chr(65+idx)
            ref = ET.SubElement(refs, f"Reference{idx+1}")
            ref.text = f"{prefix}: {link}"
    else:
        explanation.text = data["explanation"]
    return root

############################################
# Image-based Question Processing
############################################

def build_xml_for_hotspot(question, q_img, a_img):
    """
    Builds an XML element for a HOTSPOT question with robust handling for all formats.
    """
    root = ET.Element("Question")
    root.set("number", question["number"])
    root.set("type", "HOTSPOT")
    
    # Clean description but preserve references
    clean_desc = remove_noise(question["description"])
    ET.SubElement(root, "Description").text = clean_desc
    
    # Extract question text and options from question image
    q_text = gemini_extract_text_from_image(q_img)
    
    # First try the standard approach to parse question options
    standard_options = parse_question_options(q_text)
    
    # If standard parsing didn't find enough options, try direct statement extraction
    if len(standard_options) < 2:
        # Direct statement extraction approach
        direct_statements = []
        for line in q_text.splitlines():
            line = line.strip()
            # Skip empty lines, headers, or very short text
            if not line or line.lower() in ["yes", "no", "statement"] or len(line) < 15:
                continue
                
            # Avoid duplicates
            if not any(line in stmt for stmt, _ in direct_statements):
                direct_statements.append((line, ["Yes", "No"]))
        
        # Use direct statements if we found any
        if direct_statements:
            q_options = direct_statements
        else:
            q_options = standard_options
    else:
        q_options = standard_options
    
    # Try different prompts for answer extraction if initial options are found
    if q_options:
        # Create a targeted prompt based on the statements we found
        statement_list = "\n".join([f"- \"{stmt}\"" for stmt, _ in q_options])
        
        targeted_prompt = f"""
        You are analyzing an answer image for a hotspot question.
        The image shows statements with Yes/No options where some options are selected (have gray/filled circles).
        
        For each of these exact statements:
        {statement_list}
        
        Determine whether "Yes" or "No" is selected (has a gray/filled circle).
        
        Return a JSON array of objects with these exact keys:
          "statement": the exact statement text matching one from the list above,
          "answer": either "Yes" or "No" (based on which has the filled/gray circle)
        
        Output ONLY valid JSON with no extra commentary.
        """
        
        # Try the targeted prompt first
        raw_text = call_gemini_with_retry('gemini-2.0-flash', targeted_prompt, a_img)
        
        try:
            # Clean up the response
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                raw_text = re.sub(r'^json\s*', '', raw_text, flags=re.IGNORECASE).strip()
            
            a_data = json.loads(raw_text)
            
            # Validate each answer has a statement and answer
            valid_answers = []
            for item in a_data:
                if "statement" in item and "answer" in item:
                    valid_answers.append(item)
            
            # If we got valid answers, use them
            if len(valid_answers) == len(q_options):
                a_data = valid_answers
            else:
                # Try standard approach as backup
                a_data = gemini_extract_answers_from_image(a_img)
        except Exception:
            # If targeted approach failed, fall back to standard approach
            a_data = gemini_extract_answers_from_image(a_img)
    else:
        # Standard approach if we couldn't identify options
        a_data = gemini_extract_answers_from_image(a_img)
    
    # Special case: If we have very few or no options but have answer data
    if len(q_options) < len(a_data):
        # Reconstruct options from answers
        for item in a_data:
            stmt = item.get("statement", "").strip()
            if stmt and not any(stmt in opt_stmt for opt_stmt, _ in q_options):
                q_options.append((stmt, ["Yes", "No"]))
    
    # Add options to XML
    options_elem = ET.SubElement(root, "QuestionOptions")
    for idx, (stmt, opts) in enumerate(q_options, start=1):
        option_set = ET.SubElement(options_elem, "OptionSet")
        option_set.set("index", str(idx))
        ET.SubElement(option_set, "Statement").text = stmt
        opts_parent = ET.SubElement(option_set, "Options")
        for opt in opts:
            ET.SubElement(opts_parent, "Option").text = opt
    
    # Add answers to XML
    answers_elem = ET.SubElement(root, "Answers")
    
    # If we have both options and answers
    if q_options and a_data:
        # Try to match answer statements to option statements
        for item in a_data:
            ans_stmt = item.get("statement", "").strip()
            ans_value = item.get("answer", "").strip()
            
            # Try to find matching statement in options
            best_match = None
            best_match_score = 0
            
            for opt_stmt, _ in q_options:
                # Calculate similarity score - combination of common words and sequence similarity
                opt_words = set(opt_stmt.lower().split())
                ans_words = set(ans_stmt.lower().split())
                
                if opt_words and ans_words:  # Ensure non-empty
                    # Word overlap score
                    common_words = opt_words.intersection(ans_words)
                    word_score = len(common_words) / max(len(opt_words), len(ans_words))
                    
                    # Sequence matcher score
                    seq_score = difflib.SequenceMatcher(None, opt_stmt.lower(), ans_stmt.lower()).ratio()
                    
                    # Combined score (weight towards sequence score)
                    match_score = (word_score * 0.4) + (seq_score * 0.6)
                    
                    if match_score > best_match_score and match_score > 0.3:  # Lower threshold for better matching
                        best_match = opt_stmt
                        best_match_score = match_score
            
            # Use the best match if found, otherwise use the original statement
            final_statement = best_match if best_match else ans_stmt
            
            ans_elem = ET.SubElement(answers_elem, "Answer")
            ans_elem.set("statement", final_statement)
            ans_elem.text = ans_value
    else:
        # Fallback: just use the answer data directly if no matching possible
        for item in a_data:
            stmnt = item.get("statement", "").strip()
            ans = item.get("answer", "").strip()
            ans_elem = ET.SubElement(answers_elem, "Answer")
            ans_elem.set("statement", stmnt)
            ans_elem.text = ans
    
    # Final validation - ensure we have at least some options and answers
    # If for some reason we have an empty answers section but found options
    if len(list(answers_elem)) == 0 and len(list(options_elem)) > 0:
        # Create default answers based on visual analysis
        visual_answer_prompt = """
        This image shows a table with statements and Yes/No options.
        For each visible statement, tell me ONLY which option (Yes or No) appears selected.
        Typically, a selected option will have a gray or filled circle.
        Return a JSON array with one object per row, with keys:
          "row_number": the numerical position of the row (1, 2, 3, etc.)
          "selected": either "Yes" or "No" (whichever appears selected)
        Output ONLY valid JSON.
        """
        
        raw_answer_text = call_gemini_with_retry('gemini-2.0-flash', visual_answer_prompt, a_img)
        
        try:
            # Clean up response
            if raw_answer_text.startswith("```"):
                raw_answer_text = raw_answer_text.strip("`")
                raw_answer_text = re.sub(r'^json\s*', '', raw_answer_text, flags=re.IGNORECASE).strip()
                
            visual_answers = json.loads(raw_answer_text)
            
            # Match answers with options by row number
            option_sets = list(options_elem.findall("OptionSet"))
            
            for visual_ans in visual_answers:
                row_num = visual_ans.get("row_number", 0)
                selected = visual_ans.get("selected", "")
                
                if 1 <= row_num <= len(option_sets) and selected in ["Yes", "No"]:
                    # Get the statement text from the corresponding option set
                    idx = row_num - 1  # Convert to 0-based index
                    if idx < len(option_sets):
                        stmt_elem = option_sets[idx].find("Statement")
                        if stmt_elem is not None and stmt_elem.text:
                            # Add answer
                            ans_elem = ET.SubElement(answers_elem, "Answer")
                            ans_elem.set("statement", stmt_elem.text)
                            ans_elem.text = selected
        except Exception as e:
            st.warning(f"Failed to create visual analysis answers: {e}")
    
    # Extract references
    refs_match = re.search(r'References:([\s\S]+?)(?=QUESTION NO:|$)', question["description"], re.IGNORECASE)
    if refs_match:
        refs_section = refs_match.group(1).strip()
        refs = re.findall(r'(https?://\S+)', refs_section)
        if refs:
            refs_elem = ET.SubElement(root, "References")
            for idx, ref in enumerate(refs, start=1):
                ref_elem = ET.SubElement(refs_elem, f"Reference{idx}")
                ref_elem.text = ref.strip()
    
    return root

def build_xml_for_dragdrop(question, q_img, a_img):
    """
    Builds an XML element for a DRAGDROP question.
    """
    root = ET.Element("Question")
    root.set("number", question["number"])
    root.set("type", "DRAGDROP")
    
    # Clean description but preserve references
    clean_desc = remove_noise(question["description"])
    ET.SubElement(root, "Description").text = clean_desc
    
    # Extract columns data from question image
    columns_data = gemini_extract_columns_dynamic(q_img)
    
    # Add columns to XML
    dynamic_cols = ET.SubElement(root, "DynamicColumns")
    for col in columns_data.get("columns", []):
        heading = (col.get("heading") or "").strip()
        items = col.get("items", [])
        col_elem = ET.SubElement(dynamic_cols, "Column")
        col_elem.set("heading", heading)
        for it in items:
            ET.SubElement(col_elem, "Item").text = it.strip()
    
    # Extract answer pairs from answer image
    answer_pairs = gemini_extract_pairs_dynamic(a_img)
    
    # Add answer pairs to XML
    ans_pairs = ET.SubElement(root, "AnswerPairs")
    for pair in answer_pairs:
        pair_elem = ET.SubElement(ans_pairs, "Pair")
        for h_key, m_val in pair.items():
            col_elem = ET.SubElement(pair_elem, "Column")
            col_elem.set("name", h_key.strip())
            col_elem.text = m_val.strip()
    
    # Extract references
    refs_match = re.search(r'References:([\s\S]+?)(?=QUESTION NO:|$)', question["description"], re.IGNORECASE)
    if refs_match:
        refs_section = refs_match.group(1).strip()
        refs = re.findall(r'(https?://\S+)', refs_section)
        if refs:
            refs_elem = ET.SubElement(root, "References")
            for idx, ref in enumerate(refs, start=1):
                ref_elem = ET.SubElement(refs_elem, f"Reference{idx}")
                ref_elem.text = ref.strip()
    
    return root

def build_xml_for_dropdown(question, q_img, a_img):
    """
    Builds an XML element for a DROPDOWN question.
    """
    root = ET.Element("Question")
    root.set("number", question["number"])
    root.set("type", "DROPDOWN")
    
    # Clean description but preserve references
    clean_desc = remove_noise(question["description"])
    ET.SubElement(root, "Description").text = clean_desc
    
    # Extract dropdown question data from question image
    question_data = gemini_extract_dropdown_questions_from_image(q_img)
    
    # Extract dropdown answer data from answer image
    answers_data = gemini_extract_dropdown_answers_from_image(a_img)
    
    # Add options to XML
    options_elem = ET.SubElement(root, "QuestionOptions")
    for idx, qd in enumerate(question_data, start=1):
        stmt_text = (qd.get("statement") or "").strip()
        statement_header = (qd.get("statement_header") or "").strip()
        options_header = (qd.get("options_header") or "").strip()
        opts = qd.get("options") or []
        
        option_set = ET.SubElement(options_elem, "OptionSet")
        option_set.set("index", str(idx))
        
        stmt_hdr_elem = ET.SubElement(option_set, "ColumnHeaderStatement")
        stmt_hdr_elem.text = statement_header
        
        stmt_elem = ET.SubElement(option_set, "Statement")
        stmt_elem.text = stmt_text
        
        opts_hdr_elem = ET.SubElement(option_set, "ColumnHeaderOptions")
        opts_hdr_elem.text = options_header
        
        opts_parent = ET.SubElement(option_set, "Options")
        for opt in opts:
            opt_elem = ET.SubElement(opts_parent, "Option")
            opt_elem.text = opt
    
    # Add answers to XML
    answers_elem = ET.SubElement(root, "Answers")
    for ans_item in answers_data:
        statement = (ans_item.get("statement") or "").strip()
        answer = (ans_item.get("answer") or "").strip()
        statement_header = (ans_item.get("statement_header") or "").strip()
        answer_header = (ans_item.get("answer_header") or "").strip()
        
        ans_elem = ET.SubElement(answers_elem, "Answer")
        ans_elem.set("statement_header", statement_header)
        ans_elem.set("answer_header", answer_header)
        ans_elem.set("statement", statement)
        ans_elem.text = answer
    
    # Extract references
    refs_match = re.search(r'References:([\s\S]+?)(?=QUESTION NO:|$)', question["description"], re.IGNORECASE)
    if refs_match:
        refs_section = refs_match.group(1).strip()
        refs = re.findall(r'(https?://\S+)', refs_section)
        if refs:
            refs_elem = ET.SubElement(root, "References")
            for idx, ref in enumerate(refs, start=1):
                ref_elem = ET.SubElement(refs_elem, f"Reference{idx}")
                ref_elem.text = ref.strip()
    
    return root

############################################
# Chunk Processing Functions
############################################

def process_chunk(chunk_id):
    """
    Process a single chunk of questions.
    """
    # Find the chunk in the session state
    chunk = None
    for c in st.session_state.file_chunks:
        if c["id"] == chunk_id:
            chunk = c
            break
    
    if not chunk:
        st.error(f"Chunk with ID {chunk_id} not found.")
        return None
    
    # Get text and images for this chunk
    text = chunk["text"]
    question_blocks = split_questions(text)
    
    # Get relevant images for this chunk
    image_indices = chunk["image_indices"]
    all_images = st.session_state.all_images
    chunk_images = [all_images[i] for i in image_indices if i < len(all_images)]
    
    # Create image pairs
    image_pairs = []
    for i in range(0, len(chunk_images), 2):
        if i + 1 < len(chunk_images):
            image_pairs.append((chunk_images[i], chunk_images[i+1]))
    
    # Process each question
    results = []
    img_index = 0
    
    for block in question_blocks:
        if not block or not block.strip():
            continue
        
        # Get question info
        question_info = analyze_question_block(block)
        if not question_info:
            continue
        
        q_number = question_info["number"]
        q_type = question_info["type"]
        
        st.info(f"Processing Question {q_number} of type {q_type}")
        
        try:
            # Process based on question type
            if q_type in ["HOTSPOT", "DRAGDROP", "DROPDOWN"]:
                # Check if we have enough image pairs
                if img_index < len(image_pairs):
                    q_img, a_img = image_pairs[img_index]
                    
                    # Validate images
                    if not is_valid_image(q_img) or not is_valid_image(a_img):
                        st.error(f"Invalid image data for question {q_number}")
                        img_index += 1
                        continue
                    
                    # Process based on specific type
                    if q_type == "HOTSPOT":
                        q_elem = build_xml_for_hotspot(
                            {"number": q_number, "type": "HOTSPOT", "description": block},
                            q_img, a_img
                        )
                    elif q_type == "DRAGDROP":
                        q_elem = build_xml_for_dragdrop(
                            {"number": q_number, "type": "DRAGDROP", "description": block},
                            q_img, a_img
                        )
                    else:  # DROPDOWN
                        q_elem = build_xml_for_dropdown(
                            {"number": q_number, "type": "DROPDOWN", "description": block},
                            q_img, a_img
                        )
                    
                    # Always advance image index after processing image-based question
                    img_index += 1
                    
                    if q_elem is not None:
                        results.append({
                            "number": q_number,
                            "type": q_type,
                            "element": q_elem
                        })
                    
                else:
                    st.error(f"Not enough image pairs for question {q_number}")
            else:
                # Process text-based question
                data = parse_question_text_textbased(block)
                q_elem = build_xml_from_data_textbased(data)
                
                if q_elem is not None:
                    results.append({
                        "number": q_number,
                        "type": q_type,
                        "element": q_elem
                    })
                
        except Exception as e:
            st.error(f"Error processing question {q_number}: {str(e)}")
            traceback.print_exc()
            
            # For image-based questions, still advance the image counter
            if q_type in ["HOTSPOT", "DRAGDROP", "DROPDOWN"]:
                img_index += 1
    
    # Generate XML for this chunk
    if results:
        chunk_xml = create_xml_from_results(results)
        
        # Store the results in session state
        st.session_state.processed_chunks[chunk_id] = {
            "xml": chunk_xml,
            "results": results
        }
        
        # Mark the chunk as processed
        for i, c in enumerate(st.session_state.file_chunks):
            if c["id"] == chunk_id:
                st.session_state.file_chunks[i]["processed"] = True
                break
        
        return chunk_xml
    else:
        st.error("No questions were processed successfully.")
        return None

def create_xml_from_results(results):
    """
    Creates an XML string from a list of question result dictionaries.
    """
    # Create a root element for the collection
    root_collection = ET.Element("QuestionsCollection")
    
    # Sort results by question number to ensure proper order
    results.sort(key=lambda x: int(x["number"]))
    
    # Add each question to the collection
    for result in results:
        root_collection.append(result["element"])
    
    # Convert to pretty XML
    rough_string = ET.tostring(root_collection, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    final_xml = reparsed.toprettyxml(indent="    ")
    
    return final_xml

def combine_all_processed_chunks():
    """
    Combines all processed chunks into a single XML file.
    """
    if not st.session_state.processed_chunks:
        st.error("No chunks have been processed yet.")
        return None
    
    # Collect all results from all processed chunks
    all_results = []
    for chunk_id, chunk_data in st.session_state.processed_chunks.items():
        all_results.extend(chunk_data["results"])
    
    # Create combined XML
    if all_results:
        return create_xml_from_results(all_results)
    else:
        st.error("No processed results found.")
        return None

############################################
# Main Application Functions
############################################

def reset_app_state():
    """Reset the application state."""
    st.session_state.file_chunks = []
    st.session_state.processed_chunks = {}
    st.session_state.all_images = []

def split_uploaded_file(uploaded_file, batch_size):
    """
    Split the uploaded ODT file into chunks and store in session state.
    """
    try:
        # Reset any existing chunks
        st.session_state.file_chunks = []
        st.session_state.processed_chunks = {}
        
        # Split the file into chunks
        chunks, images = split_odt_into_chunks(uploaded_file, batch_size)
        
        # Store chunks and images in session state
        st.session_state.file_chunks = chunks
        st.session_state.all_images = images
        
        return True
    except Exception as e:
        st.error(f"Error splitting file: {str(e)}")
        traceback.print_exc()
        return False

def main():
    st.set_page_config(
        page_title="ODT to XML Converter",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("ODT Questions to XML Converter - Interactive Batch Processing")
    
    api_key = st.sidebar.text_input("Write Your Api Key here", type="password")
    # Configure Gemini API key
    genai.configure(api_key=api_key)  # Replace with your API key
    # Sidebar with configuration options
    st.sidebar.header("Configuration")
    
    batch_size = st.sidebar.slider(
        "Questions per batch", 
        min_value=10, 
        max_value=100, 
        value=100,
        help="How many questions to include in each batch."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info("""
    This application converts ODT files containing questions to structured XML.
    
    **Features:**
    - Splits the ODT into smaller batches of questions
    - Process each batch separately to avoid content contamination
    - Download individual batch XMLs for verification
    - Combine all processed batches into a final XML
    """)
    
    # Reset button in sidebar
    if st.sidebar.button("Reset Application"):
        reset_app_state()
        st.rerun()
    
    # Main content
    st.write("""
    ## Instructions
    
    1. Upload an ODT file that contains multiple questions.
    2. Click "Split File" to divide it into manageable batches.
    3. Process each batch by clicking the "Process" button.
    4. After all batches are processed, generate the combined XML.
    
    ### Why process in batches?
    Processing in smaller batches helps prevent errors in one section from affecting others,
    particularly with complex image-based questions like hotspots.
    """)
    
    # Upload section
    uploaded_file = st.file_uploader("Upload your ODT file", type=["odt"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            split_button = st.button("Split File", type="primary", use_container_width=True)
        
        with col2:
            cancel_button = st.button("Cancel", type="secondary", use_container_width=True)
        
        if split_button:
            with st.spinner("Splitting file into batches..."):
                if split_uploaded_file(uploaded_file, batch_size):
                    st.success(f"File successfully split into {len(st.session_state.file_chunks)} batches")
    
    # Display chunks if available
    if hasattr(st.session_state, 'file_chunks') and st.session_state.file_chunks:
        st.subheader("Question Batches")
        
        # Create a DataFrame for better display
        chunks_data = []
        for chunk in st.session_state.file_chunks:
            chunks_data.append({
                "Batch": chunk["batch_idx"],
                "Questions": f"Q{chunk['first_q']}-Q{chunk['last_q']}",
                "Count": chunk["question_count"],
                "Image Count": chunk["image_count"],
                "Status": "Processed ✅" if chunk["processed"] else "Pending ⏳",
                "ID": chunk["id"]
            })
        
        df = pd.DataFrame(chunks_data)
        
        # Display table with actions
        for i, row in df.iterrows():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
            
            with col1:
                st.write(f"**Batch {row['Batch']}:** {row['Questions']}")
            
            with col2:
                st.write(f"{row['Count']} questions, {row['Image Count']} images")
            
            with col3:
                st.write(row['Status'])
            
            with col4:
                # If not processed, show process button
                chunk_id = row['ID']
                if row['Status'] == "Pending ⏳":
                    if st.button(f"Process Batch {row['Batch']}", key=f"process_{chunk_id}"):
                        with st.spinner(f"Processing Batch {row['Batch']}..."):
                            xml = process_chunk(chunk_id)
                            if xml:
                                st.success(f"Batch {row['Batch']} processed successfully")
                                st.download_button(
                                    f"Download Batch {row['Batch']} XML",
                                    xml,
                                    file_name=f"batch_{row['Batch']}_questions.xml",
                                    mime="application/xml",
                                    key=f"download_{chunk_id}"
                                )
                else:
                    # If processed, show download button
                    if chunk_id in st.session_state.processed_chunks:
                        xml = st.session_state.processed_chunks[chunk_id]["xml"]
                        st.download_button(
                            f"Download XML",
                            xml,
                            file_name=f"batch_{row['Batch']}_questions.xml",
                            mime="application/xml",
                            key=f"download_{chunk_id}"
                        )
        
        # Check if all chunks are processed
        all_processed = all(chunk["processed"] for chunk in st.session_state.file_chunks)
        
        st.markdown("---")
        
        # Show combined download button if all chunks are processed
        if all_processed:
            st.success("All batches have been processed!")
            
            if st.button("Generate Combined XML", type="primary"):
                with st.spinner("Generating combined XML..."):
                    final_xml = combine_all_processed_chunks()
                    if final_xml:
                        st.success("Combined XML generated successfully")
                        st.download_button(
                            "Download Complete XML File",
                            final_xml,
                            file_name="all_questions_combined.xml",
                            mime="application/xml",
                            use_container_width=True
                        )
                        
                        # Preview section
                        with st.expander("Preview Combined XML"):
                            st.code(final_xml[:5000] + "\n...\n" + final_xml[-1000:] if len(final_xml) > 6000 else final_xml, language="xml")
        else:
            # Show message about remaining chunks
            pending_count = sum(1 for chunk in st.session_state.file_chunks if not chunk["processed"])
            st.info(f"{pending_count} batches still need to be processed before the combined XML can be generated.")

if __name__ == "__main__":
    main()

