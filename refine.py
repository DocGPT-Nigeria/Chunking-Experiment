# --- Imports ---
import os
import json
import time  # We'll use this to be respectful of the API rate limits
from dotenv import load_dotenv  # To load our secret .env file
import google.generativeai as genai  # The Google AI SDK

# --- Import our own functions from our other files ---
from ingest import load_text_from_pdf
from chunk import chunk_the_text

# --- Configuration ---

# == IMPORTANT: SET THIS TO FALSE FOR THE FULL RUN! ==
# When True, we'll only process the first 3 chunks as a quick test.
# When False, it will process ALL chunks (which could take a long time).
TESTING_MODE = False
# ====================================================

PDF_FILE_PATH = "B09514-eng.pdf"
OUTPUT_FILE_PATH = "processed_data.json"  # Our final, clean file!

# This is Chisom's prompt, adapted for the API.
# It tells the AI to fix broken text and return *only* JSON.
MASTER_PROMPT_TEMPLATE = """
You are an expert AI medical data analyst. Your task is to analyze the following medical text chunk, which was extracted from a PDF and may contain OCR errors (like 'r ecommenda tion' or 'pr evention o f').

Your two jobs are:
1.  **Fix the text:** Correct any broken words or spacing errors to produce a clean, readable text block.
2.  **Extract metadata:** Generate a title, category, and keywords based on the *fixed* text.

You MUST return your analysis ONLY in the following JSON format. Do not add any conversational text or markdown.

{
  "category": "The inferred medical category (e.g., 'Malaria Prevention', 'Diagnosis', 'Treatment')",
  "title": "A short, descriptive title for this text chunk (less than 15 words)",
  "content": "The *full text* of the chunk, with all OCR errors and broken words (like 'r ecommenda tion') corrected to be clean, readable text (like 'recommendation').",
  "keywords": [
    "keyword1",
    "keyword2",
    "keyword3"
  ],
  "source": "WHO Guidelines for Malaria"
}
"""

# --- Main Refinery Logic ---

def configure_api():
    """Loads the API key and configures the Gemini client."""
    # 1. Load the .env file (it finds GEMINI_API_KEY)
    load_dotenv()
    
    # 2. Read the API key from the environment
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    # 3. Safety check
    if not API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not found. Make sure it's in your .env file.")
        
    # 4. Configure the Google AI SDK
    genai.configure(api_key=API_KEY)
    print("Google AI SDK configured successfully.")

def get_refinery_model():
    """Initializes and returns the Gemini model we'll use."""
    
    # This setting tells the API to *guarantee* its output is JSON.
    # This is a powerful feature that makes our job much easier.
    generation_config = {
        "response_mime_type": "application/json",
    }
    
    # We use a fast and modern model.
    # 'gemini-2.5-flash-preview-09-2025' is a great choice.
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025",
        generation_config=generation_config
    )
    print(f"Using model: {model.model_name}")
    return model

def refine_chunk(model, messy_chunk):
    """
    Sends one messy chunk to the API and gets back the clean JSON text.
    """
    # Combine the master prompt with the actual chunk text
    full_prompt = f"{MASTER_PROMPT_TEMPLATE}\n\nHere is the text to analyze:\n\n{messy_chunk}"
    
    try:
        # 4. Make the API call!
        response = model.generate_content(full_prompt)
        
        # 5. Return the raw JSON text from the response
        return response.text
        
    except Exception as e:
        # 6. Handle errors (e.g., API limits, safety blocks)
        print(f"  !! Error processing chunk: {e}")
        return None

# --- This is the part that runs when you execute the script ---
if __name__ == "__main__":
    
    # --- Setup ---
    configure_api()
    model = get_refinery_model()
    
    # --- Step 1 & 2: Load and Chunk ---
    print("\n--- Step 1: Loading Raw Text ---")
    raw_text = load_text_from_pdf(PDF_FILE_PATH)
    
    if not raw_text:
        print("Failed to load text. Exiting.")
        exit()
        
    print("\n--- Step 2: Chunking Messy Text ---")
    chunks_list = chunk_the_text(raw_text)
    
    if not chunks_list:
        print("Failed to chunk text. Exiting.")
        exit()
        
    # --- Step 3: The Refinery Loop ---
    print(f"\n--- Step 3: Starting Refinery (TESTING_MODE = {TESTING_MODE}) ---")
    
    # This is our "bucket" for all the clean data
    all_clean_data = []
    
    # Decide which chunks to process (all, or just the first 3 for testing)
    chunks_to_process = chunks_list
    if TESTING_MODE:
        chunks_to_process = chunks_list[:3] # Just the first 3
        print(f"*** Running in TEST MODE. Will only process {len(chunks_to_process)} chunks. ***")

    # This is the main loop!
    for i, chunk in enumerate(chunks_to_process):
        
        print(f"\nRefining chunk {i + 1} of {len(chunks_to_process)}...")
        
        # 1. Send the messy chunk to the API
        json_response_text = refine_chunk(model, chunk)
        
        if json_response_text:
            try:
                # 2. Convert the JSON text string into a Python dictionary
                data = json.loads(json_response_text)
                
                # 3. Add our own ID
                data['id'] = f"chunk_{i+1:04d}" # e.g., "chunk_0001"
                
                # 4. Add the clean data to our bucket
                all_clean_data.append(data)
                print(f"  > Success! Added chunk '{data['title']}'")
                
            except json.JSONDecodeError:
                # Handle cases where the API returns bad JSON
                print(f"  !! Error: Could not decode JSON response for this chunk.")
                print(f"     Raw response was: {json_response_text}")

        # --- IMPORTANT: RATE LIMITING ---
        # We must pause briefly between API calls to be respectful
        # and avoid "429: Too Many Requests" errors.
        # 2 seconds is a very safe pause.
        print("  > Pausing for 2 seconds...")
        time.sleep(2)
        # --------------------------------

    # --- Final Step: Save Everything ---
    print(f"\n--- REFINERY COMPLETE ---")
    
    if not all_clean_data:
        print("No data was processed. Exiting.")
        exit()

    print(f"Processed {len(all_clean_data)} chunks. Saving to {OUTPUT_FILE_PATH}...")
    
    # Save our list of dictionaries as a beautifully formatted JSON file
    try:
        with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(all_clean_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n*** SUCCESS! ***")
        print(f"Your clean, processed data is saved in '{OUTPUT_FILE_PATH}'.")
        print("You can open this file in VS Code to see the final result!")
        
    except Exception as e:
        print(f"!! Critical Error: Failed to save final JSON file: {e}")
