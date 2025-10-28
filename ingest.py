import os
from pypdf import PdfReader

# The name of the PDF file you added to the folder
PDF_FILE_PATH = "B09514-eng.pdf"

def load_text_from_pdf(file_path):
    """
    Extracts raw text from a PDF file, page by page.
    """
    # 1. Check if the file actually exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading raw text from {file_path}...")
    
    # 2. Initialize the PDF reader
    reader = PdfReader(file_path)
    
    # 3. Create a variable to hold all the text
    all_text = ""
    
    # 4. Loop through each page in the PDF
    for i, page in enumerate(reader.pages):
        # 5. Extract the text from the page
        text = page.extract_text()
        
        # 6. Add the extracted text to our variable
        if text:
            all_text += text + "\n--- Page Break ---\n"
            
    print(f"Successfully extracted text from {len(reader.pages)} pages.")
    return all_text

# This is the standard way to make a Python script runnable
if __name__ == "__main__":
    # 7. Run our function
    raw_text = load_text_from_pdf(PDF_FILE_PATH)
    
    if raw_text:
        # 8. Let's check our work!
        # We'll print the total number of characters and a small sample.
        # Printing all 400+ pages would flood your terminal!
        
        total_length = len(raw_text)
        sample_text = raw_text[:2000] # Get the first 2000 characters
        
        print(f"\n--- EXTRACTION SUCCESSFUL ---")
        print(f"Total characters extracted: {total_length}")
        
        print("\n--- SAMPLE OF RAW, MESSY TEXT (First 2000 Chars) ---")
        print(sample_text)
        print("-----------------------------------------------------")
