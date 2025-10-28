# --- Imports ---
# First, we import our *own* function from our other file
from ingest import load_text_from_pdf

# Now, we import the specific "Text Splitter" tool from LangChain
# This import path has been updated for modern langchain packages!
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
PDF_FILE_PATH = "B09514-eng.pdf"

# --- Main Chunking Logic ---
def chunk_the_text(raw_text):
    """
    Takes the giant raw text and splits it into smaller, 
    semantically meaningful chunks.
    """
    print("Starting the chunking process...")
    
    # 1. Initialize the Text Splitter
    # This is the "intelligent" splitter.
    text_splitter = RecursiveCharacterTextSplitter(
        # A) Set the target size for each chunk (in characters)
        # Chisom suggested 1000-1500 words. An average word is ~5 chars,
        # so let's aim for ~7500 characters.
        chunk_size = 7500,
        
        # B) Set a "chunk overlap" (in characters)
        # This is CRITICAL. It makes sure chunks have context.
        # If a sentence is cut off, it will be "caught" in the
        # next chunk. 400 chars is a good starting point.
        chunk_overlap = 400,
        
        # C) Use standard separators for splitting
        # It will try to split by paragraph (\n\n) first, then
        # by line (\n), then by space (" "). This is the "semantic" part.
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 2. Run the splitter on our raw text
    # This might take a moment
    chunks = text_splitter.split_text(raw_text)
    
    print(f"Successfully chunked the text into {len(chunks)} chunks.")
    return chunks

# --- This is the part that runs when you execute the script ---
if __name__ == "__main__":
    # 1. Load the raw text using our function from ingest.py
    raw_text = load_text_from_pdf(PDF_FILE_PATH)
    
    if raw_text:
        # 2. If loading was successful, chunk the text
        chunks_list = chunk_the_text(raw_text)
        
        if chunks_list:
            # 3. Let's check our work!
            # We'll print the total number of chunks and then
            # print the first chunk to see what it looks like.
            
            print("\n--- CHUNKING SUCCESSFUL ---")
            print(f"Total chunks created: {len(chunks_list)}")
            
            print("\n--- SAMPLE: FIRST CHUNK (Chunk 0) ---")
            print(chunks_list[0])
            print("---------------------------------------")
            
            print("\n--- SAMPLE: SECOND CHUNK (Chunk 1) ---")
            print(chunks_list[1][:200]) # Just the first 200 chars of chunk 1
            print("...[truncated]...")
            print("---------------------------------------")

