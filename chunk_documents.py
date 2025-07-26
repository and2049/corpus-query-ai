from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

# Define our text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

pdf_path = "papers/2507.17309v1.Confounded_Causal_Imitation_Learning_with_Instrumental_Variables.pdf"
document_text = extract_text_from_pdf(pdf_path)

chunks = text_splitter.split_text(document_text)

print(f"--- Created {len(chunks)} chunks from {pdf_path} ---")
print("\n--- Example Chunk ---")
print(chunks[0])