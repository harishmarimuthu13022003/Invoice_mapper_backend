
import os
import sys

# Add current directory to path so it can find rag_engine
sys.path.append(os.getcwd())

from rag_engine import RAGEngine

def test_parser():
    rag = RAGEngine()
    
    # Try to find a PDF in uploads
    upload_dir = "./uploads"
    if not os.path.exists(upload_dir):
        print(f"Directory {upload_dir} not found")
        return
        
    pdfs = [f for f in os.listdir(upload_dir) if f.endswith('.pdf')]
    
    if not pdfs:
        print("No PDFs found in uploads directory")
        return
    
    pdf_path = os.path.join(upload_dir, pdfs[0])
    print(f"Testing with PDF: {pdf_path}")
    
    # Test extract_text_from_pdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
            
        print("\n--- EXTRACTED TEXT (First 1000 chars) ---")
        print(full_text[:1000])
        print("---------------------------------------\n")
        
        # Test parse_line_items
        items = rag.parse_line_items(full_text)
        
        print(f"Found {len(items)} items:")
        for i, item in enumerate(items):
            print(f"{i+1}. Desc: [{item['description']}] | Amount: {item['amount']}")
            
    except Exception as e:
        print(f"Error testing: {e}")

if __name__ == "__main__":
    test_parser()
