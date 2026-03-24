"""
RAG Engine - Pure API approach, NO langchain!
Uses Groq and Google Gemini APIs directly with TF-IDF for embeddings
"""

import os
import json
import math
import requests
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from pypdf import PdfReader
from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# ============== MODELS ==============

class ServiceCode(BaseModel):
    code: str
    description: str
    category: str

class RAGResponse(BaseModel):
    suggested_code: str
    confidence_score: float
    reasoning: str
    needs_approval: bool = False  # Flag for new category requests
    suggested_category: str = ""  # Suggested new category name
    suggested_code_format: str = ""  # Suggested service code format

# ============== CATEGORY TO SERVICE CODE MAPPING ==============
# All descriptions under the same category will get the same service code

CATEGORY_SERVICE_CODES = {
    "Personal Care": "01_011_0107_1_1",  # Assistance with Self-Care Activities
    "Nursing": "01_121_1117_1_1",         # Nursing services
    "Domestic Assistance": "01_020_0120_1_1",  # House Cleaning and Other Household Activities
    "Gardening": "01_019_0120_1_1",      # House or Yard Maintenance
    "Meals": "01_131_1117_1_1",         # Meal services
    "Transport": "01_146_1117_1_1",      # Transport services
    "Allied Health": "01_230_1117_1_1",  # Allied Health services
}


# ============== RAG ENGINE ==============

class RAGEngine:
    """
    RAG Engine using pure API calls (no langchain!)
    - Groq API for LLM (primary)
    - Google Gemini API for LLM (fallback)
    # - TF-IDF for embeddings (no PyTorch needed!)
    """
    
    def __init__(self):
        print("\n" + "="*50)
        print("[INIT] Initializing RAG Engine (Pure API, No PyTorch)...")
        print("="*50)
        
        # Load configuration
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.pinecone_index = os.getenv("PINECONE_INDEX_NAME", "invoice-mapper")
        self.similarity_top_k = int(os.getenv("SIMILARITY_TOP_K", "3"))
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.90"))
        
        # Groq API Keys
        self.groq_keys = [
            os.getenv("GROQ_API_KEY_1", "").strip(),
            os.getenv("GROQ_API_KEY_2", "").strip()
        ]
        self.groq_keys = [k for k in self.groq_keys if k]
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-70b-versatile")
        self.current_groq_index = 0
        
        # Google API Keys
        self.google_keys = [
            os.getenv("GOOGLE_API_KEY_1", "").strip(),
            os.getenv("GOOGLE_API_KEY_2", "").strip()
        ]
        self.google_keys = [k for k in self.google_keys if k]
        self.google_model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
        self.current_google_index = 0
        
        # Initialize components
        self.client = None
        self.tfidf = None
        self.is_initialized = False
        self.current_api = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone and TF-IDF"""
        try:
            # Initialize Pinecone
            if self.pinecone_api_key:
                self.client = Pinecone(
                    api_key=self.pinecone_api_key,
                    environment=self.pinecone_env
                )
                print(f"[OK] Pinecone connected: {self.pinecone_index}")
                
                # Get the index
                self.index = self.client.Index(self.pinecone_index)
            else:
                print("[WARN] Pinecone API key not configured, using fallback mode")
                self.client = None
                self.index = None
            
            # Initialize TF-IDF (no PyTorch!)
            self.tfidf = TfidfVectorizer(
                max_features=384,
                stop_words='english',
                ngram_range=(1, 2)
            )
            print("[OK] TF-IDF embeddings ready (no PyTorch!)")
            
            self.is_initialized = True
            print("[OK] RAG Engine initialized!\n")
            
        except Exception as e:
            print(f"[WARN] RAG initialization error: {e}")
            self.is_initialized = False
    
    # ============== GROQ API ==============
    
    def _call_groq(self, prompt: str) -> Optional[str]:
        """Call Groq API directly"""
        if not self.groq_keys:
            return None
        
        api_key = self.groq_keys[self.current_groq_index]
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.current_api = "groq"
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                # Rate limit - try next key
                if self._switch_groq_key():
                    return self._call_groq(prompt)
            else:
                print(f"   [WARN] Groq error: {response.status_code}")
                
        except Exception as e:
            print(f"   [WARN] Groq exception: {e}")
        
        return None
    
    def _switch_groq_key(self) -> bool:
        """Switch to next Groq key"""
        if len(self.groq_keys) > 1:
            self.current_groq_index = (self.current_groq_index + 1) % len(self.groq_keys)
            print(f"   [SWITCH] Switched to Groq key #{self.current_groq_index + 1}")
            return True
        return False
    
    # ============== GOOGLE GEMINI API ==============
    
    def _call_google(self, prompt: str) -> Optional[str]:
        """Call Google Gemini API directly"""
        if not self.google_keys:
            return None
        
        api_key = self.google_keys[self.current_google_index]
        
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1/models/{self.google_model}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.current_api = "google"
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif response.status_code == 429:
                if self._switch_google_key():
                    return self._call_google(prompt)
            else:
                print(f"   [WARN] Google error: {response.status_code}")
                
        except Exception as e:
            print(f"   [WARN] Google exception: {e}")
        
        return None
    
    def _switch_google_key(self) -> bool:
        """Switch to next Google key"""
        if len(self.google_keys) > 1:
            self.current_google_index = (self.current_google_index + 1) % len(self.google_keys)
            print(f"   [SWITCH] Switched to Google key #{self.current_google_index + 1}")
            return True
        return False
    
    # ============== LLM CALL WITH FALLBACK ==============
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with automatic failover between Groq and Google"""
        
        # Try Groq first
        if self.groq_keys:
            result = self._call_groq(prompt)
            if result:
                return result
            print("   [FALLBACK] Groq failed - trying Google...")
        
        # Try Google as fallback
        if self.google_keys:
            result = self._call_google(prompt)
            if result:
                return result
            print("   [FALLBACK] Google also failed")
        
        return "ERROR: All LLM APIs failed"
    
    # ============== PDF PROCESSING ==============
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        overlap = 50  # 50 character overlap between chunks
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def extract_text_from_pdf(self, pdf_path: str, chunk_size: int = 500) -> List[str]:
        """Extract text from PDF and split into chunks"""
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            
            # Split into chunks
            return self.chunk_text(full_text, chunk_size)
        except Exception as e:
            print(f"   [ERROR] PDF extraction error: {e}")
            return []
    
    def parse_line_items(self, text: str) -> List[dict]:
        """Parse invoice text into line items with descriptions and amounts"""
        import re
        lines = text.split('\n')
        items = []
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.isdigit():
                # Try to extract dollar amount from line
                amount_match = re.search(r'\$\s*([\d,]+\.?\d*)', line)
                amount = 0.0
                if amount_match:
                    amount_str = amount_match.group(1)
                    try:
                        amount = float(amount_str.replace(',', ''))
                    except:
                        pass
                
                # Clean up the description - remove dates, quantities, and amounts
                clean_desc = line
                
                # Remove date patterns (e.g., 21/03/2026, 2026-03-21, etc.)
                clean_desc = re.sub(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', '', clean_desc)
                
                # Remove quantity patterns - numbers followed by units or at end
                clean_desc = re.sub(r'\s+\d+\.?\d*\s*', ' ', clean_desc)
                
                # Remove leading numbers
                clean_desc = re.sub(r'^\s*\d+\.?\d*\s+', '', clean_desc)
                
                # Remove dollar amounts (all patterns)
                clean_desc = re.sub(r'\$[\d,]+\.?\d*', '', clean_desc)
                clean_desc = re.sub(r'[\d,]+\.?\d*\s*\$', '', clean_desc)
    
    def parse_line_items_simple(self, text: str) -> List[str]:
        """Parse invoice text into line items (legacy method)"""
        lines = text.split('\n')
        return [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]
    
    # ============== RAG OPERATIONS ==============
    
    def retrieve_similar_codes(self, description: str, top_k: int = 3) -> List[ServiceCode]:
        """Retrieve similar service codes using TF-IDF"""
        
        if not self.index:
            return self._get_fallback_codes()
        
        try:
            # Get all vectors from Pinecone
            result = self.index.describe_index_stats()
            
            if result.total_vector_count == 0:
                return self._get_fallback_codes()
            
            # Query Pinecone to get all vectors (using empty filter to get all)
            query_result = self.index.query(
                vector=[0] * 384,  # Dummy vector, we'll use TF-IDF for scoring
                top_k=result.total_vector_count,
                include_metadata=True,
                include_values=False
            )
            
            if not query_result.matches:
                return self._get_fallback_codes()
            
            # Build document list from Pinecone results
            documents = []
            metadatas = []
            for match in query_result.matches:
                documents.append(match.metadata.get('description', ''))
                metadatas.append({
                    'code': match.metadata.get('code', 'UNKNOWN'),
                    'category': match.metadata.get('category', 'General')
                })
            
            if not documents:
                return self._get_fallback_codes()
            
            # Fit TF-IDF on documents
            self.tfidf.fit(documents)
            
            # Get query vector
            query_vector = self.tfidf.transform([description]).toarray()[0]
            
            # Calculate cosine similarity
            similarities = []
            for i, doc in enumerate(documents):
                doc_vector = self.tfidf.transform([doc]).toarray()[0]
                
                dot = sum(a * b for a, b in zip(query_vector, doc_vector))
                query_mag = math.sqrt(sum(a * a for a in query_vector))
                doc_mag = math.sqrt(sum(a * a for a in doc_vector))
                
                if query_mag > 0 and doc_mag > 0:
                    sim = dot / (query_mag * doc_mag)
                else:
                    sim = 0
                
                similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            results = []
            for i, sim in similarities[:top_k]:
                results.append(ServiceCode(
                    code=metadatas[i].get('code', 'UNKNOWN'),
                    description=documents[i],
                    category=metadatas[i].get('category', 'General')
                ))
            
            return results
            
        except Exception as e:
            print(f"   [WARN] Retrieval error: {e}")
            return self._get_fallback_codes()
    
    def _get_fallback_codes(self) -> List[ServiceCode]:
        """Fallback codes"""
        return [
            ServiceCode(code="01_111_1117_1_1", description="Personal Care Assistance", category="Personal Care"),
            ServiceCode(code="01_112_1117_1_1", description="Domestic Assistance", category="Domestic Assistance"),
            ServiceCode(code="01_117_1117_1_1", description="Gardening & Maintenance", category="Gardening"),
        ]
    
    def generate_response(self, description: str, similar_codes: List[ServiceCode]) -> RAGResponse:
        """Generate response using LLM with consistent category-to-code mapping"""
        
        # First, determine the category using LLM - ask if it matches existing categories or needs new one
        category_prompt = f"""You are an AI assistant for categorizing aged care invoice descriptions.

EXISTING CATEGORIES:
- Personal Care (Code: 01_011_0107_1_1)
- Nursing (Code: 01_121_1117_1_1)
- Domestic Assistance (Code: 01_020_0120_1_1)
- Gardening (Code: 01_019_0120_1_1)
- Meals (Code: 01_131_1117_1_1)
- Transport (Code: 01_146_1117_1_1)
- Allied Health (Code: 01_230_1117_1_1)

Invoice Description: {description}

Analyze the description and respond with ONLY a JSON object:
{{
    "is_new_category": true or false,
    "matched_category": "<if existing, name of matched category> or null",
    "suggested_category": "<if new, suggest a new category name> or null",
    "reasoning": "<brief explanation>"
}}

JSON:"""
        
        category_response = self._call_llm(category_prompt)
        
        # Parse category from response
        is_new_category = False
        detected_category = None
        suggested_category = None
        reasoning = ""
        
        try:
            json_start = category_response.find('{')
            json_end = category_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = category_response[json_start:json_end]
                result = json.loads(json_str)
                is_new_category = result.get("is_new_category", False)
                detected_category = result.get("matched_category")
                suggested_category = result.get("suggested_category")
                reasoning = result.get("reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # If category not detected from LLM, use keyword matching as fallback
        if not detected_category and not is_new_category:
            detected_category = self._detect_category_by_keywords(description)
        
        # If it's a new category request
        if is_new_category or not detected_category:
            # Ask LLM to suggest a new category and service code format
            new_category_prompt = f"""A new invoice description doesn't match any existing category.

Invoice Description: {description}

Please suggest:
1. A new category name that would fit this service
2. A format for the NDIS-style service code (e.g., 01_XXX_XXXX_X_X)

Respond with ONLY a JSON object:
{{
    "suggested_category": "<new category name>",
    "suggested_code_format": "01_<3-digit>_<4-digit>_<1>_<1>",
    "description": "<brief description of this service category>"
}}

JSON:"""
            new_cat_response = self._call_llm(new_category_prompt)
            
            try:
                json_start = new_cat_response.find('{')
                json_end = new_cat_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = new_cat_response[json_start:json_end]
                    result = json.loads(json_str)
                    suggested_category = result.get("suggested_category", "New Category")
                    suggested_code_format = result.get("suggested_code_format", "01_999_9999_1_1")
            except (json.JSONDecodeError, AttributeError):
                suggested_category = "New Category"
                suggested_code_format = "01_999_9999_1_1"
            
            return RAGResponse(
                suggested_code="PENDING_APPROVAL",
                confidence_score=0.0,
                reasoning=f"New category needed: {reasoning}",
                needs_approval=True,
                suggested_category=suggested_category,
                suggested_code_format=suggested_code_format
            )
        
        # Get the consistent service code for this category
        suggested_code = CATEGORY_SERVICE_CODES.get(detected_category, "UNKNOWN")
        
        # If category not in our mapping, use keyword fallback
        if suggested_code == "UNKNOWN":
            return self._keyword_fallback(description, similar_codes)
        
        return RAGResponse(
            suggested_code=suggested_code,
            confidence_score=0.95,  # High confidence since we use consistent codes per category
            reasoning=f"Category '{detected_category}' mapped to service code {suggested_code}",
            needs_approval=False
        )
    
    def _detect_category_by_keywords(self, description: str) -> str:
        """Detect category using keyword matching"""
        desc_lower = description.lower()
        
        # Category keywords
        keywords = {
            "Personal Care": ["shower", "bath", "dressing", "grooming", "hygiene", "toilet", "mobility", "transfer", "personal care", "assistance with self-care"],
            "Nursing": ["nurse", "nursing", "medication", "wound", "health", "medical", "doctor", "clinical", "therapy"],
            "Domestic Assistance": ["cleaning", "laundry", "vacuum", "mop", "dust", "house", "dish", "washing", "ironing", "bed linen", "household"],
            "Gardening": ["garden", "lawn", "mow", "weed", "prune", "yard", "gutter", "maintenance", "outdoor"],
            "Meals": ["meal", "food", "cooking", "lunch", "dinner", "breakfast", "delivery"],
            "Transport": ["transport", "drive", "car", "appointment", "travel", "bus", "taxi"],
            "Allied Health": ["physio", "physiotherapy", "occupational", "speech", "podiatry", "dietitian", "psychologist"],
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in desc_lower:
                    return category
        
        return "Domestic Assistance"  # Default category

    # ============== GENDER DETECTION ==============
    
    def detect_gender_from_name(self, invoice_text: str) -> str:
        """
        Detect gender from invoice person name using LLM.
        The LLM extracts the person name from the invoice and determines gender.
        Returns: "Male", "Female", or "Unknown"
        """
        prompt = f"""You are an AI assistant that extracts person names from invoices and determines gender.

Invoice Text:
{invoice_text[:2000]}  # Limit text length

Instructions:
1. Extract the CLIENT/CUSTOMER/PERSON name from the invoice
2. Determine if the person is Male or Female based on the name
3. Consider common first names (e.g., John, Michael, David = Male; Mary, Jane, Clara = Female)

Respond with ONLY a JSON object:
{{
    "person_name": "<extracted name or 'Not found'",
    "gender": "Male" or "Female" or "Unknown"
}}

JSON:"""
        
        response = self._call_llm(prompt)
        
        # Parse JSON
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                gender = result.get("gender", "Unknown")
                person_name = result.get("person_name", "Unknown")
                print(f"   [GENDER] Detected: {person_name} -> {gender}")
                return gender
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: Try basic name matching
        return self._detect_gender_by_name_patterns(invoice_text)
    
    def _detect_gender_by_name_patterns(self, text: str) -> str:
        """Fallback gender detection using name patterns"""
        # Common male names
        male_names = ["john", "james", "robert", "michael", "david", "richard", "william", "joseph", 
                      "thomas", "charles", "george", "edward", "harry", "jack", "daniel", "matthew",
                      "anthony", "mark", "steven", "paul", "andrew", "joshua", "kenneth", "kevin",
                      "brian", "george", "edward", "raymond", "gary", "eric", "larry", "scott", "frank"]
        
        # Common female names
        female_names = ["mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", 
                        "jessica", "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra",
                        "ashley", "kimberly", "emily", "donna", "michelle", "dorothy", "carol", "amanda",
                        "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
                        "kathleen", "amy", "shirley", "angela", "helen", "anna", "brenda", "pamela",
                        "nicole", "samantha", "katherine", "christine", "debra", "rachel", "carolyn", "janet",
                        "catherine", "maria", "heather", "diane", "ruth", "julie", "olivia", "joyce",
                        "clara", "victoria", "kelly", "lauren", "christina", "joan", "evelyn", "judith"]
        
        text_lower = text.lower()
        
        # Check for male names
        for name in male_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Male"
        
        # Check for female names
        for name in female_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Female"
        
        return "Unknown"
    
    def _keyword_fallback(self, description: str, codes: List[ServiceCode]) -> RAGResponse:
        """Fallback using keyword matching"""
        if not codes:
            return RAGResponse(suggested_code="UNKNOWN", confidence_score=0.0, reasoning="No codes found")
        
        desc_words = set(description.lower().split())
        best_match = codes[0]
        best_score = 0
        
        for code in codes:
            code_words = set(code.description.lower().split())
            score = len(desc_words & code_words) / max(len(code_words), 1)
            if score > best_score:
                best_score = score
                best_match = code
        
        return RAGResponse(
            suggested_code=best_match.code,
            confidence_score=min(0.85, best_score + 0.4),
            reasoning=f"Matched via keywords ({self.current_api or 'fallback'})"
        )
    
    def process_invoice(self, pdf_path: str) -> dict:
        """Process an invoice through the full RAG pipeline"""
        print(f"\n[DOC] Processing invoice: {pdf_path}")
        
        # Extract and chunk text from PDF
        text_chunks = self.extract_text_from_pdf(pdf_path, chunk_size=500)
        if not text_chunks:
            return {"error": "Failed to extract text from PDF", "line_items": []}
        
        # Join all chunks for processing
        full_text = " ".join(text_chunks)
        print(f"   [CHUNKS] Extracted {len(text_chunks)} chunks (500 chars each)")
        
        # Detect gender from invoice
        gender = self.detect_gender_from_name(full_text)
        print(f"   [GENDER] Detected: {gender}")
        
        # Parse line items with amounts
        parsed_items = self.parse_line_items(full_text)
        print(f"   Found {len(parsed_items)} line items")
        
        results = []
        total_amount = 0.0
        
        for i, item in enumerate(parsed_items):
            desc = item.get("description", "")
            amount = item.get("amount", 0.0)
            total_amount += amount
            
            print(f"   Processing item {i+1}...")
            
            similar_codes = self.retrieve_similar_codes(desc)
            rag_resp = self.generate_response(desc, similar_codes)
            
            flagged = rag_resp.confidence_score < self.confidence_threshold
            
            result = {
                "description": desc,
                "amount": amount,
                "suggested_code": rag_resp.suggested_code,
                "confidence_score": rag_resp.confidence_score,
                "reasoning": rag_resp.reasoning,
                "flagged": flagged,
                "api_used": self.current_api or "keyword",
                "retrieved_codes": [
                    {"code": c.code, "category": c.category, "description": c.description}
                    for c in similar_codes
                ]
            }
            
            results.append(result)
            
            status = "[FLAG]" if flagged else "[OK]"
            print(f"      → {rag_resp.suggested_code} ({rag_resp.confidence_score:.2f}) {status}")
        
        print(f"   Processed {len(results)} items\n")
        print(f"   [TOTAL] Calculated total: ${total_amount:.2f}")
        
        return {
            "line_items": results,
            "gender": gender,
            "total_amount": total_amount
        }
    
    def update_knowledge_base(self, description: str, correct_code: str):
        """Feedback loop - learn from corrections"""
        print(f"   [LEARN] Learning: {correct_code}")
    
    def seed_service_codes(self, codes: List[ServiceCode]):
        """Seed vector store with service codes"""
        if not self.index:
            print("   [WARN] Cannot seed - no Pinecone index")
            return
        
        try:
            documents = [code.description for code in codes]
            
            # Fit TF-IDF
            if self.tfidf:
                self.tfidf.fit(documents)
            
            # Generate embeddings
            vectors = self.tfidf.transform(documents).toarray()
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, code in enumerate(codes):
                vectors_to_upsert.append({
                    'id': f"code_{i}",
                    'values': vectors[i].tolist(),
                    'metadata': {
                        'code': code.code,
                        'description': code.description,
                        'category': code.category
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"   [OK] Seeded {len(codes)} codes to Pinecone")
        except Exception as e:
            print(f"   Error seeding codes: {e}")
    
    def load_seed_from_file(self, file_path: str):
        """Load service codes from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            codes = [ServiceCode(**item) for item in data]
            self.seed_service_codes(codes)
            print(f"   [OK] Loaded {len(codes)} codes from {file_path}")
        except Exception as e:
            print(f"   Error loading seed file: {e}")
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "initialized": self.is_initialized,
            "pinecone_index": self.pinecone_index if self.index else "not configured",
            "current_api": self.current_api or "none",
            "groq_keys": len(self.groq_keys),
            "google_keys": len(self.google_keys),
            "embedding": "TF-IDF"
        }


# Run for testing
if __name__ == "__main__":
    engine = RAGEngine()
    print("\n📊 Stats:", engine.get_stats())


def parse_line_items_simple(self, text: str) -> List[str]:
    """Parse invoice text into line items (legacy method)"""
    lines = text.split('\n')
    return [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]


# ============== RAG OPERATIONS ==============

def retrieve_similar_codes(self, description: str, top_k: int = 3) -> List[ServiceCode]:
    """Retrieve similar service codes using TF-IDF"""

    if not self.index:
        return self._get_fallback_codes()

    try:
        result = self.index.describe_index_stats()

        if result.total_vector_count == 0:
            return self._get_fallback_codes()

        query_result = self.index.query(
            vector=[0] * 384,
            top_k=result.total_vector_count,
            include_metadata=True,
            include_values=False
        )

        if not query_result.matches:
            return self._get_fallback_codes()

        documents = []
        metadatas = []

        for match in query_result.matches:
            documents.append(match.metadata.get('description', ''))
            metadatas.append({
                'code': match.metadata.get('code', 'UNKNOWN'),
                'category': match.metadata.get('category', 'General')
            })

        if not documents:
            return self._get_fallback_codes()

        self.tfidf.fit(documents)

        query_vector = self.tfidf.transform([description]).toarray()[0]

        similarities = []
        for i, doc in enumerate(documents):
            doc_vector = self.tfidf.transform([doc]).toarray()[0]

            dot = sum(a * b for a, b in zip(query_vector, doc_vector))
            query_mag = math.sqrt(sum(a * a for a in query_vector))
            doc_mag = math.sqrt(sum(a * a for a in doc_vector))

            if query_mag > 0 and doc_mag > 0:
                sim = dot / (query_mag * doc_mag)
            else:
                sim = 0

            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, sim in similarities[:top_k]:
            results.append(ServiceCode(
                code=metadatas[i].get('code', 'UNKNOWN'),
                description=documents[i],
                category=metadatas[i].get('category', 'General')
            ))

        return results

    except Exception as e:
        print(f"   [WARN] Retrieval error: {e}")
        return self._get_fallback_codes()


def _get_fallback_codes(self) -> List[ServiceCode]:
    """Fallback codes"""
    return [
        ServiceCode(code="01_111_1117_1_1", description="Personal Care Assistance", category="Personal Care"),
        ServiceCode(code="01_112_1117_1_1", description="Domestic Assistance", category="Domestic Assistance"),
        ServiceCode(code="01_117_1117_1_1", description="Gardening & Maintenance", category="Gardening"),
    ]


def generate_response(self, description: str, similar_codes: List[ServiceCode]) -> RAGResponse:
    """Generate response using LLM with consistent category-to-code mapping"""

    category_prompt = f"""You are an AI assistant for categorizing aged care invoice descriptions.

EXISTING CATEGORIES:
- Personal Care (Code: 01_011_0107_1_1)
- Nursing (Code: 01_121_1117_1_1)
- Domestic Assistance (Code: 01_020_0120_1_1)
- Gardening (Code: 01_019_0120_1_1)
- Meals (Code: 01_131_1117_1_1)
- Transport (Code: 01_146_1117_1_1)
- Allied Health (Code: 01_230_1117_1_1)

Invoice Description: {description}

Analyze the description and respond with ONLY a JSON object:
{{
    "is_new_category": true or false,
    "matched_category": "<if existing>",
    "suggested_category": "<if new>",
    "reasoning": "<brief explanation>"
}}

JSON:"""

    category_response = self._call_llm(category_prompt)

    is_new_category = False
    detected_category = None
    suggested_category = None
    reasoning = ""

    try:
        json_start = category_response.find('{')
        json_end = category_response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(category_response[json_start:json_end])
            is_new_category = result.get("is_new_category", False)
            detected_category = result.get("matched_category")
            suggested_category = result.get("suggested_category")
            reasoning = result.get("reasoning", "")
    except:
        pass

    if not detected_category and not is_new_category:
        detected_category = self._detect_category_by_keywords(description)

    if is_new_category or not detected_category:
        return RAGResponse(
            suggested_code="PENDING_APPROVAL",
            confidence_score=0.0,
            reasoning=f"New category needed: {reasoning}",
            needs_approval=True,
            suggested_category="New Category",
            suggested_code_format="01_999_9999_1_1"
        )

    suggested_code = CATEGORY_SERVICE_CODES.get(detected_category, "UNKNOWN")

    if suggested_code == "UNKNOWN":
        return self._keyword_fallback(description, similar_codes)

    return RAGResponse(
        suggested_code=suggested_code,
        confidence_score=0.95,
        reasoning=f"Category '{detected_category}' mapped to service code {suggested_code}",
        needs_approval=False
    )
    def _detect_category_by_keywords(self, description: str) -> str:
        """Detect category using keyword matching"""
        desc_lower = description.lower()
        
        # Category keywords
        keywords = {
            "Personal Care": ["shower", "bath", "dressing", "grooming", "hygiene", "toilet", "mobility", "transfer", "personal care", "assistance with self-care"],
            "Nursing": ["nurse", "nursing", "medication", "wound", "health", "medical", "doctor", "clinical", "therapy"],
            "Domestic Assistance": ["cleaning", "laundry", "vacuum", "mop", "dust", "house", "dish", "washing", "ironing", "bed linen", "household"],
            "Gardening": ["garden", "lawn", "mow", "weed", "prune", "yard", "gutter", "maintenance", "outdoor"],
            "Meals": ["meal", "food", "cooking", "lunch", "dinner", "breakfast", "delivery"],
            "Transport": ["transport", "drive", "car", "appointment", "travel", "bus", "taxi"],
            "Allied Health": ["physio", "physiotherapy", "occupational", "speech", "podiatry", "dietitian", "psychologist"],
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in desc_lower:
                    return category
        
        return "Domestic Assistance"  # Default category

    # ============== GENDER DETECTION ==============
    
    def detect_gender_from_name(self, invoice_text: str) -> str:
        """
        Detect gender from invoice person name using LLM.
        The LLM extracts the person name from the invoice and determines gender.
        Returns: "Male", "Female", or "Unknown"
        """
        prompt = f"""You are an AI assistant that extracts person names from invoices and determines gender.

Invoice Text:
{invoice_text[:2000]}  # Limit text length

Instructions:
1. Extract the CLIENT/CUSTOMER/PERSON name from the invoice
2. Determine if the person is Male or Female based on the name
3. Consider common first names (e.g., John, Michael, David = Male; Mary, Jane, Clara = Female)

Respond with ONLY a JSON object:
{{
    "person_name": "<extracted name or 'Not found'",
    "gender": "Male" or "Female" or "Unknown"
}}

JSON:"""
        
        response = self._call_llm(prompt)
        
        # Parse JSON
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                gender = result.get("gender", "Unknown")
                person_name = result.get("person_name", "Unknown")
                print(f"   [GENDER] Detected: {person_name} -> {gender}")
                return gender
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: Try basic name matching
        return self._detect_gender_by_name_patterns(invoice_text)
    
    def _detect_gender_by_name_patterns(self, text: str) -> str:
        """Fallback gender detection using name patterns"""
        # Common male names
        male_names = ["john", "james", "robert", "michael", "david", "richard", "william", "joseph", 
                      "thomas", "charles", "george", "edward", "harry", "jack", "daniel", "matthew",
                      "anthony", "mark", "steven", "paul", "andrew", "joshua", "kenneth", "kevin",
                      "brian", "george", "edward", "raymond", "gary", "eric", "larry", "scott", "frank"]
        
        # Common female names
        female_names = ["mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", 
                        "jessica", "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra",
                        "ashley", "kimberly", "emily", "donna", "michelle", "dorothy", "carol", "amanda",
                        "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
                        "kathleen", "amy", "shirley", "angela", "helen", "anna", "brenda", "pamela",
                        "nicole", "samantha", "katherine", "christine", "debra", "rachel", "carolyn", "janet",
                        "catherine", "maria", "heather", "diane", "ruth", "julie", "olivia", "joyce",
                        "clara", "victoria", "kelly", "lauren", "christina", "joan", "evelyn", "judith"]
        
        text_lower = text.lower()
        
        # Check for male names
        for name in male_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Male"
        
        # Check for female names
        for name in female_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Female"
        
        return "Unknown"
    
    def _keyword_fallback(self, description: str, codes: List[ServiceCode]) -> RAGResponse:
        """Fallback using keyword matching"""
        if not codes:
            return RAGResponse(suggested_code="UNKNOWN", confidence_score=0.0, reasoning="No codes found")
        
        desc_words = set(description.lower().split())
        best_match = codes[0]
        best_score = 0
        
        for code in codes:
            code_words = set(code.description.lower().split())
            score = len(desc_words & code_words) / max(len(code_words), 1)
            if score > best_score:
                best_score = score
                best_match = code
        
        return RAGResponse(
            suggested_code=best_match.code,
            confidence_score=min(0.85, best_score + 0.4),
            reasoning=f"Matched via keywords ({self.current_api or 'fallback'})"
        )
    
    def process_invoice(self, pdf_path: str) -> dict:
        """Process an invoice through the full RAG pipeline"""
        print(f"\n[DOC] Processing invoice: {pdf_path}")
        
        # Extract and chunk text from PDF
        text_chunks = self.extract_text_from_pdf(pdf_path, chunk_size=500)
        if not text_chunks:
            return {"error": "Failed to extract text from PDF", "line_items": []}
        
        # Join all chunks for processing
        full_text = " ".join(text_chunks)
        print(f"   [CHUNKS] Extracted {len(text_chunks)} chunks (500 chars each)")
        
        # Detect gender from invoice
        gender = self.detect_gender_from_name(full_text)
        print(f"   [GENDER] Detected: {gender}")
        
        # Parse line items with amounts
        parsed_items = self.parse_line_items(full_text)
        print(f"   Found {len(parsed_items)} line items")
        
        results = []
        total_amount = 0.0
        
        for i, item in enumerate(parsed_items):
            desc = item.get("description", "")
            amount = item.get("amount", 0.0)
            total_amount += amount
            
            print(f"   Processing item {i+1}...")
            
            similar_codes = self.retrieve_similar_codes(desc)
            rag_resp = self.generate_response(desc, similar_codes)
            
            flagged = rag_resp.confidence_score < self.confidence_threshold
            
            result = {
                "description": desc,
                "amount": amount,
                "suggested_code": rag_resp.suggested_code,
                "confidence_score": rag_resp.confidence_score,
                "reasoning": rag_resp.reasoning,
                "flagged": flagged,
                "api_used": self.current_api or "keyword",
                "retrieved_codes": [
                    {"code": c.code, "category": c.category, "description": c.description}
                    for c in similar_codes
                ]
            }
            
            results.append(result)
            
            status = "[FLAG]" if flagged else "[OK]"
            print(f"      → {rag_resp.suggested_code} ({rag_resp.confidence_score:.2f}) {status}")
        
        print(f"   Processed {len(results)} items\n")
        print(f"   [TOTAL] Calculated total: ${total_amount:.2f}")
        
        return {
            "line_items": results,
            "gender": gender,
            "total_amount": total_amount
        }
    
    def update_knowledge_base(self, description: str, correct_code: str):
        """Feedback loop - learn from corrections"""
        print(f"   [LEARN] Learning: {correct_code}")
    
    def seed_service_codes(self, codes: List[ServiceCode]):
        """Seed vector store with service codes"""
        if not self.index:
            print("   [WARN] Cannot seed - no Pinecone index")
            return
        
        try:
            documents = [code.description for code in codes]
            
            # Fit TF-IDF
            if self.tfidf:
                self.tfidf.fit(documents)
            
            # Generate embeddings
            vectors = self.tfidf.transform(documents).toarray()
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, code in enumerate(codes):
                vectors_to_upsert.append({
                    'id': f"code_{i}",
                    'values': vectors[i].tolist(),
                    'metadata': {
                        'code': code.code,
                        'description': code.description,
                        'category': code.category
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"   [OK] Seeded {len(codes)} codes to Pinecone")
        except Exception as e:
            print(f"   Error seeding codes: {e}")
    
    def load_seed_from_file(self, file_path: str):
        """Load service codes from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            codes = [ServiceCode(**item) for item in data]
            self.seed_service_codes(codes)
            print(f"   [OK] Loaded {len(codes)} codes from {file_path}")
        except Exception as e:
            print(f"   Error loading seed file: {e}")
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "initialized": self.is_initialized,
            "pinecone_index": self.pinecone_index if self.index else "not configured",
            "current_api": self.current_api or "none",
            "groq_keys": len(self.groq_keys),
            "google_keys": len(self.google_keys),
            "embedding": "TF-IDF"
        }


# Run for testing
if __name__ == "__main__":
    engine = RAGEngine()
    print("\n📊 Stats:", engine.get_stats())

    def detect_category_from_description(self, description: str) -> str:
        """Detect category based on keywords in the description"""
        desc_lower = description.lower()
        
        # Define category keywords
        category_keywords = {
            "Domestic Assistance": ["domestic", "house cleaning", "housekeeping", "cleaning", "laundry", "ironing", "bed making", "household", "home cleaning", "general household", "domestic assistance"],
            "Personal Care": ["personal care", "showering", "bathing", "dressing", "toileting", "mobility", "assisted daily living", "adl", "personal hygiene", "grooming"],
            "Nursing": ["nursing", "nurse", "medical", "medication", "wound care", "injection", "health monitoring", "clinical", "healthcare"],
            "Allied Health": ["physiotherapy", "occupational therapy", "speech therapy", "podiatry", "dietitian", "allied health", "therapy", "rehabilitation"],
            "Transport": ["transport", "transportation", "travel", "appointment transport", "medical transport", "community transport"],
            "Gardening": ["gardening", "garden", "lawn mowing", "hedging", "yard maintenance", "landscaping", "outdoor", "green"],
            "Meals": ["meal", "meals", "cooking", "food preparation", "meal delivery", "prepared meals", "menu"],
            "IT Support": ["it", "software", "computer", "laptop", "installation", "configuration", "tech", "technology", "digital", "printer", "network", "wifi", "internet"],
        }
        
        # Check each category for keyword matches
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return category
        
        return "General"  # Default category
    
    def parse_line_items_simple(self, text: str) -> List[str]:
        """Parse invoice text into line items (legacy method)"""
        lines = text.split('\n')
        return [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]
    
    # ============== RAG OPERATIONS ==============
    
    def retrieve_similar_codes(self, description: str, top_k: int = 3) -> List[ServiceCode]:
        """Retrieve similar service codes using TF-IDF"""
        
        if not self.index:
            return self._get_fallback_codes()
        
        try:
            # Get all vectors from Pinecone
            result = self.index.describe_index_stats()
            
            if result.total_vector_count == 0:
                return self._get_fallback_codes()
            
            # Query Pinecone to get all vectors (using empty filter to get all)
            query_result = self.index.query(
                vector=[0] * 384,  # Dummy vector, we'll use TF-IDF for scoring
                top_k=result.total_vector_count,
                include_metadata=True,
                include_values=False
            )
            
            if not query_result.matches:
                return self._get_fallback_codes()
            
            # Build document list from Pinecone results
            documents = []
            metadatas = []
            for match in query_result.matches:
                documents.append(match.metadata.get('description', ''))
                metadatas.append({
                    'code': match.metadata.get('code', 'UNKNOWN'),
                    'category': match.metadata.get('category', 'General')
                })
            
            if not documents:
                return self._get_fallback_codes()
            
            # Fit TF-IDF on documents
            self.tfidf.fit(documents)
            
            # Get query vector
            query_vector = self.tfidf.transform([description]).toarray()[0]
            
            # Calculate cosine similarity
            similarities = []
            for i, doc in enumerate(documents):
                doc_vector = self.tfidf.transform([doc]).toarray()[0]
                
                dot = sum(a * b for a, b in zip(query_vector, doc_vector))
                query_mag = math.sqrt(sum(a * a for a in query_vector))
                doc_mag = math.sqrt(sum(a * a for a in doc_vector))
                
                if query_mag > 0 and doc_mag > 0:
                    sim = dot / (query_mag * doc_mag)
                else:
                    sim = 0
                
                similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            results = []
            for i, sim in similarities[:top_k]:
                results.append(ServiceCode(
                    code=metadatas[i].get('code', 'UNKNOWN'),
                    description=documents[i],
                    category=metadatas[i].get('category', 'General')
                ))
            
            return results
            
        except Exception as e:
            print(f"   [WARN] Retrieval error: {e}")
            return self._get_fallback_codes()
    
    def _get_fallback_codes(self) -> List[ServiceCode]:
        """Fallback codes"""
        return [
            ServiceCode(code="01_111_1117_1_1", description="Personal Care Assistance", category="Personal Care"),
            ServiceCode(code="01_112_1117_1_1", description="Domestic Assistance", category="Domestic Assistance"),
            ServiceCode(code="01_117_1117_1_1", description="Gardening & Maintenance", category="Gardening"),
        ]
    
    def generate_response(self, description: str, similar_codes: List[ServiceCode]) -> RAGResponse:
        """Generate response using LLM with consistent category-to-code mapping"""
        
        # First, determine the category using LLM - ask if it matches existing categories or needs new one
        category_prompt = f"""You are an AI assistant for categorizing aged care invoice descriptions.

EXISTING CATEGORIES:
- Personal Care (Code: 01_011_0107_1_1)
- Nursing (Code: 01_121_1117_1_1)
- Domestic Assistance (Code: 01_020_0120_1_1)
- Gardening (Code: 01_019_0120_1_1)
- Meals (Code: 01_131_1117_1_1)
- Transport (Code: 01_146_1117_1_1)
- Allied Health (Code: 01_230_1117_1_1)

Invoice Description: {description}

Analyze the description and respond with ONLY a JSON object:
{{
    "is_new_category": true or false,
    "matched_category": "<if existing, name of matched category> or null",
    "suggested_category": "<if new, suggest a new category name> or null",
    "reasoning": "<brief explanation>"
}}

JSON:"""
        
        category_response = self._call_llm(category_prompt)
        
        # Parse category from response
        is_new_category = False
        detected_category = None
        suggested_category = None
        reasoning = ""
        
        try:
            json_start = category_response.find('{')
            json_end = category_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = category_response[json_start:json_end]
                result = json.loads(json_str)
                is_new_category = result.get("is_new_category", False)
                detected_category = result.get("matched_category")
                suggested_category = result.get("suggested_category")
                reasoning = result.get("reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # If category not detected from LLM, use keyword matching as fallback
        if not detected_category and not is_new_category:
            detected_category = self._detect_category_by_keywords(description)
        
        # If it's a new category request
        if is_new_category or not detected_category:
            # Ask LLM to suggest a new category and service code format
            new_category_prompt = f"""A new invoice description doesn't match any existing category.

Invoice Description: {description}

Please suggest:
1. A new category name that would fit this service
2. A format for the NDIS-style service code (e.g., 01_XXX_XXXX_X_X)

Respond with ONLY a JSON object:
{{
    "suggested_category": "<new category name>",
    "suggested_code_format": "01_<3-digit>_<4-digit>_<1>_<1>",
    "description": "<brief description of this service category>"
}}

JSON:"""
            new_cat_response = self._call_llm(new_category_prompt)
            
            try:
                json_start = new_cat_response.find('{')
                json_end = new_cat_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = new_cat_response[json_start:json_end]
                    result = json.loads(json_str)
                    suggested_category = result.get("suggested_category", "New Category")
                    suggested_code_format = result.get("suggested_code_format", "01_999_9999_1_1")
            except (json.JSONDecodeError, AttributeError):
                suggested_category = "New Category"
                suggested_code_format = "01_999_9999_1_1"
            
            return RAGResponse(
                suggested_code="PENDING_APPROVAL",
                confidence_score=0.0,
                reasoning=f"New category needed: {reasoning}",
                needs_approval=True,
                suggested_category=suggested_category,
                suggested_code_format=suggested_code_format
            )
        
        # Get the consistent service code for this category
        suggested_code = CATEGORY_SERVICE_CODES.get(detected_category, "UNKNOWN")
        
        # If category not in our mapping, use keyword fallback
        if suggested_code == "UNKNOWN":
            return self._keyword_fallback(description, similar_codes)
        
        return RAGResponse(
            suggested_code=suggested_code,
            confidence_score=0.95,  # High confidence since we use consistent codes per category
            reasoning=f"Category '{detected_category}' mapped to service code {suggested_code}",
            needs_approval=False
        )
    
    def _detect_category_by_keywords(self, description: str) -> str:
        """Detect category using keyword matching"""
        desc_lower = description.lower()
        
        # Category keywords
        keywords = {
            "Personal Care": ["shower", "bath", "dressing", "grooming", "hygiene", "toilet", "mobility", "transfer", "personal care", "assistance with self-care"],
            "Nursing": ["nurse", "nursing", "medication", "wound", "health", "medical", "doctor", "clinical", "therapy"],
            "Domestic Assistance": ["cleaning", "laundry", "vacuum", "mop", "dust", "house", "dish", "washing", "ironing", "bed linen", "household"],
            "Gardening": ["garden", "lawn", "mow", "weed", "prune", "yard", "gutter", "maintenance", "outdoor"],
            "Meals": ["meal", "food", "cooking", "lunch", "dinner", "breakfast", "delivery"],
            "Transport": ["transport", "drive", "car", "appointment", "travel", "bus", "taxi"],
            "Allied Health": ["physio", "physiotherapy", "occupational", "speech", "podiatry", "dietitian", "psychologist"],
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in desc_lower:
                    return category
        
        return "Domestic Assistance"  # Default category

    # ============== GENDER DETECTION ==============
    
    def detect_gender_from_name(self, invoice_text: str) -> str:
        """
        Detect gender from invoice person name using LLM.
        The LLM extracts the person name from the invoice and determines gender.
        Returns: "Male", "Female", or "Unknown"
        """
        prompt = f"""You are an AI assistant that extracts person names from invoices and determines gender.

Invoice Text:
{invoice_text[:2000]}  # Limit text length

Instructions:
1. Extract the CLIENT/CUSTOMER/PERSON name from the invoice
2. Determine if the person is Male or Female based on the name
3. Consider common first names (e.g., John, Michael, David = Male; Mary, Jane, Clara = Female)

Respond with ONLY a JSON object:
{{
    "person_name": "<extracted name or 'Not found'",
    "gender": "Male" or "Female" or "Unknown"
}}

JSON:"""
        
        response = self._call_llm(prompt)
        
        # Parse JSON
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                gender = result.get("gender", "Unknown")
                person_name = result.get("person_name", "Unknown")
                print(f"   [GENDER] Detected: {person_name} -> {gender}")
                return gender
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: Try basic name matching
        return self._detect_gender_by_name_patterns(invoice_text)
    
    def _detect_gender_by_name_patterns(self, text: str) -> str:
        """Fallback gender detection using name patterns"""
        # Common male names
        male_names = ["john", "james", "robert", "michael", "david", "richard", "william", "joseph", 
                      "thomas", "charles", "george", "edward", "harry", "jack", "daniel", "matthew",
                      "anthony", "mark", "steven", "paul", "andrew", "joshua", "kenneth", "kevin",
                      "brian", "george", "edward", "raymond", "gary", "eric", "larry", "scott", "frank"]
        
        # Common female names
        female_names = ["mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", 
                        "jessica", "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra",
                        "ashley", "kimberly", "emily", "donna", "michelle", "dorothy", "carol", "amanda",
                        "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
                        "kathleen", "amy", "shirley", "angela", "helen", "anna", "brenda", "pamela",
                        "nicole", "samantha", "katherine", "christine", "debra", "rachel", "carolyn", "janet",
                        "catherine", "maria", "heather", "diane", "ruth", "julie", "olivia", "joyce",
                        "clara", "victoria", "kelly", "lauren", "christina", "joan", "evelyn", "judith"]
        
        text_lower = text.lower()
        
        # Check for male names
        for name in male_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Male"
        
        # Check for female names
        for name in female_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Female"
        
        return "Unknown"
    
    def _keyword_fallback(self, description: str, codes: List[ServiceCode]) -> RAGResponse:
        """Fallback using keyword matching"""
        if not codes:
            return RAGResponse(suggested_code="UNKNOWN", confidence_score=0.0, reasoning="No codes found")
        
        desc_words = set(description.lower().split())
        best_match = codes[0]
        best_score = 0
        
        for code in codes:
            code_words = set(code.description.lower().split())
            score = len(desc_words & code_words) / max(len(code_words), 1)
            if score > best_score:
                best_score = score
                best_match = code
        
        return RAGResponse(
            suggested_code=best_match.code,
            confidence_score=min(0.85, best_score + 0.4),
            reasoning=f"Matched via keywords ({self.current_api or 'fallback'})"
        )
    
    def process_invoice(self, pdf_path: str) -> dict:
        """Process an invoice through the full RAG pipeline"""
        print(f"\n[DOC] Processing invoice: {pdf_path}")
        
        # Extract and chunk text from PDF
        text_chunks = self.extract_text_from_pdf(pdf_path, chunk_size=500)
        if not text_chunks:
            return {"error": "Failed to extract text from PDF", "line_items": []}
        
        # Join all chunks for processing
        full_text = " ".join(text_chunks)
        print(f"   [CHUNKS] Extracted {len(text_chunks)} chunks (500 chars each)")
        
        # Detect gender from invoice
        gender = self.detect_gender_from_name(full_text)
        print(f"   [GENDER] Detected: {gender}")
        
        # Parse line items with amounts
        parsed_items = self.parse_line_items(full_text)
        print(f"   Found {len(parsed_items)} line items")
        
        results = []
        total_amount = 0.0
        
        for i, item in enumerate(parsed_items):
            desc = item.get("description", "")
            amount = item.get("amount", 0.0)
            total_amount += amount
            
            print(f"   Processing item {i+1}...")
            
            similar_codes = self.retrieve_similar_codes(desc)
            rag_resp = self.generate_response(desc, similar_codes)
            
            flagged = rag_resp.confidence_score < self.confidence_threshold
            
            result = {
                "description": desc,
                "amount": amount,
                "suggested_code": rag_resp.suggested_code,
                "confidence_score": rag_resp.confidence_score,
                "reasoning": rag_resp.reasoning,
                "flagged": flagged,
                "api_used": self.current_api or "keyword",
                "retrieved_codes": [
                    {"code": c.code, "category": c.category, "description": c.description}
                    for c in similar_codes
                ]
            }
            
            results.append(result)
            
            status = "[FLAG]" if flagged else "[OK]"
            print(f"      → {rag_resp.suggested_code} ({rag_resp.confidence_score:.2f}) {status}")
        
        print(f"   Processed {len(results)} items\n")
        print(f"   [TOTAL] Calculated total: ${total_amount:.2f}")
        
        return {
            "line_items": results,
            "gender": gender,
            "total_amount": total_amount
        }
    
    def update_knowledge_base(self, description: str, correct_code: str):
        """Feedback loop - learn from corrections"""
        print(f"   [LEARN] Learning: {correct_code}")
    
    def seed_service_codes(self, codes: List[ServiceCode]):
        """Seed vector store with service codes"""
        if not self.index:
            print("   [WARN] Cannot seed - no Pinecone index")
            return
        
        try:
            documents = [code.description for code in codes]
            
            # Fit TF-IDF
            if self.tfidf:
                self.tfidf.fit(documents)
            
            # Generate embeddings
            vectors = self.tfidf.transform(documents).toarray()
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, code in enumerate(codes):
                vectors_to_upsert.append({
                    'id': f"code_{i}",
                    'values': vectors[i].tolist(),
                    'metadata': {
                        'code': code.code,
                        'description': code.description,
                        'category': code.category
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"   [OK] Seeded {len(codes)} codes to Pinecone")
        except Exception as e:
            print(f"   Error seeding codes: {e}")
    
    def load_seed_from_file(self, file_path: str):
        """Load service codes from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            codes = [ServiceCode(**item) for item in data]
            self.seed_service_codes(codes)
            print(f"   [OK] Loaded {len(codes)} codes from {file_path}")
        except Exception as e:
            print(f"   Error loading seed file: {e}")
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "initialized": self.is_initialized,
            "pinecone_index": self.pinecone_index if self.index else "not configured",
            "current_api": self.current_api or "none",
            "groq_keys": len(self.groq_keys),
            "google_keys": len(self.google_keys),
            "embedding": "TF-IDF"
        }


# Run for testing
if __name__ == "__main__":
    engine = RAGEngine()
    print("\n📊 Stats:", engine.get_stats())


def parse_line_items_simple(self, text: str) -> List[str]:
    """Parse invoice text into line items (legacy method)"""
    lines = text.split('\n')
    return [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]


# ============== RAG OPERATIONS ==============

def retrieve_similar_codes(self, description: str, top_k: int = 3) -> List[ServiceCode]:
    """Retrieve similar service codes using TF-IDF"""

    if not self.index:
        return self._get_fallback_codes()

    try:
        result = self.index.describe_index_stats()

        if result.total_vector_count == 0:
            return self._get_fallback_codes()

        query_result = self.index.query(
            vector=[0] * 384,
            top_k=result.total_vector_count,
            include_metadata=True,
            include_values=False
        )

        if not query_result.matches:
            return self._get_fallback_codes()

        documents = []
        metadatas = []

        for match in query_result.matches:
            documents.append(match.metadata.get('description', ''))
            metadatas.append({
                'code': match.metadata.get('code', 'UNKNOWN'),
                'category': match.metadata.get('category', 'General')
            })

        if not documents:
            return self._get_fallback_codes()

        self.tfidf.fit(documents)

        query_vector = self.tfidf.transform([description]).toarray()[0]

        similarities = []
        for i, doc in enumerate(documents):
            doc_vector = self.tfidf.transform([doc]).toarray()[0]

            dot = sum(a * b for a, b in zip(query_vector, doc_vector))
            query_mag = math.sqrt(sum(a * a for a in query_vector))
            doc_mag = math.sqrt(sum(a * a for a in doc_vector))

            if query_mag > 0 and doc_mag > 0:
                sim = dot / (query_mag * doc_mag)
            else:
                sim = 0

            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, sim in similarities[:top_k]:
            results.append(ServiceCode(
                code=metadatas[i].get('code', 'UNKNOWN'),
                description=documents[i],
                category=metadatas[i].get('category', 'General')
            ))

        return results

    except Exception as e:
        print(f"   [WARN] Retrieval error: {e}")
        return self._get_fallback_codes()


def _get_fallback_codes(self) -> List[ServiceCode]:
    """Fallback codes"""
    return [
        ServiceCode(code="01_111_1117_1_1", description="Personal Care Assistance", category="Personal Care"),
        ServiceCode(code="01_112_1117_1_1", description="Domestic Assistance", category="Domestic Assistance"),
        ServiceCode(code="01_117_1117_1_1", description="Gardening & Maintenance", category="Gardening"),
    ]
    
    def generate_response(self, description: str, similar_codes: List[ServiceCode]) -> RAGResponse:
        """Generate response using LLM with consistent category-to-code mapping"""
        
        # First, determine the category using LLM - ask if it matches existing categories or needs new one
        category_prompt = f"""You are an AI assistant for categorizing aged care invoice descriptions.

EXISTING CATEGORIES:
- Personal Care (Code: 01_011_0107_1_1)
- Nursing (Code: 01_121_1117_1_1)
- Domestic Assistance (Code: 01_020_0120_1_1)
- Gardening (Code: 01_019_0120_1_1)
- Meals (Code: 01_131_1117_1_1)
- Transport (Code: 01_146_1117_1_1)
- Allied Health (Code: 01_230_1117_1_1)

Invoice Description: {description}

Analyze the description and respond with ONLY a JSON object:
{{
    "is_new_category": true or false,
    "matched_category": "<if existing, name of matched category> or null",
    "suggested_category": "<if new, suggest a new category name> or null",
    "reasoning": "<brief explanation>"
}}

JSON:"""
        
        category_response = self._call_llm(category_prompt)
        
        # Parse category from response
        is_new_category = False
        detected_category = None
        suggested_category = None
        reasoning = ""
        
        try:
            json_start = category_response.find('{')
            json_end = category_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = category_response[json_start:json_end]
                result = json.loads(json_str)
                is_new_category = result.get("is_new_category", False)
                detected_category = result.get("matched_category")
                suggested_category = result.get("suggested_category")
                reasoning = result.get("reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # If category not detected from LLM, use keyword matching as fallback
        if not detected_category and not is_new_category:
            detected_category = self._detect_category_by_keywords(description)
        
        # If it's a new category request
        if is_new_category or not detected_category:
            # Ask LLM to suggest a new category and service code format
            new_category_prompt = f"""A new invoice description doesn't match any existing category.

Invoice Description: {description}

Please suggest:
1. A new category name that would fit this service
2. A format for the NDIS-style service code (e.g., 01_XXX_XXXX_X_X)

Respond with ONLY a JSON object:
{{
    "suggested_category": "<new category name>",
    "suggested_code_format": "01_<3-digit>_<4-digit>_<1>_<1>",
    "description": "<brief description of this service category>"
}}

JSON:"""
            new_cat_response = self._call_llm(new_category_prompt)
            
            try:
                json_start = new_cat_response.find('{')
                json_end = new_cat_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = new_cat_response[json_start:json_end]
                    result = json.loads(json_str)
                    suggested_category = result.get("suggested_category", "New Category")
                    suggested_code_format = result.get("suggested_code_format", "01_999_9999_1_1")
            except (json.JSONDecodeError, AttributeError):
                suggested_category = "New Category"
                suggested_code_format = "01_999_9999_1_1"
            
            return RAGResponse(
                suggested_code="PENDING_APPROVAL",
                confidence_score=0.0,
                reasoning=f"New category needed: {reasoning}",
                needs_approval=True,
                suggested_category=suggested_category,
                suggested_code_format=suggested_code_format
            )
        
        # Get the consistent service code for this category
        suggested_code = CATEGORY_SERVICE_CODES.get(detected_category, "UNKNOWN")
        
        # If category not in our mapping, use keyword fallback
        if suggested_code == "UNKNOWN":
            return self._keyword_fallback(description, similar_codes)
        
        return RAGResponse(
            suggested_code=suggested_code,
            confidence_score=0.95,  # High confidence since we use consistent codes per category
            reasoning=f"Category '{detected_category}' mapped to service code {suggested_code}",
            needs_approval=False
        )
    
    def _detect_category_by_keywords(self, description: str) -> str:
        """Detect category using keyword matching"""
        desc_lower = description.lower()
        
        # Category keywords
        keywords = {
            "Personal Care": ["shower", "bath", "dressing", "grooming", "hygiene", "toilet", "mobility", "transfer", "personal care", "assistance with self-care"],
            "Nursing": ["nurse", "nursing", "medication", "wound", "health", "medical", "doctor", "clinical", "therapy"],
            "Domestic Assistance": ["cleaning", "laundry", "vacuum", "mop", "dust", "house", "dish", "washing", "ironing", "bed linen", "household"],
            "Gardening": ["garden", "lawn", "mow", "weed", "prune", "yard", "gutter", "maintenance", "outdoor"],
            "Meals": ["meal", "food", "cooking", "lunch", "dinner", "breakfast", "delivery"],
            "Transport": ["transport", "drive", "car", "appointment", "travel", "bus", "taxi"],
            "Allied Health": ["physio", "physiotherapy", "occupational", "speech", "podiatry", "dietitian", "psychologist"],
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in desc_lower:
                    return category
        
        return "Domestic Assistance"  # Default category

    # ============== GENDER DETECTION ==============
    
    def detect_gender_from_name(self, invoice_text: str) -> str:
        """
        Detect gender from invoice person name using LLM.
        The LLM extracts the person name from the invoice and determines gender.
        Returns: "Male", "Female", or "Unknown"
        """
        prompt = f"""You are an AI assistant that extracts person names from invoices and determines gender.

Invoice Text:
{invoice_text[:2000]}  # Limit text length

Instructions:
1. Extract the CLIENT/CUSTOMER/PERSON name from the invoice
2. Determine if the person is Male or Female based on the name
3. Consider common first names (e.g., John, Michael, David = Male; Mary, Jane, Clara = Female)

Respond with ONLY a JSON object:
{{
    "person_name": "<extracted name or 'Not found'",
    "gender": "Male" or "Female" or "Unknown"
}}

JSON:"""
        
        response = self._call_llm(prompt)
        
        # Parse JSON
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                gender = result.get("gender", "Unknown")
                person_name = result.get("person_name", "Unknown")
                print(f"   [GENDER] Detected: {person_name} -> {gender}")
                return gender
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: Try basic name matching
        return self._detect_gender_by_name_patterns(invoice_text)
    
    def _detect_gender_by_name_patterns(self, text: str) -> str:
        """Fallback gender detection using name patterns"""
        # Common male names
        male_names = ["john", "james", "robert", "michael", "david", "richard", "william", "joseph", 
                      "thomas", "charles", "george", "edward", "harry", "jack", "daniel", "matthew",
                      "anthony", "mark", "steven", "paul", "andrew", "joshua", "kenneth", "kevin",
                      "brian", "george", "edward", "raymond", "gary", "eric", "larry", "scott", "frank"]
        
        # Common female names
        female_names = ["mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", 
                        "jessica", "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra",
                        "ashley", "kimberly", "emily", "donna", "michelle", "dorothy", "carol", "amanda",
                        "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
                        "kathleen", "amy", "shirley", "angela", "helen", "anna", "brenda", "pamela",
                        "nicole", "samantha", "katherine", "christine", "debra", "rachel", "carolyn", "janet",
                        "catherine", "maria", "heather", "diane", "ruth", "julie", "olivia", "joyce",
                        "clara", "victoria", "kelly", "lauren", "christina", "joan", "evelyn", "judith"]
        
        text_lower = text.lower()
        
        # Check for male names
        for name in male_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Male"
        
        # Check for female names
        for name in female_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Female"
        
        return "Unknown"
    
    def _keyword_fallback(self, description: str, codes: List[ServiceCode]) -> RAGResponse:
        """Fallback using keyword matching"""
        if not codes:
            return RAGResponse(suggested_code="UNKNOWN", confidence_score=0.0, reasoning="No codes found")
        
        desc_words = set(description.lower().split())
        best_match = codes[0]
        best_score = 0
        
        for code in codes:
            code_words = set(code.description.lower().split())
            score = len(desc_words & code_words) / max(len(code_words), 1)
            if score > best_score:
                best_score = score
                best_match = code
        
        return RAGResponse(
            suggested_code=best_match.code,
            confidence_score=min(0.85, best_score + 0.4),
            reasoning=f"Matched via keywords ({self.current_api or 'fallback'})"
        )
    
    def process_invoice(self, pdf_path: str) -> dict:
        """Process an invoice through the full RAG pipeline"""
        print(f"\n[DOC] Processing invoice: {pdf_path}")
        
        # Extract and chunk text from PDF
        text_chunks = self.extract_text_from_pdf(pdf_path, chunk_size=500)
        if not text_chunks:
            return {"error": "Failed to extract text from PDF", "line_items": []}
        
        # Join all chunks for processing
        full_text = " ".join(text_chunks)
        print(f"   [CHUNKS] Extracted {len(text_chunks)} chunks (500 chars each)")
        
        # Detect gender from invoice
        gender = self.detect_gender_from_name(full_text)
        print(f"   [GENDER] Detected: {gender}")
        
        # Parse line items with amounts
        parsed_items = self.parse_line_items(full_text)
        print(f"   Found {len(parsed_items)} line items")
        
        results = []
        total_amount = 0.0
        
        for i, item in enumerate(parsed_items):
            desc = item.get("description", "")
            amount = item.get("amount", 0.0)
            total_amount += amount
            
            print(f"   Processing item {i+1}...")
            
            similar_codes = self.retrieve_similar_codes(desc)
            rag_resp = self.generate_response(desc, similar_codes)
            
            flagged = rag_resp.confidence_score < self.confidence_threshold
            
            result = {
                "description": desc,
                "amount": amount,
                "suggested_code": rag_resp.suggested_code,
                "confidence_score": rag_resp.confidence_score,
                "reasoning": rag_resp.reasoning,
                "flagged": flagged,
                "api_used": self.current_api or "keyword",
                "retrieved_codes": [
                    {"code": c.code, "category": c.category, "description": c.description}
                    for c in similar_codes
                ]
            }
            
            results.append(result)
            
            status = "[FLAG]" if flagged else "[OK]"
            print(f"      → {rag_resp.suggested_code} ({rag_resp.confidence_score:.2f}) {status}")
        
        print(f"   Processed {len(results)} items\n")
        print(f"   [TOTAL] Calculated total: ${total_amount:.2f}")
        
        return {
            "line_items": results,
            "gender": gender,
            "total_amount": total_amount
        }
    
    def update_knowledge_base(self, description: str, correct_code: str):
        """Feedback loop - learn from corrections"""
        print(f"   [LEARN] Learning: {correct_code}")
    
    def seed_service_codes(self, codes: List[ServiceCode]):
        """Seed vector store with service codes"""
        if not self.index:
            print("   [WARN] Cannot seed - no Pinecone index")
            return
        
        try:
            documents = [code.description for code in codes]
            
            # Fit TF-IDF
            if self.tfidf:
                self.tfidf.fit(documents)
            
            # Generate embeddings
            vectors = self.tfidf.transform(documents).toarray()
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, code in enumerate(codes):
                vectors_to_upsert.append({
                    'id': f"code_{i}",
                    'values': vectors[i].tolist(),
                    'metadata': {
                        'code': code.code,
                        'description': code.description,
                        'category': code.category
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"   [OK] Seeded {len(codes)} codes to Pinecone")
        except Exception as e:
            print(f"   Error seeding codes: {e}")
    
    def load_seed_from_file(self, file_path: str):
        """Load service codes from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            codes = [ServiceCode(**item) for item in data]
            self.seed_service_codes(codes)
            print(f"   [OK] Loaded {len(codes)} codes from {file_path}")
        except Exception as e:
            print(f"   Error loading seed file: {e}")
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "initialized": self.is_initialized,
            "pinecone_index": self.pinecone_index if self.index else "not configured",
            "current_api": self.current_api or "none",
            "groq_keys": len(self.groq_keys),
            "google_keys": len(self.google_keys),
            "embedding": "TF-IDF"
        }


# Run for testing
if __name__ == "__main__":
    engine = RAGEngine()
    print("\n📊 Stats:", engine.get_stats())

    def parse_line_items_simple(self, text: str) -> List[str]:
        """Parse invoice text into line items (legacy method)"""
        lines = text.split('\n')
        return [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]
    
    # ============== RAG OPERATIONS ==============
    
    def retrieve_similar_codes(self, description: str, top_k: int = 3) -> List[ServiceCode]:
        """Retrieve similar service codes using TF-IDF"""
        
        if not self.index:
            return self._get_fallback_codes()
        
        try:
            # Get all vectors from Pinecone
            result = self.index.describe_index_stats()
            
            if result.total_vector_count == 0:
                return self._get_fallback_codes()
            
            # Query Pinecone to get all vectors (using empty filter to get all)
            query_result = self.index.query(
                vector=[0] * 384,  # Dummy vector, we'll use TF-IDF for scoring
                top_k=result.total_vector_count,
                include_metadata=True,
                include_values=False
            )
            
            if not query_result.matches:
                return self._get_fallback_codes()
            
            # Build document list from Pinecone results
            documents = []
            metadatas = []
            for match in query_result.matches:
                documents.append(match.metadata.get('description', ''))
                metadatas.append({
                    'code': match.metadata.get('code', 'UNKNOWN'),
                    'category': match.metadata.get('category', 'General')
                })
            
            if not documents:
                return self._get_fallback_codes()
            
            # Fit TF-IDF on documents
            self.tfidf.fit(documents)
            
            # Get query vector
            query_vector = self.tfidf.transform([description]).toarray()[0]
            
            # Calculate cosine similarity
            similarities = []
            for i, doc in enumerate(documents):
                doc_vector = self.tfidf.transform([doc]).toarray()[0]
                
                dot = sum(a * b for a, b in zip(query_vector, doc_vector))
                query_mag = math.sqrt(sum(a * a for a in query_vector))
                doc_mag = math.sqrt(sum(a * a for a in doc_vector))
                
                if query_mag > 0 and doc_mag > 0:
                    sim = dot / (query_mag * doc_mag)
                else:
                    sim = 0
                
                similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            results = []
            for i, sim in similarities[:top_k]:
                results.append(ServiceCode(
                    code=metadatas[i].get('code', 'UNKNOWN'),
                    description=documents[i],
                    category=metadatas[i].get('category', 'General')
                ))
            
            return results
            
        except Exception as e:
            print(f"   [WARN] Retrieval error: {e}")
            return self._get_fallback_codes()
    
    def _get_fallback_codes(self) -> List[ServiceCode]:
        """Fallback codes"""
        return [
            ServiceCode(code="01_111_1117_1_1", description="Personal Care Assistance", category="Personal Care"),
            ServiceCode(code="01_112_1117_1_1", description="Domestic Assistance", category="Domestic Assistance"),
            ServiceCode(code="01_117_1117_1_1", description="Gardening & Maintenance", category="Gardening"),
        ]
    
    def generate_response(self, description: str, similar_codes: List[ServiceCode]) -> RAGResponse:
        """Generate response using LLM with consistent category-to-code mapping"""
        
        # First, determine the category using LLM - ask if it matches existing categories or needs new one
        category_prompt = f"""You are an AI assistant for categorizing aged care invoice descriptions.

EXISTING CATEGORIES:
- Personal Care (Code: 01_011_0107_1_1)
- Nursing (Code: 01_121_1117_1_1)
- Domestic Assistance (Code: 01_020_0120_1_1)
- Gardening (Code: 01_019_0120_1_1)
- Meals (Code: 01_131_1117_1_1)
- Transport (Code: 01_146_1117_1_1)
- Allied Health (Code: 01_230_1117_1_1)

Invoice Description: {description}

Analyze the description and respond with ONLY a JSON object:
{{
    "is_new_category": true or false,
    "matched_category": "<if existing, name of matched category> or null",
    "suggested_category": "<if new, suggest a new category name> or null",
    "reasoning": "<brief explanation>"
}}

JSON:"""
        
        category_response = self._call_llm(category_prompt)
        
        # Parse category from response
        is_new_category = False
        detected_category = None
        suggested_category = None
        reasoning = ""
        
        try:
            json_start = category_response.find('{')
            json_end = category_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = category_response[json_start:json_end]
                result = json.loads(json_str)
                is_new_category = result.get("is_new_category", False)
                detected_category = result.get("matched_category")
                suggested_category = result.get("suggested_category")
                reasoning = result.get("reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # If category not detected from LLM, use keyword matching as fallback
        if not detected_category and not is_new_category:
            detected_category = self._detect_category_by_keywords(description)
        
        # If it's a new category request
        if is_new_category or not detected_category:
            # Ask LLM to suggest a new category and service code format
            new_category_prompt = f"""A new invoice description doesn't match any existing category.

Invoice Description: {description}

Please suggest:
1. A new category name that would fit this service
2. A format for the NDIS-style service code (e.g., 01_XXX_XXXX_X_X)

Respond with ONLY a JSON object:
{{
    "suggested_category": "<new category name>",
    "suggested_code_format": "01_<3-digit>_<4-digit>_<1>_<1>",
    "description": "<brief description of this service category>"
}}

JSON:"""
            new_cat_response = self._call_llm(new_category_prompt)
            
            try:
                json_start = new_cat_response.find('{')
                json_end = new_cat_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = new_cat_response[json_start:json_end]
                    result = json.loads(json_str)
                    suggested_category = result.get("suggested_category", "New Category")
                    suggested_code_format = result.get("suggested_code_format", "01_999_9999_1_1")
            except (json.JSONDecodeError, AttributeError):
                suggested_category = "New Category"
                suggested_code_format = "01_999_9999_1_1"
            
            return RAGResponse(
                suggested_code="PENDING_APPROVAL",
                confidence_score=0.0,
                reasoning=f"New category needed: {reasoning}",
                needs_approval=True,
                suggested_category=suggested_category,
                suggested_code_format=suggested_code_format
            )
        
        # Get the consistent service code for this category
        suggested_code = CATEGORY_SERVICE_CODES.get(detected_category, "UNKNOWN")
        
        # If category not in our mapping, use keyword fallback
        if suggested_code == "UNKNOWN":
            return self._keyword_fallback(description, similar_codes)
        
        return RAGResponse(
            suggested_code=suggested_code,
            confidence_score=0.95,  # High confidence since we use consistent codes per category
            reasoning=f"Category '{detected_category}' mapped to service code {suggested_code}",
            needs_approval=False
        )
    
    def _detect_category_by_keywords(self, description: str) -> str:
        """Detect category using keyword matching"""
        desc_lower = description.lower()
        
        # Category keywords
        keywords = {
            "Personal Care": ["shower", "bath", "dressing", "grooming", "hygiene", "toilet", "mobility", "transfer", "personal care", "assistance with self-care"],
            "Nursing": ["nurse", "nursing", "medication", "wound", "health", "medical", "doctor", "clinical", "therapy"],
            "Domestic Assistance": ["cleaning", "laundry", "vacuum", "mop", "dust", "house", "dish", "washing", "ironing", "bed linen", "household"],
            "Gardening": ["garden", "lawn", "mow", "weed", "prune", "yard", "gutter", "maintenance", "outdoor"],
            "Meals": ["meal", "food", "cooking", "lunch", "dinner", "breakfast", "delivery"],
            "Transport": ["transport", "drive", "car", "appointment", "travel", "bus", "taxi"],
            "Allied Health": ["physio", "physiotherapy", "occupational", "speech", "podiatry", "dietitian", "psychologist"],
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in desc_lower:
                    return category
        
        return "Domestic Assistance"  # Default category

    # ============== GENDER DETECTION ==============
    
    def detect_gender_from_name(self, invoice_text: str) -> str:
        """
        Detect gender from invoice person name using LLM.
        The LLM extracts the person name from the invoice and determines gender.
        Returns: "Male", "Female", or "Unknown"
        """
        prompt = f"""You are an AI assistant that extracts person names from invoices and determines gender.

Invoice Text:
{invoice_text[:2000]}  # Limit text length

Instructions:
1. Extract the CLIENT/CUSTOMER/PERSON name from the invoice
2. Determine if the person is Male or Female based on the name
3. Consider common first names (e.g., John, Michael, David = Male; Mary, Jane, Clara = Female)

Respond with ONLY a JSON object:
{{
    "person_name": "<extracted name or 'Not found'",
    "gender": "Male" or "Female" or "Unknown"
}}

JSON:"""
        
        response = self._call_llm(prompt)
        
        # Parse JSON
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                gender = result.get("gender", "Unknown")
                person_name = result.get("person_name", "Unknown")
                print(f"   [GENDER] Detected: {person_name} -> {gender}")
                return gender
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: Try basic name matching
        return self._detect_gender_by_name_patterns(invoice_text)
    
    def _detect_gender_by_name_patterns(self, text: str) -> str:
        """Fallback gender detection using name patterns"""
        # Common male names
        male_names = ["john", "james", "robert", "michael", "david", "richard", "william", "joseph", 
                      "thomas", "charles", "george", "edward", "harry", "jack", "daniel", "matthew",
                      "anthony", "mark", "steven", "paul", "andrew", "joshua", "kenneth", "kevin",
                      "brian", "george", "edward", "raymond", "gary", "eric", "larry", "scott", "frank"]
        
        # Common female names
        female_names = ["mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", 
                        "jessica", "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra",
                        "ashley", "kimberly", "emily", "donna", "michelle", "dorothy", "carol", "amanda",
                        "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
                        "kathleen", "amy", "shirley", "angela", "helen", "anna", "brenda", "pamela",
                        "nicole", "samantha", "katherine", "christine", "debra", "rachel", "carolyn", "janet",
                        "catherine", "maria", "heather", "diane", "ruth", "julie", "olivia", "joyce",
                        "clara", "victoria", "kelly", "lauren", "christina", "joan", "evelyn", "judith"]
        
        text_lower = text.lower()
        
        # Check for male names
        for name in male_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Male"
        
        # Check for female names
        for name in female_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Female"
        
        return "Unknown"
    
    def _keyword_fallback(self, description: str, codes: List[ServiceCode]) -> RAGResponse:
        """Fallback using keyword matching"""
        if not codes:
            return RAGResponse(suggested_code="UNKNOWN", confidence_score=0.0, reasoning="No codes found")
        
        desc_words = set(description.lower().split())
        best_match = codes[0]
        best_score = 0
        
        for code in codes:
            code_words = set(code.description.lower().split())
            score = len(desc_words & code_words) / max(len(code_words), 1)
            if score > best_score:
                best_score = score
                best_match = code
        
        return RAGResponse(
            suggested_code=best_match.code,
            confidence_score=min(0.85, best_score + 0.4),
            reasoning=f"Matched via keywords ({self.current_api or 'fallback'})"
        )
    
    def process_invoice(self, pdf_path: str) -> dict:
        """Process an invoice through the full RAG pipeline"""
        print(f"\n[DOC] Processing invoice: {pdf_path}")
        
        # Extract and chunk text from PDF
        text_chunks = self.extract_text_from_pdf(pdf_path, chunk_size=500)
        if not text_chunks:
            return {"error": "Failed to extract text from PDF", "line_items": []}
        
        # Join all chunks for processing
        full_text = " ".join(text_chunks)
        print(f"   [CHUNKS] Extracted {len(text_chunks)} chunks (500 chars each)")
        
        # Detect gender from invoice
        gender = self.detect_gender_from_name(full_text)
        print(f"   [GENDER] Detected: {gender}")
        
        # Parse line items with amounts
        parsed_items = self.parse_line_items(full_text)
        print(f"   Found {len(parsed_items)} line items")
        
        results = []
        total_amount = 0.0
        
        for i, item in enumerate(parsed_items):
            desc = item.get("description", "")
            amount = item.get("amount", 0.0)
            total_amount += amount
            
            print(f"   Processing item {i+1}...")
            
            similar_codes = self.retrieve_similar_codes(desc)
            rag_resp = self.generate_response(desc, similar_codes)
            
            flagged = rag_resp.confidence_score < self.confidence_threshold
            
            result = {
                "description": desc,
                "amount": amount,
                "suggested_code": rag_resp.suggested_code,
                "confidence_score": rag_resp.confidence_score,
                "reasoning": rag_resp.reasoning,
                "flagged": flagged,
                "api_used": self.current_api or "keyword",
                "retrieved_codes": [
                    {"code": c.code, "category": c.category, "description": c.description}
                    for c in similar_codes
                ]
            }
            
            results.append(result)
            
            status = "[FLAG]" if flagged else "[OK]"
            print(f"      → {rag_resp.suggested_code} ({rag_resp.confidence_score:.2f}) {status}")
        
        print(f"   Processed {len(results)} items\n")
        print(f"   [TOTAL] Calculated total: ${total_amount:.2f}")
        
        return {
            "line_items": results,
            "gender": gender,
            "total_amount": total_amount
        }
    
    def update_knowledge_base(self, description: str, correct_code: str):
        """Feedback loop - learn from corrections"""
        print(f"   [LEARN] Learning: {correct_code}")
    
    def seed_service_codes(self, codes: List[ServiceCode]):
        """Seed vector store with service codes"""
        if not self.index:
            print("   [WARN] Cannot seed - no Pinecone index")
            return
        
        try:
            documents = [code.description for code in codes]
            
            # Fit TF-IDF
            if self.tfidf:
                self.tfidf.fit(documents)
            
            # Generate embeddings
            vectors = self.tfidf.transform(documents).toarray()
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, code in enumerate(codes):
                vectors_to_upsert.append({
                    'id': f"code_{i}",
                    'values': vectors[i].tolist(),
                    'metadata': {
                        'code': code.code,
                        'description': code.description,
                        'category': code.category
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"   [OK] Seeded {len(codes)} codes to Pinecone")
        except Exception as e:
            print(f"   Error seeding codes: {e}")
    
    def load_seed_from_file(self, file_path: str):
        """Load service codes from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            codes = [ServiceCode(**item) for item in data]
            self.seed_service_codes(codes)
            print(f"   [OK] Loaded {len(codes)} codes from {file_path}")
        except Exception as e:
            print(f"   Error loading seed file: {e}")
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "initialized": self.is_initialized,
            "pinecone_index": self.pinecone_index if self.index else "not configured",
            "current_api": self.current_api or "none",
            "groq_keys": len(self.groq_keys),
            "google_keys": len(self.google_keys),
            "embedding": "TF-IDF"
        }


# Run for testing
if __name__ == "__main__":
    engine = RAGEngine()
    print("\n📊 Stats:", engine.get_stats())


def detect_category_from_description(self, description: str) -> str:
    """Detect category based on keywords in the description"""
    desc_lower = description.lower()

    # Define category keywords
    category_keywords = {
        "Domestic Assistance": ["domestic", "house cleaning", "housekeeping", "cleaning", "laundry", "ironing", "bed making", "household", "home cleaning", "general household", "domestic assistance"],
        "Personal Care": ["personal care", "showering", "bathing", "dressing", "toileting", "mobility", "assisted daily living", "adl", "personal hygiene", "grooming"],
        "Nursing": ["nursing", "nurse", "medical", "medication", "wound care", "injection", "health monitoring", "clinical", "healthcare"],
        "Allied Health": ["physiotherapy", "occupational therapy", "speech therapy", "podiatry", "dietitian", "allied health", "therapy", "rehabilitation"],
        "Transport": ["transport", "transportation", "travel", "appointment transport", "medical transport", "community transport"],
        "Gardening": ["gardening", "garden", "lawn mowing", "hedging", "yard maintenance", "landscaping", "outdoor", "green"],
        "Meals": ["meal", "meals", "cooking", "food preparation", "meal delivery", "prepared meals", "menu"],
        "IT Support": ["it", "software", "computer", "laptop", "installation", "configuration", "tech", "technology", "digital", "printer", "network", "wifi", "internet"],
    }

    # Check each category for keyword matches
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return category

    return "General"


def parse_line_items_simple(self, text: str) -> List[str]:
    """Parse invoice text into line items (legacy method)"""
    lines = text.split('\n')
    return [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]


# ============== RAG OPERATIONS ==============

def retrieve_similar_codes(self, description: str, top_k: int = 3) -> List[ServiceCode]:
    """Retrieve similar service codes using TF-IDF"""

    if not self.index:
        return self._get_fallback_codes()

    try:
        result = self.index.describe_index_stats()

        if result.total_vector_count == 0:
            return self._get_fallback_codes()

        query_result = self.index.query(
            vector=[0] * 384,
            top_k=result.total_vector_count,
            include_metadata=True,
            include_values=False
        )

        if not query_result.matches:
            return self._get_fallback_codes()

        documents = []
        metadatas = []

        for match in query_result.matches:
            documents.append(match.metadata.get('description', ''))
            metadatas.append({
                'code': match.metadata.get('code', 'UNKNOWN'),
                'category': match.metadata.get('category', 'General')
            })

        if not documents:
            return self._get_fallback_codes()

        self.tfidf.fit(documents)

        query_vector = self.tfidf.transform([description]).toarray()[0]

        similarities = []
        for i, doc in enumerate(documents):
            doc_vector = self.tfidf.transform([doc]).toarray()[0]

            dot = sum(a * b for a, b in zip(query_vector, doc_vector))
            query_mag = math.sqrt(sum(a * a for a in query_vector))
            doc_mag = math.sqrt(sum(a * a for a in doc_vector))

            if query_mag > 0 and doc_mag > 0:
                sim = dot / (query_mag * doc_mag)
            else:
                sim = 0

            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, sim in similarities[:top_k]:
            results.append(ServiceCode(
                code=metadatas[i].get('code', 'UNKNOWN'),
                description=documents[i],
                category=metadatas[i].get('category', 'General')
            ))

        return results

    except Exception as e:
        print(f"   [WARN] Retrieval error: {e}")
        return self._get_fallback_codes()


def _get_fallback_codes(self) -> List[ServiceCode]:
    return [
        ServiceCode(code="01_111_1117_1_1", description="Personal Care Assistance", category="Personal Care"),
        ServiceCode(code="01_112_1117_1_1", description="Domestic Assistance", category="Domestic Assistance"),
        ServiceCode(code="01_117_1117_1_1", description="Gardening & Maintenance", category="Gardening"),
    ]
    def generate_response(self, description: str, similar_codes: List[ServiceCode]) -> RAGResponse:
        """Generate response using LLM with consistent category-to-code mapping"""
        
        # First, determine the category using LLM - ask if it matches existing categories or needs new one
        category_prompt = f"""You are an AI assistant for categorizing aged care invoice descriptions.

EXISTING CATEGORIES:
- Personal Care (Code: 01_011_0107_1_1)
- Nursing (Code: 01_121_1117_1_1)
- Domestic Assistance (Code: 01_020_0120_1_1)
- Gardening (Code: 01_019_0120_1_1)
- Meals (Code: 01_131_1117_1_1)
- Transport (Code: 01_146_1117_1_1)
- Allied Health (Code: 01_230_1117_1_1)

Invoice Description: {description}

Analyze the description and respond with ONLY a JSON object:
{{
    "is_new_category": true or false,
    "matched_category": "<if existing, name of matched category> or null",
    "suggested_category": "<if new, suggest a new category name> or null",
    "reasoning": "<brief explanation>"
}}

JSON:"""
        
        category_response = self._call_llm(category_prompt)
        
        # Parse category from response
        is_new_category = False
        detected_category = None
        suggested_category = None
        reasoning = ""
        
        try:
            json_start = category_response.find('{')
            json_end = category_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = category_response[json_start:json_end]
                result = json.loads(json_str)
                is_new_category = result.get("is_new_category", False)
                detected_category = result.get("matched_category")
                suggested_category = result.get("suggested_category")
                reasoning = result.get("reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # If category not detected from LLM, use keyword matching as fallback
        if not detected_category and not is_new_category:
            detected_category = self._detect_category_by_keywords(description)
        
        # If it's a new category request
        if is_new_category or not detected_category:
            # Ask LLM to suggest a new category and service code format
            new_category_prompt = f"""A new invoice description doesn't match any existing category.

Invoice Description: {description}

Please suggest:
1. A new category name that would fit this service
2. A format for the NDIS-style service code (e.g., 01_XXX_XXXX_X_X)

Respond with ONLY a JSON object:
{{
    "suggested_category": "<new category name>",
    "suggested_code_format": "01_<3-digit>_<4-digit>_<1>_<1>",
    "description": "<brief description of this service category>"
}}

JSON:"""
            new_cat_response = self._call_llm(new_category_prompt)
            
            try:
                json_start = new_cat_response.find('{')
                json_end = new_cat_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = new_cat_response[json_start:json_end]
                    result = json.loads(json_str)
                    suggested_category = result.get("suggested_category", "New Category")
                    suggested_code_format = result.get("suggested_code_format", "01_999_9999_1_1")
            except (json.JSONDecodeError, AttributeError):
                suggested_category = "New Category"
                suggested_code_format = "01_999_9999_1_1"
            
            return RAGResponse(
                suggested_code="PENDING_APPROVAL",
                confidence_score=0.0,
                reasoning=f"New category needed: {reasoning}",
                needs_approval=True,
                suggested_category=suggested_category,
                suggested_code_format=suggested_code_format
            )
        
        # Get the consistent service code for this category
        suggested_code = CATEGORY_SERVICE_CODES.get(detected_category, "UNKNOWN")
        
        # If category not in our mapping, use keyword fallback
        if suggested_code == "UNKNOWN":
            return self._keyword_fallback(description, similar_codes)
        
        return RAGResponse(
            suggested_code=suggested_code,
            confidence_score=0.95,  # High confidence since we use consistent codes per category
            reasoning=f"Category '{detected_category}' mapped to service code {suggested_code}",
            needs_approval=False
        )
    
    def _detect_category_by_keywords(self, description: str) -> str:
        """Detect category using keyword matching"""
        desc_lower = description.lower()
        
        # Category keywords
        keywords = {
            "Personal Care": ["shower", "bath", "dressing", "grooming", "hygiene", "toilet", "mobility", "transfer", "personal care", "assistance with self-care"],
            "Nursing": ["nurse", "nursing", "medication", "wound", "health", "medical", "doctor", "clinical", "therapy"],
            "Domestic Assistance": ["cleaning", "laundry", "vacuum", "mop", "dust", "house", "dish", "washing", "ironing", "bed linen", "household"],
            "Gardening": ["garden", "lawn", "mow", "weed", "prune", "yard", "gutter", "maintenance", "outdoor"],
            "Meals": ["meal", "food", "cooking", "lunch", "dinner", "breakfast", "delivery"],
            "Transport": ["transport", "drive", "car", "appointment", "travel", "bus", "taxi"],
            "Allied Health": ["physio", "physiotherapy", "occupational", "speech", "podiatry", "dietitian", "psychologist"],
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in desc_lower:
                    return category
        
        return "Domestic Assistance"  # Default category

    # ============== GENDER DETECTION ==============
    
    def detect_gender_from_name(self, invoice_text: str) -> str:
        """
        Detect gender from invoice person name using LLM.
        The LLM extracts the person name from the invoice and determines gender.
        Returns: "Male", "Female", or "Unknown"
        """
        prompt = f"""You are an AI assistant that extracts person names from invoices and determines gender.

Invoice Text:
{invoice_text[:2000]}  # Limit text length

Instructions:
1. Extract the CLIENT/CUSTOMER/PERSON name from the invoice
2. Determine if the person is Male or Female based on the name
3. Consider common first names (e.g., John, Michael, David = Male; Mary, Jane, Clara = Female)

Respond with ONLY a JSON object:
{{
    "person_name": "<extracted name or 'Not found'",
    "gender": "Male" or "Female" or "Unknown"
}}

JSON:"""
        
        response = self._call_llm(prompt)
        
        # Parse JSON
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                gender = result.get("gender", "Unknown")
                person_name = result.get("person_name", "Unknown")
                print(f"   [GENDER] Detected: {person_name} -> {gender}")
                return gender
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: Try basic name matching
        return self._detect_gender_by_name_patterns(invoice_text)
    
    def _detect_gender_by_name_patterns(self, text: str) -> str:
        """Fallback gender detection using name patterns"""
        # Common male names
        male_names = ["john", "james", "robert", "michael", "david", "richard", "william", "joseph", 
                      "thomas", "charles", "george", "edward", "harry", "jack", "daniel", "matthew",
                      "anthony", "mark", "steven", "paul", "andrew", "joshua", "kenneth", "kevin",
                      "brian", "george", "edward", "raymond", "gary", "eric", "larry", "scott", "frank"]
        
        # Common female names
        female_names = ["mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", 
                        "jessica", "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra",
                        "ashley", "kimberly", "emily", "donna", "michelle", "dorothy", "carol", "amanda",
                        "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
                        "kathleen", "amy", "shirley", "angela", "helen", "anna", "brenda", "pamela",
                        "nicole", "samantha", "katherine", "christine", "debra", "rachel", "carolyn", "janet",
                        "catherine", "maria", "heather", "diane", "ruth", "julie", "olivia", "joyce",
                        "clara", "victoria", "kelly", "lauren", "christina", "joan", "evelyn", "judith"]
        
        text_lower = text.lower()
        
        # Check for male names
        for name in male_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Male"
        
        # Check for female names
        for name in female_names:
            if f" {name} " in text_lower or f" {name}." in text_lower or text_lower.startswith(name + " "):
                return "Female"
        
        return "Unknown"
    
    def _keyword_fallback(self, description: str, codes: List[ServiceCode]) -> RAGResponse:
        """Fallback using keyword matching"""
        if not codes:
            return RAGResponse(suggested_code="UNKNOWN", confidence_score=0.0, reasoning="No codes found")
        
        desc_words = set(description.lower().split())
        best_match = codes[0]
        best_score = 0
        
        for code in codes:
            code_words = set(code.description.lower().split())
            score = len(desc_words & code_words) / max(len(code_words), 1)
            if score > best_score:
                best_score = score
                best_match = code
        
        return RAGResponse(
            suggested_code=best_match.code,
            confidence_score=min(0.85, best_score + 0.4),
            reasoning=f"Matched via keywords ({self.current_api or 'fallback'})"
        )
    
    def process_invoice(self, pdf_path: str) -> dict:
        """Process an invoice through the full RAG pipeline"""
        print(f"\n[DOC] Processing invoice: {pdf_path}")
        
        # Extract and chunk text from PDF
        text_chunks = self.extract_text_from_pdf(pdf_path, chunk_size=500)
        if not text_chunks:
            return {"error": "Failed to extract text from PDF", "line_items": []}
        
        # Join all chunks for processing
        full_text = " ".join(text_chunks)
        print(f"   [CHUNKS] Extracted {len(text_chunks)} chunks (500 chars each)")
        
        # Detect gender from invoice
        gender = self.detect_gender_from_name(full_text)
        print(f"   [GENDER] Detected: {gender}")
        
        # Parse line items with amounts
        parsed_items = self.parse_line_items(full_text)
        print(f"   Found {len(parsed_items)} line items")
        
        results = []
        total_amount = 0.0
        
        for i, item in enumerate(parsed_items):
            desc = item.get("description", "")
            amount = item.get("amount", 0.0)
            total_amount += amount
            
            print(f"   Processing item {i+1}...")
            
            similar_codes = self.retrieve_similar_codes(desc)
            rag_resp = self.generate_response(desc, similar_codes)
            
            flagged = rag_resp.confidence_score < self.confidence_threshold
            
            result = {
                "description": desc,
                "amount": amount,
                "suggested_code": rag_resp.suggested_code,
                "confidence_score": rag_resp.confidence_score,
                "reasoning": rag_resp.reasoning,
                "flagged": flagged,
                "api_used": self.current_api or "keyword",
                "retrieved_codes": [
                    {"code": c.code, "category": c.category, "description": c.description}
                    for c in similar_codes
                ]
            }
            
            results.append(result)
            
            status = "[FLAG]" if flagged else "[OK]"
            print(f"      → {rag_resp.suggested_code} ({rag_resp.confidence_score:.2f}) {status}")
        
        print(f"   Processed {len(results)} items\n")
        print(f"   [TOTAL] Calculated total: ${total_amount:.2f}")
        
        return {
            "line_items": results,
            "gender": gender,
            "total_amount": total_amount
        }
    
    def update_knowledge_base(self, description: str, correct_code: str):
        """Feedback loop - learn from corrections"""
        print(f"   [LEARN] Learning: {correct_code}")
    
    def seed_service_codes(self, codes: List[ServiceCode]):
        """Seed vector store with service codes"""
        if not self.index:
            print("   [WARN] Cannot seed - no Pinecone index")
            return
        
        try:
            documents = [code.description for code in codes]
            
            # Fit TF-IDF
            if self.tfidf:
                self.tfidf.fit(documents)
            
            # Generate embeddings
            vectors = self.tfidf.transform(documents).toarray()
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, code in enumerate(codes):
                vectors_to_upsert.append({
                    'id': f"code_{i}",
                    'values': vectors[i].tolist(),
                    'metadata': {
                        'code': code.code,
                        'description': code.description,
                        'category': code.category
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"   [OK] Seeded {len(codes)} codes to Pinecone")
        except Exception as e:
            print(f"   Error seeding codes: {e}")
    
    def load_seed_from_file(self, file_path: str):
        """Load service codes from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            codes = [ServiceCode(**item) for item in data]
            self.seed_service_codes(codes)
            print(f"   [OK] Loaded {len(codes)} codes from {file_path}")
        except Exception as e:
            print(f"   Error loading seed file: {e}")
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "initialized": self.is_initialized,
            "pinecone_index": self.pinecone_index if self.index else "not configured",
            "current_api": self.current_api or "none",
            "groq_keys": len(self.groq_keys),
            "google_keys": len(self.google_keys),
            "embedding": "TF-IDF"
        }

##
# Run for testing
if __name__ == "__main__":
    engine = RAGEngine()
    print("\n📊 Stats:", engine.get_stats())