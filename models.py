from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

# ============== USER MODELS ==============

class User(BaseModel):
    """User model for authentication"""
    id: Optional[str] = None
    username: str
    role: str = Field(description="Role: Administrator, Supplier, Finance, Technical Monitor")
    email: str
    password: Optional[str] = None  # Only for registration, never returned


class UserCreate(BaseModel):
    """Request model for creating a user"""
    username: str
    email: str
    password: str
    role: str = "Supplier"  # Default role


# ============== SERVICE CODE MODELS ==============

class ServiceCode(BaseModel):
    """Service code model for Sensible Care catalog"""
    code: str = Field(description="e.g., 01_111_1117_1_1")
    description: str
    category: str = Field(description="Personal Care, Nursing, Domestic Assistance, Gardening, Meals, Transport, Allied Health")
    status: str = "Active"
    metadata: Optional[dict] = None


class ServiceCodeCreate(BaseModel):
    """Request model for creating a service code"""
    code: str
    description: str
    category: str
    status: str = "Active"


# ============== LINE ITEM MODELS ==============

class LineItem(BaseModel):
    """Line item extracted from invoice"""
    description: str
    suggested_code: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    detected_category: Optional[str] = None
    category: Optional[str] = None
    final_code: Optional[str] = None
    flagged: bool = False
    retrieved_codes: Optional[List[dict]] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class LineItemUpdate(BaseModel):
    """Request model for updating a line item"""
    final_code: str
    notes: Optional[str] = None


# ============== INVOICE MODELS ==============

class Invoice(BaseModel):
    """Invoice model"""
    id: Optional[str] = None
    invoice_id: str
    supplier_id: str
    pdf_path: str
    status: str = Field(description="Pending Review, Processing, Approved, Rejected")
    uploaded_at: datetime
    line_items: List[LineItem] = []
    total_amount: Optional[float] = 0.0
    processed_by: Optional[str] = "RAG Engine"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


class InvoiceCreate(BaseModel):
    """Request model for creating an invoice"""
    supplier_id: str
    pdf_path: str
    line_items: List[LineItem]


class InvoiceUpdate(BaseModel):
    """Request model for updating an invoice"""
    status: Optional[str] = None
    total_amount: Optional[float] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


# ============== AUDIT LOG MODELS ==============

class AuditLog(BaseModel):
    """Audit log for tracking all actions"""
    id: Optional[str] = None
    invoice_id: str
    action: str
    user_id: str
    timestamp: datetime
    details: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None


class AuditLogCreate(BaseModel):
    """Request model for creating an audit log"""
    invoice_id: str
    action: str
    user_id: str
    details: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None


# ============== RAG RESPONSE MODELS ==============

class RAGResponse(BaseModel):
    """Response from RAG engine"""
    suggested_code: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class RAGProcessRequest(BaseModel):
    """Request to process a single description through RAG"""
    description: str
    top_k: int = 3


class RAGProcessResponse(BaseModel):
    """Response from processing"""
    results: List[RAGResponse]
    flagged: bool = False


# ============== STATS MODELS ==============

class SystemStats(BaseModel):
    """System statistics"""
    total_invoices: int
    total_service_codes: int
    total_audit_logs: int
    invoices_by_status: dict
    codes_by_category: dict


class RAGStats(BaseModel):
    """RAG engine statistics"""
    initialized: bool
    pinecone_index: str
    llm_model: Optional[str]
    embedding_model: str
    total_documents: int


# ============== API RESPONSE MODELS ==============

class MessageResponse(BaseModel):
    """Standard message response"""
    message: str
    success: bool = True


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    success: bool = False


# ============== CATEGORY REQUEST MODELS ==============

class CategoryRequest(BaseModel):
    """Model for requesting new category/service code approval"""
    id: Optional[str] = None
    description: str  # The invoice description that triggered this request
    suggested_category: str  # Proposed new category name
    suggested_code: str  # Proposed service code format
    requested_by: str  # Supplier who requested this
    invoice_id: str  # Associated invoice
    status: str = "Pending"  # Pending, Approved, Rejected
    requested_at: datetime
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    notes: Optional[str] = None


class CategoryRequestCreate(BaseModel):
    """Request model for creating a category request"""
    description: str
    suggested_category: str
    suggested_code: str
    invoice_id: str


class CategoryRequestUpdate(BaseModel):
    """Request model for updating a category request (approval/rejection)"""
    status: str  # Approved or Rejected
    notes: Optional[str] = None


# ============== INVOICE APPROVAL REQUEST MODELS ==============

class InvoiceApprovalRequest(BaseModel):
    """Model for invoice approval requests to finance officer"""
    id: Optional[str] = None
    invoice_id: str
    supplier_id: str
    line_items: List[dict]  # The line items with suggested codes
    status: str = "Pending"  # Pending, Approved, Rejected
    requested_at: datetime
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    notes: Optional[str] = None


class SupplierConfirmRequest(BaseModel):
    """Request model for supplier to confirm invoice suggestions"""
    confirmed: bool  # True if supplier confirms, False if rejects
    notes: Optional[str] = None