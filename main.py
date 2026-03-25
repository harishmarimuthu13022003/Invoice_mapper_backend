"""
AI Invoice Mapper - FastAPI Backend
Uses MongoDB for data storage and RAG Engine for AI processing
"""

import os
import json
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Body, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# MongoDB imports
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

# Local imports
from rag_engine import RAGEngine
from models import User, Invoice, LineItem, AuditLog, ServiceCode, SupplierConfirmRequest

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "invoice_mapper")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8001"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# ============================================
# APP INITIALIZATION
# ============================================

app = FastAPI(
    title="AI Invoice Mapper API",
    description="RAG-powered Invoice-to-Service Code Mapping for Aged Care",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Flutter web app
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "Content-Type", "Authorization", "x-role"],
)

# MongoDB Connection
client: Optional[AsyncIOMotorClient] = None
db = None

@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB and RAG Engine on startup"""
    global client, db, rag
    
    print("\n" + "="*50)
    print("[START] Starting AI Invoice Mapper Backend")
    print("="*50)
    
    # Connect to MongoDB
    try:
        client = AsyncIOMotorClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        
        # Create indexes
        await db.invoices.create_index("invoice_id")
        await db.invoices.create_index("supplier_id")
        await db.audit_logs.create_index("timestamp")
        await db.service_codes.create_index("code", unique=True)
        
        print(f"[OK] MongoDB connected: {MONGODB_DATABASE}")
    except Exception as e:
        print(f"[WARN] MongoDB connection failed: {e}")
        print("[INFO] Server will run without MongoDB (some features disabled)")
        client = None
        db = None
    
    # Initialize RAG Engine
    try:
        rag = RAGEngine()
        print(f"[OK] RAG Engine initialized")
    except Exception as e:
        print(f"[WARN] RAG Engine initialization issue: {e}")
        rag = None
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"[OK] Upload directory: {UPLOAD_DIR}")
    print("="*50 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if client:
        client.close()
        print("[OK] MongoDB connection closed")

# ============================================
# PYDANTIC MODELS (Request/Response)
# ============================================

class UserResponse(BaseModel):
    username: str
    role: str
    email: str

class InvoiceResponse(BaseModel):
    invoice_id: str
    supplier_id: str
    pdf_path: str
    status: str
    uploaded_at: datetime
    line_items: List[dict]
    total_amount: Optional[float] = None

class LineItemUpdate(BaseModel):
    final_code: str
    notes: Optional[str] = None

# ============================================
# AUTHENTICATION (Simplified)
# ============================================

# In production, use JWT tokens
async def get_current_user(x_role: str = Header(default="Administrator")) -> User:
    """Get current user based on role header (simplified auth)"""
    return User(
        username="system_user",
        role=x_role,
        email="system@invoice-mapper.com"
    )

def require_role(allowed_roles: List[str]):
    """Dependency to check role authorization"""
    async def role_checker(user: User = Depends(get_current_user)):
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' not authorized for this action"
            )
        return user
    return role_checker

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "AI Invoice Mapper API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    mongo_status = "connected"
    rag_status = "initialized" if (rag and rag.is_initialized) else "disabled"
    
    try:
        await db.invoices.count_documents({})
    except:
        mongo_status = "disconnected"
    
    return {
        "mongodb": mongo_status,
        "rag_engine": rag_status,
        "timestamp": datetime.now().isoformat()
    }

# ============== INVOICE ENDPOINTS ==============

@app.post("/invoices/upload", response_model=dict)
async def upload_invoice(
    file: UploadFile = File(...),
    user: User = Depends(require_role(["Supplier"]))
):
    """
    Upload and process an invoice
    1. Save PDF to uploads folder
    2. Process with RAG engine
    3. Store in MongoDB
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted"
        )
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{datetime.now().timestamp()}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process with RAG
    gender = "Unknown"  # Default gender
    category_requests_created = []  # Track category requests
    if rag:
        rag_result = rag.process_invoice(file_path)
        if isinstance(rag_result, dict):
            results = rag_result.get("line_items", [])
            gender = rag_result.get("gender", "Unknown")
        else:
            results = rag_result
    else:
        results = [{"error": "RAG engine not available"}]
    
    # Create line items
    line_items = []
    has_pending_approval = False  # Track if any item needs approval
    for item in results:
        if "error" not in item:
            # Check if this item needs approval for new category
            needs_approval = item.get("needs_approval", False)
            if needs_approval:
                has_pending_approval = True
                # Create a category request
                category_request = {
                    "description": item.get("description", ""),
                    "suggested_category": item.get("suggested_category", "New Category"),
                    "suggested_code": item.get("suggested_code_format", "01_999_9999_1_1"),
                    "requested_by": user.username,
                    "invoice_id": f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "status": "Pending",
                    "requested_at": datetime.now()
                }
                # Insert category request
                cat_result = await db.category_requests.insert_one(category_request)
                category_request["_id"] = str(cat_result.inserted_id)
                category_requests_created.append(category_request)
            
            line_items.append(LineItem(
                description=item.get("description", ""),
                suggested_code=item.get("suggested_code", ""),
                confidence_score=item.get("confidence_score", 0.0),
                reasoning=item.get("reasoning", ""),
                final_code=item.get("suggested_code") if not needs_approval else "PENDING_APPROVAL",  # Keep pending if needs approval
                flagged=needs_approval or item.get("flagged", False),
                retrieved_codes=item.get("retrieved_codes", [])
            ))
    
    # Determine invoice status
    # Flow: Upload -> RAG Process -> Finance Officer Approval -> Approved
    # For new category: Upload -> RAG -> Admin Approval -> Finance Approval -> Approved
    if has_pending_approval:
        # New category needs admin approval first
        invoice_status = "Pending Admin Approval"  # Waiting for admin to approve new category
    else:
        # Existing category - goes directly to Finance Officer for approval (no supplier confirmation needed)
        invoice_status = "Pending Finance Approval"
    
    # Create invoice document
    invoice_id = f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    invoice_data = {
        "invoice_id": invoice_id,
        "supplier_id": user.username,
        "pdf_path": file_path,
        "status": invoice_status,
        "uploaded_at": datetime.now(),
        "line_items": [item.dict() for item in line_items],
        "total_amount": rag_result.get("total_amount", 0.0),
        "processed_by": "RAG Engine",
        "gender": gender,  # Store gender detected from invoice
        "category_requests": [cr["_id"] for cr in category_requests_created] if category_requests_created else [],
        "supplier_confirmed": False,
        "finance_approved": False
    }
    
    # Save to MongoDB
    result = await db.invoices.insert_one(invoice_data)
    invoice_data["_id"] = str(result.inserted_id)
    
    # Log audit
    await db.audit_logs.insert_one({
        "invoice_id": invoice_data["invoice_id"],
        "action": "INVOICE_UPLOADED",
        "user_id": user.username,
        "timestamp": datetime.now(),
        "details": f"Uploaded {file.filename}, {len(line_items)} line items processed"
    })
    
    return {
        "invoice_id": invoice_data["invoice_id"],
        "status": invoice_data["status"],
        "line_items_count": len(line_items),
        "flagged_count": sum(1 for item in line_items if item.flagged),
        "gender": gender,
        "category_requests_count": len(category_requests_created),
        "category_requests": category_requests_created
    }


@app.get("/invoices", response_model=List[dict])
async def get_invoices(
    gender: Optional[str] = Query(None, description="Filter by gender: Male, Female, or Unknown"),
    user: User = Depends(require_role(["Supplier", "Finance", "Administrator"]))
):
    """
    Get all invoices based on user role
    - Suppliers see their own invoices
    - Finance/Admin see all invoices
    - Can filter by gender (Male, Female, Unknown)
    """
    query = {}
    
    if user.role == "Supplier":
        query["supplier_id"] = user.username
    
    # Add gender filter if provided
    if gender:
        query["gender"] = gender
    
    cursor = db.invoices.find(query).sort("uploaded_at", -1)
    invoices = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        invoices.append(doc)
    
    return invoices


@app.get("/invoices/{invoice_id}", response_model=dict)
async def get_invoice(
    invoice_id: str,
    user: User = Depends(require_role(["Supplier", "Finance", "Administrator", "Technical Monitor"]))
):
    """Get a specific invoice by ID"""
    doc = await db.invoices.find_one({"invoice_id": invoice_id})
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invoice {invoice_id} not found"
        )
    
    doc["_id"] = str(doc["_id"])
    return doc


@app.put("/invoices/{invoice_id}/line-items/{item_index}")
async def update_line_item(
    invoice_id: str,
    item_index: int,
    update: LineItemUpdate,
    user: User = Depends(require_role(["Finance", "Administrator"]))
):
    """
    Update a line item's final code
    This triggers the feedback loop to improve RAG
    """
    # Get invoice
    invoice = await db.invoices.find_one({"invoice_id": invoice_id})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    # Get original description for learning
    if item_index >= len(invoice["line_items"]):
        raise HTTPException(status_code=400, detail="Invalid item index")
    
    original_desc = invoice["line_items"][item_index]["description"]
    old_code = invoice["line_items"][item_index]["suggested_code"]
    new_code = update.final_code
    
    # Update the line item
    invoice["line_items"][item_index]["final_code"] = new_code
    invoice["line_items"][item_index]["updated_at"] = datetime.now().isoformat()
    invoice["line_items"][item_index]["updated_by"] = user.username
    
    # Update status if all items are confirmed
    all_confirmed = all(
        item.get("final_code") and item.get("final_code") != ""
        for item in invoice["line_items"]
    )
    if all_confirmed:
        invoice["status"] = "Approved"
    
    # Save to MongoDB
    await db.invoices.update_one(
        {"invoice_id": invoice_id},
        {"$set": invoice}
    )
    
    # FEEDBACK LOOP: Learn from this correction
    if rag and new_code != old_code:
        rag.update_knowledge_base(original_desc, new_code)
        learning_note = "RAG knowledge base updated"
    else:
        learning_note = "RAG not available for learning"
    
    # Log audit
    await db.audit_logs.insert_one({
        "invoice_id": invoice_id,
        "action": "LINE_ITEM_UPDATED",
        "user_id": user.username,
        "timestamp": datetime.now(),
        "details": f"Item {item_index}: {old_code} → {new_code}. {learning_note}",
        "old_code": old_code,
        "new_code": new_code
    })
    
    return {
        "message": "Line item updated successfully",
        "learning": learning_note
    }


@app.post("/invoices/{invoice_id}/approve")
async def approve_invoice(
    invoice_id: str,
    user: User = Depends(require_role(["Finance", "Administrator"]))
):
    """Approve an invoice"""
    invoice = await db.invoices.find_one({"invoice_id": invoice_id})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    # Update status
    await db.invoices.update_one(
        {"invoice_id": invoice_id},
        {"$set": {"status": "Approved", "approved_by": user.username, "approved_at": datetime.now()}}
    )
    
    # Audit log
    await db.audit_logs.insert_one({
        "invoice_id": invoice_id,
        "action": "INVOICE_APPROVED",
        "user_id": user.username,
        "timestamp": datetime.now(),
        "details": f"Invoice approved by {user.username}"
    })
    
    return {"message": "Invoice approved"}


# ============== SERVICE CODES ENDPOINTS ==============

@app.get("/service-codes", response_model=List[dict])
async def get_service_codes(
    user: User = Depends(require_role(["Administrator"])),
    category: Optional[str] = None
):
    """Get all service codes, optionally filtered by category"""
    query = {}
    if category:
        query["category"] = category
    
    cursor = db.service_codes.find(query)
    codes = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        codes.append(doc)
    
    return codes


@app.post("/service-codes/seed")
async def seed_service_codes(
    user: User = Depends(require_role(["Administrator"]))
):
    """Seed service codes from JSON file to MongoDB and ChromaDB"""
    try:
        # Load from file
        seed_file = "../database/service_codes_seed.json"
        with open(seed_file, 'r') as f:
            data = json.load(f)
        
        # Insert to MongoDB
        codes = []
        for item in data:
            codes.append({
                "code": item["code"],
                "description": item["description"],
                "category": item["category"],
                "status": "Active",
                "created_at": datetime.now()
            })
        
        # Clear and insert
        await db.service_codes.delete_many({})
        await db.service_codes.insert_many(codes)
        
        # Also seed to ChromaDB if RAG is available
        if rag:
            service_code_objects = [ServiceCode(**item) for item in data]
            rag.seed_service_codes(service_code_objects)
        
        return {"message": f"Seeded {len(codes)} service codes", "count": len(codes)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/service-codes/stats")
async def get_codes_stats(
    user: User = Depends(require_role(["Administrator"]))
):
    """Get service codes statistics"""
    total = await db.service_codes.count_documents({})
    
    pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}}
    ]
    
    categories = await db.service_codes.aggregate(pipeline).to_list(length=100)
    
    return {
        "total_codes": total,
        "by_category": {cat["_id"]: cat["count"] for cat in categories}
    }


# ============== AUDIT LOG ENDPOINTS ==============

@app.get("/audit-logs", response_model=List[dict])
async def get_audit_logs(
    user: User = Depends(require_role(["Technical Monitor", "Administrator"])),
    invoice_id: Optional[str] = None,
    limit: int = 50
):
    """Get audit logs, optionally filtered by invoice"""
    query = {}
    if invoice_id:
        query["invoice_id"] = invoice_id
    
    cursor = db.audit_logs.find(query).sort("timestamp", -1).limit(limit)
    logs = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        logs.append(doc)
    
    return logs


@app.get("/audit-logs/stats")
async def get_audit_stats(
    user: User = Depends(require_role(["Technical Monitor", "Administrator"]))
):
    """Get audit statistics for monitoring"""
    total_logs = await db.audit_logs.count_documents({})
    
    # Action counts
    pipeline = [
        {"$group": {"_id": "$action", "count": {"$sum": 1}}}
    ]
    actions = await db.audit_logs.aggregate(pipeline).to_list(length=100)
    
    # Invoice status counts
    invoice_pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    statuses = await db.invoices.aggregate(invoice_pipeline).to_list(length=10)
    
    return {
        "total_logs": total_logs,
        "by_action": {a["_id"]: a["count"] for a in actions},
        "invoice_statuses": {s["_id"]: s["count"] for s in statuses}
    }


# ============== RAG ENDPOINTS ==============

@app.get("/rag/stats")
async def get_rag_stats(
    user: User = Depends(require_role(["Administrator"]))
):
    """Get RAG engine statistics"""
    if not rag:
        return {"error": "RAG not initialized"}
    
    stats = rag.get_stats()
    
    # Add MongoDB stats
    stats["mongodb_docs"] = await db.invoices.count_documents({})
    stats["service_codes"] = await db.service_codes.count_documents({})
    
    return stats


@app.post("/rag/seed")
async def rag_seed(
    user: User = Depends(require_role(["Administrator"]))
):
    """Manually trigger RAG vector store seeding"""
    if not rag:
        raise HTTPException(status_code=500, detail="RAG not initialized")
    
    try:
        # Get codes from MongoDB
        cursor = db.service_codes.find({})
        codes = []
        async for doc in cursor:
            codes.append(ServiceCode(
                code=doc["code"],
                description=doc["description"],
                category=doc["category"]
            ))
        
        # Seed to ChromaDB
        rag.seed_service_codes(codes)
        
        return {"message": f"Seeded {len(codes)} codes to ChromaDB"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/suggest")
async def rag_suggest(
    request: Request,
    description: str = Body(..., embed=True)
):
    """Get service code suggestion for a description"""
    if not rag:
        raise HTTPException(status_code=500, detail="RAG not initialized")
    
    try:
        # Get similar codes
        similar_codes = rag.retrieve_similar_codes(description)
        
        # Generate response
        response = rag.generate_response(description, similar_codes)
        
        return {
            "suggested_code": response.suggested_code,
            "confidence_score": response.confidence_score,
            "reasoning": response.reasoning,
            "retrieved_codes": [
                {"code": c.code, "category": c.category, "description": c.description}
                for c in similar_codes
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== CATEGORY REQUEST ENDPOINTS ==============

@app.get("/category-requests", response_model=List[dict])
async def get_category_requests(
    status: Optional[str] = Query(None, description="Filter by status: Pending, Approved, Rejected"),
    user: User = Depends(require_role(["Finance", "Administrator"]))
):
    """
    Get all category requests
    - Finance/Admin can view all requests
    - Can filter by status
    """
    query = {}
    if status:
        query["status"] = status
    
    cursor = db.category_requests.find(query).sort("requested_at", -1)
    requests = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        requests.append(doc)
    
    return requests


@app.get("/category-requests/{request_id}", response_model=dict)
async def get_category_request(
    request_id: str,
    user: User = Depends(require_role(["Finance", "Administrator"]))
):
    """Get a specific category request by ID"""
    doc = await db.category_requests.find_one({"_id": request_id})
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category request {request_id} not found"
        )
    
    doc["_id"] = str(doc["_id"])
    return doc


@app.put("/category-requests/{request_id}/approve")
async def approve_category_request(
    request_id: str,
    user: User = Depends(require_role(["Administrator"]))  # Only Administrator can approve new categories
):
    """
    Approve a category request (Administrator only)
    - Creates new category and service code
    - Stores in vector knowledge base
    - Updates associated invoice with new code
    - Then sends to supplier for confirmation
    """
    # Get the category request
    cat_request = await db.category_requests.find_one({"_id": request_id})
    if not cat_request:
        raise HTTPException(status_code=404, detail="Category request not found")
    
    if cat_request["status"] != "Pending":
        raise HTTPException(status_code=400, detail="Request already processed")
    
    # Create new service code in the database
    new_service_code = {
        "code": cat_request["suggested_code"],
        "description": cat_request["description"],
        "category": cat_request["suggested_category"],
        "status": "Active",
        "created_at": datetime.now(),
        "created_by": user.username
    }
    
    await db.service_codes.insert_one(new_service_code)
    
    # Store in vector knowledge base (RAG)
    if rag:
        from rag_engine import ServiceCode
        rag.seed_service_codes([ServiceCode(
            code=new_service_code["code"],
            description=new_service_code["description"],
            category=new_service_code["category"]
        )])
    
    # Update category request status
    await db.category_requests.update_one(
        {"_id": request_id},
        {"$set": {
            "status": "Approved",
            "resolved_at": datetime.now(),
            "resolved_by": user.username
        }}
    )
    
    # Update the associated invoice with the new service code
    invoice_id = cat_request.get("invoice_id")
    if invoice_id:
        # Find and update line items that were pending
        invoice = await db.invoices.find_one({"invoice_id": invoice_id})
        if invoice:
            updated_line_items = []
            for item in invoice.get("line_items", []):
                if item.get("suggested_code") == "PENDING_APPROVAL":
                    item["suggested_code"] = cat_request["suggested_code"]
                    item["final_code"] = cat_request["suggested_code"]
                    item["flagged"] = False
                    item["reasoning"] = f"Approved: New category '{cat_request['suggested_category']}' created"
                updated_line_items.append(item)
            
            # Update invoice - send to Finance Officer for final approval
            await db.invoices.update_one(
                {"invoice_id": invoice_id},
                {"$set": {
                    "line_items": updated_line_items,
                    "status": "Pending Finance Approval"  # After Admin approves new category, send to Finance
                }}
            )
    
    # Log audit
    await db.audit_logs.insert_one({
        "invoice_id": invoice_id or "N/A",
        "action": "CATEGORY_REQUEST_APPROVED",
        "user_id": user.username,
        "timestamp": datetime.now(),
        "details": f"Approved new category '{cat_request['suggested_category']}' with code {cat_request['suggested_code']}",
        "old_value": None,
        "new_value": cat_request["suggested_code"]
    })
    
    return {
        "message": "Category request approved",
        "new_service_code": new_service_code
    }


@app.put("/category-requests/{request_id}/reject")
async def reject_category_request(
    request_id: str,
    notes: Optional[str] = Body(None),
    user: User = Depends(require_role(["Finance", "Administrator"]))
):
    """
    Reject a category request
    """
    # Get the category request
    cat_request = await db.category_requests.find_one({"_id": request_id})
    if not cat_request:
        raise HTTPException(status_code=404, detail="Category request not found")
    
    if cat_request["status"] != "Pending":
        raise HTTPException(status_code=400, detail="Request already processed")
    
    # Update category request status
    await db.category_requests.update_one(
        {"_id": request_id},
        {"$set": {
            "status": "Rejected",
            "resolved_at": datetime.now(),
            "resolved_by": user.username,
            "notes": notes or "Rejected by finance officer"
        }}
    )
    
    # Update the associated invoice status
    invoice_id = cat_request.get("invoice_id")
    if invoice_id:
        await db.invoices.update_one(
            {"invoice_id": invoice_id},
            {"$set": {"status": "Rejected"}}
        )
    
    # Log audit
    await db.audit_logs.insert_one({
        "invoice_id": invoice_id or "N/A",
        "action": "CATEGORY_REQUEST_REJECTED",
        "user_id": user.username,
        "timestamp": datetime.now(),
        "details": f"Rejected category request: {notes or 'No reason provided'}",
        "old_value": None,
        "new_value": "Rejected"
    })
    
    return {
        "message": "Category request rejected"
    }


# ============== SUPPLIER CONFIRMATION ENDPOINTS ==============

@app.put("/invoices/{invoice_id}/confirm")
async def supplier_confirm_invoice(
    invoice_id: str,
    request: SupplierConfirmRequest,
    user: User = Depends(require_role(["Supplier"]))
):
    """
    Supplier confirms or rejects the invoice suggestions
    - If confirmed: Invoice goes to Finance Officer for approval
    - If rejected: Invoice is marked as rejected
    """
    # Get the invoice
    invoice = await db.invoices.find_one({"invoice_id": invoice_id})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    # Verify supplier owns this invoice
    if invoice.get("supplier_id") != user.username:
        raise HTTPException(status_code=403, detail="Not authorized to confirm this invoice")
    
    if invoice.get("status") not in ["Pending Supplier Confirmation", "Pending Admin Approval"]:
        raise HTTPException(status_code=400, detail="Invoice is not pending confirmation")
    
    if request.confirmed:
        # Update invoice status to pending finance officer approval
        new_status = "Pending Finance Approval"
        await db.invoices.update_one(
            {"invoice_id": invoice_id},
            {"$set": {
                "status": new_status,
                "supplier_confirmed": True,
                "supplier_confirmed_at": datetime.now()
            }}
        )
        
        # Log audit
        await db.audit_logs.insert_one({
            "invoice_id": invoice_id,
            "action": "INVOICE_CONFIRMED_BY_SUPPLIER",
            "user_id": user.username,
            "timestamp": datetime.now(),
            "details": f"Supplier confirmed the invoice suggestions. Invoice now pending finance officer approval."
        })
        
        return {
            "message": "Invoice confirmed. Now pending Finance Officer approval.",
            "status": new_status
        }
    else:
        # Supplier rejected
        await db.invoices.update_one(
            {"invoice_id": invoice_id},
            {"$set": {
                "status": "Rejected by Supplier",
                "supplier_confirmed": False,
                "supplier_notes": request.notes or "Rejected by supplier"
            }}
        )
        
        # Log audit
        await db.audit_logs.insert_one({
            "invoice_id": invoice_id,
            "action": "INVOICE_REJECTED_BY_SUPPLIER",
            "user_id": user.username,
            "timestamp": datetime.now(),
            "details": f"Supplier rejected the invoice. Reason: {request.notes or 'Not provided'}"
        })
        
        return {
            "message": "Invoice rejected by supplier.",
            "status": "Rejected by Supplier"
        }


# ============== FINANCE OFFICER APPROVAL ENDPOINTS ==============

@app.put("/invoices/{invoice_id}/finance-approve")
async def finance_approve_invoice(
    invoice_id: str,
    request_body: dict = {},
    user: User = Depends(require_role(["Finance", "Administrator"]))
):
    notes = request_body.get("notes")
    """
    Finance Officer approves an invoice
    - Only Finance Officer or Administrator can approve
    """
    # Get the invoice
    invoice = await db.invoices.find_one({"invoice_id": invoice_id})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    if invoice.get("status") != "Pending Finance Approval":
        raise HTTPException(status_code=400, detail="Invoice is not pending finance approval")
    
    # Update invoice status to approved
    await db.invoices.update_one(
        {"invoice_id": invoice_id},
        {"$set": {
            "status": "Approved",
            "finance_approved": True,
            "finance_approved_by": user.username,
            "finance_approved_at": datetime.now(),
            "finance_notes": notes
        }}
    )
    
    # Log audit
    await db.audit_logs.insert_one({
        "invoice_id": invoice_id,
        "action": "INVOICE_APPROVED_BY_FINANCE",
        "user_id": user.username,
        "timestamp": datetime.now(),
        "details": f"Invoice approved by Finance Officer. Notes: {notes or 'None'}"
    })
    
    return {
        "message": "Invoice approved successfully.",
        "status": "Approved"
    }


@app.put("/invoices/{invoice_id}/finance-reject")
async def finance_reject_invoice(
    invoice_id: str,
    request_body: dict = {},
    user: User = Depends(require_role(["Finance", "Administrator"]))
):
    notes = request_body.get("notes")
    """
    Finance Officer rejects an invoice
    """
    # Get the invoice
    invoice = await db.invoices.find_one({"invoice_id": invoice_id})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    if invoice.get("status") != "Pending Finance Approval":
        raise HTTPException(status_code=400, detail="Invoice is not pending finance approval")
    
    # Update invoice status to rejected
    await db.invoices.update_one(
        {"invoice_id": invoice_id},
        {"$set": {
            "status": "Rejected",
            "finance_approved": False,
            "finance_rejected_by": user.username,
            "finance_rejected_at": datetime.now(),
            "finance_notes": notes or "Rejected by Finance Officer"
        }}
    )
    
    # Log audit
    await db.audit_logs.insert_one({
        "invoice_id": invoice_id,
        "action": "INVOICE_REJECTED_BY_FINANCE",
        "user_id": user.username,
        "timestamp": datetime.now(),
        "details": f"Invoice rejected by Finance Officer. Reason: {notes or 'Not provided'}"
    })
    
    return {
        "message": "Invoice rejected.",
        "status": "Rejected"
    }


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)