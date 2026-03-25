from fpdf import FPDF
import os

def create_invoice(filename, invoice_no, client_name, items):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="INVOICE", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Invoice Number: {invoice_no}", ln=True)
    pdf.cell(200, 10, txt=f"Date: 2026-03-25", ln=True)
    pdf.cell(200, 10, txt=f"Client Name: {client_name}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(140, 10, txt="Description", border=1)
    pdf.cell(40, 10, txt="Amount", border=1, ln=True, align='R')
    
    pdf.set_font("Arial", '', 11)
    total = 0
    for desc, amt in items:
        pdf.cell(140, 10, txt=desc, border=1)
        pdf.cell(40, 10, txt=f"${amt:.2f}", border=1, ln=True, align='R')
        total += amt
        
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(140, 10, txt="TOTAL", align='R')
    pdf.cell(40, 10, txt=f"${total:.2f}", ln=True, align='R')
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Thank you for your business!", ln=True, align='C')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pdf.output(filename)
    print(f"Created {filename}")

# Define sample invoices
upload_dir = r"c:\Users\HP\OneDrive\Desktop\invoice_mapper\backend\uploads"

# Sample 1: Home Care (Male)
create_invoice(
    os.path.join(upload_dir, "sample_invoice_homecare_john.pdf"),
    "INV-1001",
    "John Smith",
    [
        ("Personal Care - Assistance with Showering & Dressing", 65.00),
        ("Domestic Assistance - General Household Cleaning", 55.00)
    ]
)

# Sample 2: Nursing & Transport (Female)
create_invoice(
    os.path.join(upload_dir, "sample_invoice_nursing_mary.pdf"),
    "INV-2002",
    "Mary Jane",
    [
        ("Nursing Care - Medication Administration", 85.00),
        ("Transport - Medical Appointment Shuttle Service", 45.00)
    ]
)

# Sample 3: Gardening (Male)
create_invoice(
    os.path.join(upload_dir, "sample_invoice_gardening_robert.pdf"),
    "INV-3003",
    "Robert Brown",
    [
        ("Garden Maintenance - Lawn Mowing and Edging", 50.00),
        ("Yard Cleanup - Hedge Trimming and Waste Removal", 75.00)
    ]
)
