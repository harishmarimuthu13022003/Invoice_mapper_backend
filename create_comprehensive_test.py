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
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(140, 10, txt="Service Description", border=1)
    pdf.cell(40, 10, txt="Total Amount", border=1, ln=True, align='R')
    
    pdf.set_font("Arial", '', 9)
    total = 0
    for desc, amt in items:
        # Multi-line cell for description if needed
        pdf.cell(140, 10, txt=desc, border=1)
        pdf.cell(40, 10, txt=f"${amt:.2f}", border=1, ln=True, align='R')
        total += amt
        
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(140, 10, txt="TOTAL", align='R')
    pdf.cell(40, 10, txt=f"${total:.2f}", ln=True, align='R')
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pdf.output(filename)
    print(f"Created {filename}")

# Define a comprehensive test invoice with all requested categories
upload_dir = r"c:\Users\HP\OneDrive\Desktop\invoice_mapper\backend\uploads"

create_invoice(
    os.path.join(upload_dir, "sample_invoice_comprehensive_test.pdf"),
    "INV-9999",
    "Emma Thompson",
    [
        ("Assisted with morning shower and hair wash", 45.00),
        ("Help with dressing and grooming", 30.00),
        ("Medication prompting and supervision", 25.00),
        ("Toileting assistance during day", 20.00),
        ("General house cleaning and vacuuming", 50.00),
        ("Weekly laundry, folding and ironing", 40.00),
        ("Unaccompanied grocery shopping trip", 35.00),
        ("Kitchen and bathroom deep scrub", 60.00),
        ("Lawn mowing and edge trimming", 55.00),
        ("Gutter cleaning and leaf removal", 80.00),
        ("Changing lightbulbs and smoke alarm battery", 30.00),
        ("Pruning garden hedges and yard cleanup", 45.00)
    ]
)
