================================================================================
                    SAMPLE INVOICE FORMAT FOR UPLOAD
                    Invoice Mapper Application - NDIS
================================================================================

This document shows the recommended format for invoices to be uploaded to the
system. The RAG engine will process the PDF and extract line items to map them
to appropriate service codes.

================================================================================
                         RECOMMENDED PDF STRUCTURE
================================================================================

INVOICE HEADER (should include):
---------------------------------
• Invoice Number: INV-2024-001234
• Invoice Date: 15/03/2024
• Due Date: 30/03/2024
• Supplier Name: [Your Company Name]
• Supplier ABN: 12 345 678 901
• Customer/Participant Name: John Doe
• Customer NDIS Number: 1234-5678-90
• Customer Plan Manager: [Plan Manager Name]

INVOICE DETAILS (line items table):
-----------------------------------
| Line | Service Description                    | Qty | Rate    | Amount  |
|------|----------------------------------------|-----|---------|---------|
| 1    | Personal Care - Morning assist        | 2   | $65.00  | $130.00 |
| 2    | Assistance with showering & dressing | 1   | $55.00  | $55.00  |
| 3    | Domestic cleaning - Standard home     | 2   | $45.00  | $90.00  |
| 4    | Garden maintenance - Hourly rate       | 1   | $50.00  | $50.00  |
| 5    | Transport - Medical appointment       | 2   | $35.00  | $70.00  |
| 6    | Meal preparation - Lunch service       | 5   | $25.00  | $125.00 |
| 7    | Nursing - Wound dressing change       | 1   | $85.00  | $85.00  |

SUBTOTAL: $605.00
GST (10%): $60.50
TOTAL: $665.50

FOOTER:
-------
• Payment Terms: Net 14 days
• Bank Details: [Bank Name] BSB: 123-456 Account: 12345678
• Reference: INV-2024-001234

================================================================================
                         SERVICE CODE MAPPING REFERENCE
================================================================================

The system recognizes the following service code categories. Your invoice
descriptions should align with these categories for best matching:

PERSONAL CARE:
--------------
• PC001 - Assistance with personal hygiene (bathing, dressing)
• PC002 - Help with meal preparation and feeding
• PC003 - Mobility and transfer assistance
• PC004 - Overnight care and supervision
• PC005 - Community access assistance

NURSING:
--------
• NU001 - Nursing care for wound dressing
• NU002 - Medication administration
• NU003 - Health monitoring and observations
• NU004 - Catheter care
• NU005 - Complex nursing procedures

DOMESTIC ASSISTANCE:
--------------------
• DA001 - House cleaning and laundry services
• DA002 - Shopping and errands
• DA003 - Meal preparation
• DA004 - Ironing and clothing maintenance
• DA005 - General household tasks

GARDENING:
----------
• GD001 - Garden maintenance (hourly)
• GD002 - Lawn mowing and edging
• GD003 - Hedge trimming
• GD004 - General yard cleanup
• GD005 - Minor garden repairs

MEALS:
------
• ME001 - Meal delivery
• ME002 - Meal preparation at home
• ME003 - Special dietary requirements
• ME004 - Grocery shopping for meals

TRANSPORT:
----------
• TR001 - Transport to medical appointments
• TR002 - Transport to social activities
• TR003 - Transport to shopping
• TR004 - Emergency transport
• TR005 - Community transport

ALLIED HEALTH:
--------------
• AH001 - Physiotherapy sessions
• AH002 - Occupational therapy
• AH003 - Speech therapy
• AH004 - Psychology services
• AH005 - Exercise physiology

================================================================================
                         SAMPLE JSON OUTPUT (after RAG processing)
================================================================================

{
  "invoice_id": "INV-2024-001234",
  "supplier_id": "supplier001",
  "status": "Pending Review",
  "uploaded_at": "2024-03-15T10:30:00Z",
  "total_amount": 665.50,
  "line_items": [
    {
      "description": "Personal Care - Morning assist",
      "suggested_code": "PC001",
      "confidence_score": 0.95,
      "reasoning": "Description matches 'Assistance with personal hygiene' service code",
      "flagged": false,
      "final_code": null
    },
    {
      "description": "Assistance with showering & dressing",
      "suggested_code": "PC001",
      "confidence_score": 0.92,
      "reasoning": "Matches personal hygiene assistance code",
      "flagged": false,
      "final_code": null
    },
    {
      "description": "Domestic cleaning - Standard home",
      "suggested_code": "DA001",
      "confidence_score": 0.98,
      "reasoning": "Direct match to house cleaning service code",
      "flagged": false,
      "final_code": null
    },
    {
      "description": "Garden maintenance - Hourly rate",
      "suggested_code": "GD001",
      "confidence_score": 0.90,
      "reasoning": "Matches garden maintenance category",
      "flagged": false,
      "final_code": null
    },
    {
      "description": "Transport - Medical appointment",
      "suggested_code": "TR001",
      "confidence_score": 0.97,
      "reasoning": "Direct match to medical transport code",
      "flagged": false,
      "final_code": null
    },
    {
      "description": "Meal preparation - Lunch service",
      "suggested_code": "ME002",
      "confidence_score": 0.88,
      "reasoning": "Matches meal preparation at home code",
      "flagged": false,
      "final_code": null
    },
    {
      "description": "Nursing - Wound dressing change",
      "suggested_code": "NU001",
      "confidence_score": 0.99,
      "reasoning": "Direct match to nursing wound dressing code",
      "flagged": false,
      "final_code": null
    }
  ],
  "processed_by": "RAG Engine"
}

================================================================================
                         BEST PRACTICES FOR INVOICE SUBMISSION
================================================================================

1. CLEAR DESCRIPTIONS: Write line item descriptions that clearly describe
   the service provided. Avoid vague terms like "Care" or "Assistance".

2. USE STANDARD CODES: Reference NDIS support categories in your descriptions
   for better automatic matching.

3. INCLUDE DATES: Each line item should include the service date(s).

4. PROPER FORMAT: Ensure the PDF is readable (not scanned images of prints).

5. COMPLETE INFORMATION: Include all participant details required for
   claim verification.

6. MATCH CATEGORIES: Align your descriptions with the system categories:
   - Personal Care
   - Nursing
   - Domestic Assistance
   - Gardening
   - Meals
   - Transport
   - Allied Health

================================================================================