# INVOICE - NEW CATEGORY TEST

## TechAssist Solutions - IT Support Services
**Invoice Number:** INV-2026-0324-NEW01
**Date:** 24/03/2026
**Supplier:** TechAssist Solutions Pty Ltd
**ABN:** 12 345 678 901

---

## Client Details
**Client Name:** John Smith
**Address:** 123 Main Street, Sydney NSW 2000
**NDIS Number:** 123456789

---

## Service Details

| Date | Description | Hours | Rate | Amount |
|------|-------------|-------|------|--------|
| 2026-03-20 | IT Support and Computer Training - basic skills lesson | 2.0 | $85.00 | $170.00 |
| 2026-03-21 | Software Installation and Configuration - Microsoft Office setup | 1.0 | $95.00 | $95.00 |
| 2026-03-22 | Wifi Network Setup and Troubleshooting - home network | 1.5 | $90.00 | $135.00 |

---

## Summary
Subtotal: $400.00
GST (10%): $40.00
**Total: $440.00**

---

## Payment Terms
Payment due within 30 days
Bank Transfer: BSB 123-456, Account 98765432

---

**Authorised Signatory:** _______________
**Date:** _______________

---

## Why This Triggers New Category

The service descriptions in this invoice contain keywords that do NOT match any existing category in the vector store:

### Existing Categories in Vector Store:
- **Personal Care**: shower, bath, dressing, grooming, hygiene, toilet, mobility
- **Nursing**: nurse, nursing, medication, wound, health, medical, clinical
- **Domestic Assistance**: cleaning, laundry, vacuum, mop, dust, house, dish
- **Gardening**: garden, lawn, mow, weed, prune, yard, gutter
- **Meals**: meal, food, cooking, lunch, dinner, breakfast, delivery
- **Transport**: transport, drive, car, appointment, travel, bus, taxi
- **Allied Health**: physio, physiotherapy, occupational, speech, podiatry

### New Keywords in This Invoice:
- IT Support
- Computer Training
- Software Installation
- Wifi Network Setup
- Network Troubleshooting

### Expected System Response:
When processed, this invoice should return:
- `needs_approval: true`
- `suggested_category: "Technology Assistance"` or `"IT Support"`
- `suggested_code_format: "01_XXX_XXXX_X_X"`
- Request admin approval for the new category

---

## Notes
This is a test invoice for the new category approval workflow. The services provided (IT support, computer training, network setup) are not covered by any existing NDIS service categories in the current system.
