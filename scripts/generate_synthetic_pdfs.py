# First install: pip install fpdf2
from fpdf import FPDF
import os

os.makedirs("sample_pdfs", exist_ok=True)

company_names = ["Acme Corp", "Apex Financial", "Starlight Inc", "Global Close LLC"]

for i in range(1, 51):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    company = company_names[i % len(company_names)]
    q_revenue = 10.5 + (i * 0.4)
    q_expenses = 4.2 + (i * 0.1)

    # Render layout text and structured tabular data
    pdf.cell(200, 10, text=f"{company} - Q{(i % 4) + 1} Financial Report", new_x="LMARGIN",
             new_y="NEXT", align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10,
                   text=f"Document ID: FIN-2026-{i:04d}\nOperating Summary for fiscal period.")
    pdf.ln(5)

    # Simple financial table format
    pdf.cell(90, 10, text="Metric Category", border=1)
    pdf.cell(90, 10, text="Amount (USD)", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(90, 10, text="Total Revenue", border=1)
    pdf.cell(90, 10, text=f"${q_revenue:.2f}M", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(90, 10, text="Operating Expenses", border=1)
    pdf.cell(90, 10, text=f"${q_expenses:.2f}M", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.output(f"sample_pdfs/financial_report_{i}.pdf")

print("Generated 50 synthetic financial PDFs in ./sample_pdfs/")