import fitz  # PyMuPDF

pdf_path = "./thesis.pdf"
md_path = "./thesis.md"

# Otwórz PDF
pdf_document = fitz.open(pdf_path)

# Wyciągnij tekst
full_text = ""
for page in pdf_document:
    full_text += page.get_text()

# Zapisz do pliku .md
with open(md_path, "w", encoding="utf-8") as md_file:
    md_file.write(full_text)

md_path
