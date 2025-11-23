import pdfplumber

def extract_pdf_text(path):
    result = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                result.append(text)
    return "\n".join(result)
