"""
Invoice OCR Extraction Pipeline
================================
Input: Image or PDF
Pipeline: PDF→Image (PyMuPDF) → OCR (DocTR) → Field Extraction (Ollama/LLM) → Q&A → Output
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# 1. PDF → Image Conversion  (PyMuPDF)
# ─────────────────────────────────────────────

def convert_pdf_to_images(pdf_path: str, output_dir: str = "temp_pages", dpi: int = 200) -> list[str]:
    """
    Convert a PDF into a list of image paths (one per page) using PyMuPDF (fitz).

    Args:
        pdf_path:   Path to the PDF file.
        output_dir: Directory to save rendered page images.
        dpi:        Resolution in DPI (default 200; higher = better OCR, more RAM).

    Returns:
        List of PNG file paths, one per page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    os.makedirs(output_dir, exist_ok=True)
    zoom = dpi / 72.0          # 72 pt/inch is PDF's native resolution unit
    mat = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    print(f"[PDF->Image] PyMuPDF * {len(image_paths)} page(s) @ {dpi} DPI")
    return image_paths


# ─────────────────────────────────────────────
# 2. OCR Engine — DocTR only
# ─────────────────────────────────────────────

def run_doctr(image_paths: list[str]) -> str:
    """Run DocTR OCR on a list of image paths."""
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError:
        raise ImportError("DocTR not installed. Run: pip install python-doctr[torch]")

    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images(image_paths)
    result = model(doc)
    text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                text += " ".join([w.value for w in line.words]) + "\n"
    return text.strip()


def run_ocr(image_paths: list[str]) -> str:
    """Run DocTR OCR on the provided image paths."""
    print("[OCR] Running engine: DocTR")
    return run_doctr(image_paths)


# ─────────────────────────────────────────────
# 3. Field Extraction (from OCR text via LLM)
# ─────────────────────────────────────────────

INVOICE_FIELDS = {
    "supplier_name": "Supplier Name",
    "supplier_phone": "Supplier Phone Number",
    "supplier_company_registration": {
        "phone_no": "Supplier Company Phone No",
        "type": "Supplier Company Type"
    },
    "customer_company_registration": {
        "phone_no": "Customer Company Phone No",
        "type": "Customer Company Type"
    },
    "invoice_number": "Invoice Number",
    "date": "Invoice Date",
    "total_amount": "Total Amount",
    "total_net": "Total Net",
    "total_tax": "Total Tax",
    "taxes": {
        "rate": "Tax Rate",
        "base": "Tax Base",
        "amount": "Tax Amount"
    },
    "line_items": [
        {
            "description": "Item Description",
            "quantity": "Quantity",
            "unit_price": "Unit Price",
            "total_price": "Total Price",
            "product_code": "Product Code",
            "tax_amount": "Item Tax Amount",
            "tax_rate": "Item Tax Rate",
            "unit_measure": "Unit of Measure"
        }
    ],
    "document_type": "Document Type",
    "locale": {
        "language": "Language",
        "country": "Country",
        "currency": "Currency"
    },
    "customer_name": "Customer Name",
    "customer_address": {
        "address": "Street Address",
        "street_no": "Street Number",
        "street_name": "Street Name",
        "po_box": "PO Box",
        "city": "City",
        "postal_code": "Postal Code",
        "state": "State",
        "country": "Country"
    },
    "shipping_address": "Shipping Address (same as customer if not different)",
    "billing_address": "Billing Address (same as customer if not different)",
    "supplier_address": "Supplier Address (same as customer format)",
    "due_date": "Due Date",
    "purchase_order": "Purchase Order Number",
    "reference_number": "Reference Number",
    "payment_date": "Payment Date",
    "supplier_payment_details": {
        "iban": "IBAN",
        "swift": "SWIFT/BIC",
        "account_number": "Account Number",
        "routing_number": "Routing Number"
    },
    "supplier_website": "Supplier Website",
    "supplier_email": "Supplier Email",
    "customer_id": "Customer ID"
}


EXTRACTION_PROMPT = """Extract invoice data from the OCR text below.
CRITICAL: Return ONLY a raw JSON object. No markdown. No ```json. No explanation. No text before or after the JSON.
Start your response with {{ and end with }}. Use null for missing fields.

Required JSON structure:
{{
  "invoice_number": null,
  "date": null,
  "due_date": null,
  "document_type": null,
  "reference_number": null,
  "purchase_order": null,
  "payment_date": null,
  "total_amount": null,
  "total_net": null,
  "total_tax": null,
  "taxes": {{"rate": null, "base": null, "amount": null}},
  "supplier_name": null,
  "supplier_phone": null,
  "supplier_email": null,
  "supplier_website": null,
  "supplier_company_registration": {{"phone_no": null, "type": null}},
  "supplier_address": {{"address": null, "street_no": null, "street_name": null, "po_box": null, "city": null, "postal_code": null, "state": null, "country": null}},
  "supplier_payment_details": {{"iban": null, "swift": null, "account_number": null, "routing_number": null}},
  "customer_name": null,
  "customer_id": null,
  "customer_company_registration": {{"phone_no": null, "type": null}},
  "customer_address": {{"address": null, "street_no": null, "street_name": null, "po_box": null, "city": null, "postal_code": null, "state": null, "country": null}},
  "shipping_address": null,
  "billing_address": null,
  "locale": {{"language": null, "country": null, "currency": null}},
  "line_items": []
}}

OCR TEXT:
{ocr_text}

Remember: Output ONLY the JSON object, nothing else."""


# ── All extractable fields registry ──────────────────────────
ALL_FIELDS_REGISTRY = {
    # category -> list of (field_key, display_label, json_path, default_on)
    "Invoice": [
        ("invoice_number",  "Invoice Number",   "invoice_number",  True),
        ("date",            "Invoice Date",      "date",            True),
        ("due_date",        "Due Date",          "due_date",        True),
        ("document_type",   "Document Type",     "document_type",   True),
        ("reference_number","Reference Number",  "reference_number",True),
        ("purchase_order",  "Purchase Order",    "purchase_order",  False),
        ("payment_date",    "Payment Date",      "payment_date",    False),
    ],
    "Amounts": [
        ("total_amount",    "Total Amount",      "total_amount",    True),
        ("total_net",       "Total Net",         "total_net",       True),
        ("total_tax",       "Total Tax",         "total_tax",       True),
        ("tax_rate",        "Tax Rate",          "taxes.rate",      False),
        ("tax_base",        "Tax Base",          "taxes.base",      False),
        ("tax_amount_val",  "Tax Amount",        "taxes.amount",    False),
    ],
    "Supplier": [
        ("supplier_name",   "Supplier Name",     "supplier_name",   True),
        ("supplier_phone",  "Supplier Phone",    "supplier_phone",  True),
        ("supplier_email",  "Supplier Email",    "supplier_email",  True),
        ("supplier_website","Supplier Website",  "supplier_website",False),
        ("sup_reg_type",    "Supplier Co. Type", "supplier_company_registration.type", False),
        ("sup_reg_phone",   "Supplier Co. Phone","supplier_company_registration.phone_no", False),
        ("supplier_address","Supplier Address",  "supplier_address",False),
        ("sup_iban",        "IBAN",              "supplier_payment_details.iban",    False),
        ("sup_swift",       "SWIFT / BIC",       "supplier_payment_details.swift",   False),
        ("sup_account",     "Account Number",    "supplier_payment_details.account_number", False),
        ("sup_routing",     "Routing Number",    "supplier_payment_details.routing_number", False),
    ],
    "Customer": [
        ("customer_name",   "Customer Name",     "customer_name",   True),
        ("customer_id",     "Customer ID",       "customer_id",     True),
        ("customer_address","Customer Address",  "customer_address",True),
        ("shipping_address","Shipping Address",  "shipping_address",False),
        ("billing_address", "Billing Address",   "billing_address", False),
        ("cust_reg_type",   "Customer Co. Type", "customer_company_registration.type", False),
        ("cust_reg_phone",  "Customer Co. Phone","customer_company_registration.phone_no", False),
    ],
    "Line Items": [
        ("line_items",      "Line Items (all)",  "line_items",      True),
    ],
    "Locale": [
        ("locale_language", "Language",          "locale.language", False),
        ("locale_country",  "Country",           "locale.country",  False),
        ("locale_currency", "Currency",          "locale.currency", True),
    ],
}

DEFAULT_FIELDS = [
    fk for cat_fields in ALL_FIELDS_REGISTRY.values()
    for fk, _, _, on in cat_fields if on
]


def build_dynamic_extraction_prompt(ocr_text: str, selected_keys: list) -> str:
    """Build extraction prompt for only the user-selected fields."""
    key_info = {}
    for cat_fields in ALL_FIELDS_REGISTRY.values():
        for fk, label, path, _ in cat_fields:
            key_info[fk] = {"label": label, "path": path}

    lines = []
    for fk in selected_keys:
        info = key_info.get(fk, {})
        path = info.get("path", fk)
        label = info.get("label", fk)
        if "." in path:
            lines.append(f'  "{path}": null  // {label}')
        else:
            lines.append(f'  "{fk}": null  // {label}')

    fields_str = "{\n" + ",\n".join(lines) + "\n}"

    return f"""Extract invoice data from the OCR text. Return ONLY raw JSON, no markdown, no explanation.
Start with {{ and end with }}. Use null for missing fields.

Extract ONLY these fields:
{fields_str}

For line_items use this structure:
[{{"description": null, "quantity": null, "unit_price": null, "total_price": null,
   "product_code": null, "tax_amount": null, "tax_rate": null, "unit_measure": null}}]

OCR TEXT:
{ocr_text}

Output ONLY the JSON object."""


def extract_fields_with_llm(ocr_text: str, provider: str = "ollama", model: str = None,
                             api_key: str = None, selected_keys: list = None) -> dict:
    """Use an LLM to extract structured invoice fields from raw OCR text."""
    if selected_keys:
        prompt = build_dynamic_extraction_prompt(ocr_text, selected_keys)
    else:
        prompt = EXTRACTION_PROMPT.format(ocr_text=ocr_text)

    if provider == "openai":
        return _extract_openai(prompt, model or "gpt-4o", api_key)
    elif provider == "anthropic":
        return _extract_anthropic(prompt, model or "claude-3-5-sonnet-20241022", api_key)
    elif provider == "ollama":
        return _extract_ollama(prompt, model or "llama3")
    else:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose: openai, anthropic, ollama")


def _extract_openai(prompt: str, model: str, api_key: str = None) -> dict:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except ImportError:
        raise ImportError("OpenAI not installed. Run: pip install openai")


def _extract_anthropic(prompt: str, model: str, api_key: str = None) -> dict:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except ImportError:
        raise ImportError("Anthropic not installed. Run: pip install anthropic")


def _parse_json_robust(text: str) -> dict:
    """
    Multi-strategy JSON parser that handles common LLM output quirks:
    - Wrapped in ```json ... ``` fences
    - Leading/trailing prose around the JSON
    - Trailing commas before } or ]
    - Single quotes instead of double quotes
    """
    import re

    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract first { ... } block (handles leading prose)
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Strategy 4: fix trailing commas then re-parse
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Strategy 5: replace single quotes with double quotes
        try:
            single_fixed = candidate.replace("'", '"')
            return json.loads(single_fixed)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON from LLM response", text, 0)


def _extract_ollama(prompt: str, model: str) -> dict:
    try:
        import requests
    except ImportError:
        raise ImportError("Requests not installed. Run: pip install requests")

    system = ("You are an invoice data extraction assistant. "
              "You MUST respond with ONLY a valid JSON object. "
              "Do NOT include any markdown, code fences, or explanation. "
              "Start your response with { and end with }.")

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4096}
        },
        timeout=300
    )
    response.raise_for_status()
    raw = response.json().get("response", "").strip()

    try:
        return _parse_json_robust(raw)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw_response": raw}


# ─────────────────────────────────────────────
# 4. LLM Q&A on extracted data
# ─────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are an intelligent invoice assistant.
You have access to the following extracted invoice data (JSON) and the raw OCR text.
Answer user questions clearly and concisely based on this data.

EXTRACTED DATA:
{extracted_json}

RAW OCR TEXT:
{ocr_text}
"""


def answer_question(question: str, extracted_data: dict, ocr_text: str,
                    provider: str = "ollama", model: str = None, api_key: str = None) -> str:
    """Answer a natural language question about the extracted invoice data."""
    system = QA_SYSTEM_PROMPT.format(
        extracted_json=json.dumps(extracted_data, indent=2),
        ocr_text=ocr_text
    )

    if provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model or "gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": question}
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("OpenAI not installed.")

    elif provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model or "claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": question}]
            )
            return message.content[0].text
        except ImportError:
            raise ImportError("Anthropic not installed.")

    elif provider == "ollama":
        try:
            import requests
            full_prompt = system + f"\n\nQuestion: {question}"
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model or "llama3", "prompt": full_prompt, "stream": False}
            )
            return response.json()["response"]
        except ImportError:
            raise ImportError("Requests not installed.")

    raise ValueError(f"Unknown provider: {provider}")


# ─────────────────────────────────────────────
# 5. Main Pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    dpi: int = 200,
    llm_provider: str = "ollama",
    llm_model: str = None,
    llm_api_key: str = None,
    extract_fields: bool = True,
    questions: list[str] = None,
    output_path: str = "output.json",
    temp_dir: str = "temp_pages"
) -> dict:
    """
    Full pipeline:
    1. Input (Image or PDF)
    2. PDF -> Image conversion via PyMuPDF (if needed)
    3. OCR via DocTR
    4. (Optional) Field extraction via LLM
    5. (Optional) Q&A
    6. Output JSON
    """
    input_path = str(input_path)
    result = {
        "input_file": input_path,
        "ocr_engine": "doctr",
        "ocr_text": "",
        "extracted_fields": {},
        "qa_results": []
    }

    suffix = Path(input_path).suffix.lower()
    if suffix == ".pdf":
        print(f"[Pipeline] PDF detected -> converting to images (PyMuPDF @ {dpi} DPI)...")
        image_paths = convert_pdf_to_images(input_path, temp_dir, dpi=dpi)
    elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
        print(f"[Pipeline] Image detected: {input_path}")
        image_paths = [input_path]
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF or image.")

    ocr_text = run_ocr(image_paths)
    result["ocr_text"] = ocr_text
    print(f"[OCR] Extracted {len(ocr_text)} characters")

    if extract_fields:
        print(f"[LLM] Extracting invoice fields using {llm_provider}...")
        extracted = extract_fields_with_llm(ocr_text, llm_provider, llm_model, llm_api_key)
        result["extracted_fields"] = extracted
        print(f"[LLM] Extracted {len(extracted)} top-level fields")

    if questions:
        for q in questions:
            print(f"[Q&A] Question: {q}")
            answer = answer_question(q, result["extracted_fields"], ocr_text, llm_provider, llm_model, llm_api_key)
            result["qa_results"].append({"question": q, "answer": answer})
            print(f"[Q&A] Answer: {answer}")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Output saved to: {output_path}")

    return result


# ─────────────────────────────────────────────
# 6. CLI Interface
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Invoice OCR Extraction Pipeline -- DocTR + PyMuPDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py invoice.pdf
  python main.py invoice.pdf --llm openai --api-key sk-...
  python main.py invoice.pdf --llm ollama --model llama3
  python main.py invoice.pdf --no-extract
  python main.py invoice.pdf --dpi 300
  python main.py invoice.pdf --fields invoice_number date total_amount
  python main.py invoice.pdf -q "What is total?" -q "Who is supplier?"
  python main.py dummy --list-models
  python main.py dummy --list-fields
        """
    )
    parser.add_argument("input", help="Path to invoice image or PDF")
    parser.add_argument("--llm", choices=["openai", "anthropic", "ollama"], default="ollama",
                        help="LLM provider for field extraction and Q&A (default: ollama)")
    parser.add_argument("--model", help="LLM model name (optional, uses sensible default)")
    parser.add_argument("--api-key", help="API key for LLM provider (or set env var)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="DPI for PDF->image rendering via PyMuPDF (default: 200)")
    parser.add_argument("--no-extract", action="store_true", help="Skip field extraction (OCR only)")
    parser.add_argument("--fields", nargs="+", metavar="FIELD",
                        help="Extract only specific fields (space-separated keys)")
    parser.add_argument("--question", "-q", action="append", dest="questions",
                        help="Question(s) to ask about the invoice (can use multiple times)")
    parser.add_argument("--output", "-o", default="output.json",
                        help="Output JSON path (default: output.json)")
    parser.add_argument("--temp-dir", default="temp_pages",
                        help="Dir for temp page images (default: temp_pages)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama llama models and exit")
    parser.add_argument("--list-fields", action="store_true",
                        help="List all extractable fields and exit")

    args = parser.parse_args()

    if args.list_fields:
        print("Extractable fields:")
        for cat, fields in ALL_FIELDS_REGISTRY.items():
            print(f"\n  [{cat}]")
            for fk, label, _, on in fields:
                default = " (default)" if on else ""
                print(f"    {fk:<22} {label}{default}")
        return

    if args.list_models:
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", []) if "llama" in m["name"].lower()]
            print("Available Ollama llama models:")
            for m in models:
                print(f"  {m}")
        except Exception as e:
            print(f"Could not reach Ollama: {e}")
        return

    result = run_pipeline(
        input_path=args.input,
        dpi=args.dpi,
        llm_provider=args.llm,
        llm_model=args.model,
        llm_api_key=args.api_key,
        extract_fields=not args.no_extract,
        questions=args.questions or [],
        output_path=args.output,
        temp_dir=args.temp_dir,
    )

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    fields = result.get("extracted_fields", {})
    if fields:
        print(json.dumps(fields, indent=2, ensure_ascii=False))
    else:
        print("[OCR Text Preview]")
        print(result["ocr_text"][:1000])


if __name__ == "__main__":
    main()
