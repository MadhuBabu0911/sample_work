# Invoice OCR Pipeline — Streamlit UI + FastAPI

## 🚀 Quick Start

### Streamlit UI (Original)
```bash
pip install -r Requirements.txt

# Start Ollama
ollama serve
ollama pull llama3

# Launch UI
streamlit run app.py
```

### FastAPI (New)
```bash
pip install -r Requirements.txt
uvicorn app:app --reload  # http://localhost:8000/docs
```

---

## 📋 API Endpoints (Swagger UI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-ocr/` | Upload PDF/image → OCR text |
| `POST` | `/extract-fields/` | OCR text → Extract fields (JSON) |
| `POST` | `/qa/` | Ask questions about invoice |
| `GET`  | `/fields` | List all extractable fields |

**Swagger:** http://localhost:8000/docs  
**Postman:** Use Swagger "Try it out" → import cURL

---

## OCR Engines

| Engine    | Install | Notes |
|-----------|---------|-------|
| **EasyOCR**  | `pip install easyocr` | Default. CPU-friendly. |
| **DocTR**    | `pip install python-doctr[torch]` | Higher accuracy. More RAM. |
| **GLM-OCR**  | `pip install zhipuai` (cloud) or `transformers torch` (local) | Vision LLM OCR. Best for complex layouts. |

### GLM-OCR Setup
**Cloud (ZhipuAI API):**
```bash
pip install zhipuai
export ZHIPUAI_API_KEY=your_key_here
# or enter key in the UI sidebar
```

**Local (no internet after download):**
```bash
pip install transformers accelerate torch pillow
# Model THUDM/glm-4v-9b downloads automatically (~20GB)
# Requires GPU with 20GB+ VRAM, or CPU with 32GB+ RAM
```

---

## LLM — Llama Models Only

Only llama-family models are supported for extraction and Q&A:

```bash
ollama pull llama3          # 4.7 GB — best quality
ollama pull llama3.1        # 4.7 GB — latest
ollama pull llama3.2        # 2.0 GB — smaller
ollama pull llama3.2:1b     # 1.3 GB — fastest
ollama pull llama3.1:70b    # 40 GB  — highest quality (needs GPU)
ollama pull llama2          # 3.8 GB — older generation
```

---

## Field Manager (UI)

The Field Manager lets you toggle which of the 26 fields to extract:

- Click a field pill to toggle it on/off
- Use **Select All** / **Clear All** / **Reset** buttons
- Only selected fields are sent to the LLM → faster, cheaper, more focused

**Categories:**
| Category | Fields |
|---|---|
| Invoice | Number, Date, Due Date, Type, Reference, PO, Payment Date |
| Amounts | Total, Net, Tax, Tax Rate, Tax Base, Tax Amount |
| Supplier | Name, Phone, Email, Website, IBAN, SWIFT, Account, Routing, Address |
| Customer | Name, ID, Address, Shipping, Billing, Company Reg |
| Items | Line Items (Desc, Qty, Unit Price, Total, Code, Tax, Unit) |
| Meta | Language, Country, Currency |

---

## CLI Usage

```bash
# OCR only
python main.py invoice.pdf --no-extract

# Full pipeline (EasyOCR + llama3)
python main.py invoice.pdf

# GLM-OCR cloud
python main.py invoice.pdf --ocr glm --glm-key YOUR_KEY

# GLM-OCR local
python main.py invoice.pdf --ocr glm --glm-local

# Specific llama model
python main.py invoice.pdf --model llama3.2

# Specific fields only
python main.py invoice.pdf --fields invoice_number date total_amount supplier_name

# Ask questions
python main.py invoice.pdf -q "What is total?" -q "Who is supplier?"

# List available llama models
python main.py dummy --list-models

# List all extractable fields
python main.py dummy --list-fields
```