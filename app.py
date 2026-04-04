"""
Invoice OCR Pipeline -- Streamlit UI (Light Theme)
* OCR: DocTR (via PyMuPDF for PDF conversion)
* LLM: Ollama (llama models) -- 100% local
* Interactive field manager
Run: streamlit run app.py
"""

import streamlit as st
import json, os, re, tempfile
from pathlib import Path
from collections import defaultdict

from main import (
    convert_pdf_to_images,
    run_ocr,
    extract_fields_with_llm,
    answer_question,
    ALL_FIELDS_REGISTRY,
    DEFAULT_FIELDS,
)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Invoice OCR Extractor", page_icon="🧾", layout="wide")

# ─────────────────────────────────────────────────────────────
# CSS  — light theme + field-pill styles
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #f8fafc !important;
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] * { color: #334155 !important; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
    padding: 22px 28px; border-radius: 12px; margin-bottom: 20px;
}
.app-header h1 { color: white !important; font-size: 1.6rem; font-weight: 700; margin: 0; }
.app-header p  { color: #bfdbfe !important; margin: 3px 0 0; font-size: 0.88rem; }

/* Metric card */
.metric-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-label { font-size: 0.7rem; font-weight: 600; letter-spacing:.08em;
                text-transform:uppercase; color:#94a3b8; margin-bottom:5px; }
.metric-value { font-size: 1.2rem; font-weight: 700; color: #0f172a; }
.metric-value.nv { color: #cbd5e1; font-weight:400; font-size:.95rem; }

/* Section card */
.section-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.section-title {
    font-size: 0.75rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
    color:#64748b; border-bottom:1px solid #f1f5f9; padding-bottom:8px; margin-bottom:12px;
}
.field-row { display:flex; justify-content:space-between; align-items:flex-start;
             padding:5px 0; border-bottom:1px solid #f8fafc; }
.field-row:last-child { border-bottom:none; }
.fkey { font-size:.81rem; color:#64748b; font-weight:500; min-width:140px; }
.fval { font-size:.87rem; color:#1e293b; font-weight:500; text-align:right; word-break:break-all; }
.fval.nv { color:#cbd5e1; font-weight:400; }

/* Totals */
.totals-row { display:flex; gap:14px; margin:10px 0; }
.total-chip { flex:1; background:#f0f9ff; border:1px solid #bae6fd;
              border-radius:8px; padding:12px 14px; text-align:center; }
.total-chip .lbl { font-size:.68rem; color:#0284c7; font-weight:600;
                   text-transform:uppercase; letter-spacing:.08em; }
.total-chip .val { font-size:1.25rem; font-weight:700; color:#0c4a6e; margin-top:2px; }
.total-chip .val.nv { color:#94a3b8; font-size:.95rem; font-weight:400; }

/* -- Field Manager -- */
.fm-wrap { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:18px; }
.fm-category-label {
    font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:#94a3b8; margin:14px 0 6px 2px;
}
.fm-category-label:first-child { margin-top:0; }

/* active pill */
.pill-on {
    display:inline-flex; align-items:center; gap:5px;
    background:#eff6ff; border:1.5px solid #3b82f6; color:#1d4ed8;
    font-size:.76rem; font-weight:600; padding:5px 12px; border-radius:20px;
    margin:3px; cursor:pointer; transition:all .15s; user-select:none;
}
.pill-on:hover { background:#dbeafe; border-color:#2563eb; }

/* inactive pill */
.pill-off {
    display:inline-flex; align-items:center; gap:5px;
    background:#f8fafc; border:1.5px dashed #cbd5e1; color:#94a3b8;
    font-size:.76rem; font-weight:500; padding:5px 12px; border-radius:20px;
    margin:3px; cursor:pointer; transition:all .15s; user-select:none;
}
.pill-off:hover { background:#f1f5f9; border-color:#94a3b8; color:#475569; }

/* Tabs */
button[data-baseweb="tab"] { font-family:'Inter',sans-serif !important;
    color:#64748b !important; font-size:.87rem !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color:#2563eb !important; border-bottom-color:#2563eb !important; }

/* Buttons */
.stButton > button {
    background:#2563eb !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:.84rem !important;
}
.stButton > button:hover { background:#1d4ed8 !important; }
.stDownloadButton > button {
    background:#f1f5f9 !important; color:#334155 !important;
    border:1px solid #e2e8f0 !important; border-radius:8px !important;
}

textarea { font-family:'JetBrains Mono',monospace !important; font-size:.79rem !important;
           background:#f8fafc !important; border:1px solid #e2e8f0 !important; color:#334155 !important; }
[data-testid="stDataFrame"] { border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:8px !important; }
.divider { border:none; border-top:1px solid #e2e8f0; margin:18px 0; }
.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:.7rem; font-weight:600; }
.badge-blue  { background:#dbeafe; color:#1d4ed8; }
.badge-green { background:#dcfce7; color:#15803d; }
.badge-gray  { background:#f1f5f9; color:#64748b; }
.badge-purple{ background:#ede9fe; color:#6d28d9; }

/* Ollama status */
.ollama-ok  { background:#dcfce7; color:#15803d; border:1px solid #86efac;
              border-radius:8px; padding:6px 12px; font-size:.78rem; font-weight:600; }
.ollama-off { background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5;
              border-radius:8px; padding:6px 12px; font-size:.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _v(val, fallback="None"):
    if val is None or val == "" or val == [] or val == {}:
        return fallback
    return str(val)

def _metric(label, value):
    v = _v(value)
    c = "nv" if v == "None" else ""
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {c}">{v}</div></div>'

def _field(key, val):
    v = _v(val)
    c = "nv" if v == "None" else ""
    return f'<div class="field-row"><span class="fkey">{key}</span><span class="fval {c}">{v}</span></div>'

def _addr(d):
    if not d or not isinstance(d, dict):
        return "None"
    parts = [d.get(k) for k in ["street_no","street_name","address","city","state","postal_code","country"] if d.get(k)]
    return ", ".join(parts) if parts else "None"

def _check_ollama(host="http://localhost:11434"):
    try:
        import requests
        return requests.get(f"{host}/api/tags", timeout=2).status_code == 200
    except:
        return False

def _list_llama_models(host="http://localhost:11434"):
    try:
        import requests
        r = requests.get(f"{host}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", []) if "llama" in m["name"].lower()]
    except:
        return []


# ─────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────
for k, v in {
    "selected_fields": list(DEFAULT_FIELDS),
    "ocr_text": "", "extracted": {}, "qa_history": [],
    "ocr_done": False, "tmp_path": None, "tmp_suffix": "", "tmp_dir": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# Sidebar — Ollama + DPI only (DocTR is the fixed OCR engine)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    # OCR engine — fixed to DocTR
    st.markdown("**OCR Engine**")
    st.markdown(
        '<div style="background:#eff6ff;border:1.5px solid #3b82f6;border-radius:8px;'
        'padding:8px 14px;font-size:.84rem;font-weight:600;color:#1d4ed8;">🔬 DocTR (PyMuPDF)</div>',
        unsafe_allow_html=True
    )
    st.caption("High-accuracy deep-learning OCR. PDF pages are rendered with PyMuPDF.")

    st.markdown("---")

    # Ollama config
    st.markdown("**🦙 Ollama — Llama Models**")
    ollama_host = st.text_input("Host", "http://localhost:11434", label_visibility="collapsed")

    ollama_ok = _check_ollama(ollama_host)
    if ollama_ok:
        st.markdown('<div class="ollama-ok">● Ollama running</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ollama-off">● Ollama offline</div>', unsafe_allow_html=True)

    live_models = _list_llama_models(ollama_host)
    ALL_LLAMA = ["llama3","llama3.1","llama3.2","llama3.2:1b","llama3.1:8b","llama3:8b","llama2"]
    model_opts = live_models if live_models else ALL_LLAMA
    llm_model = st.selectbox("Llama Model", model_opts, label_visibility="collapsed")

    if not live_models:
        st.caption("Pull a model first:")
        st.code("ollama serve\nollama pull llama3", language="bash")

    st.markdown("---")
    do_extract = st.checkbox("Extract structured fields", value=True)
    dpi = st.slider("PDF render DPI", 100, 400, 200, 25,
                    help="Higher DPI = better OCR quality but more RAM and slower rendering.")


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🧾 Invoice OCR Extractor</h1>
  <p>Upload invoice → DocTR OCR (PyMuPDF) → Configure fields → Extract → Ask questions &nbsp;·&nbsp; 100% local</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Two-column top: Upload  |  Field Manager
# ─────────────────────────────────────────────────────────────
col_up, col_fm = st.columns([1, 1], gap="large")

# ── Upload ────────────────────────────────────────────────────
with col_up:
    st.markdown("#### 📤 Upload Invoice")
    uploaded = st.file_uploader(
        "Drop PDF or image",
        type=["pdf","png","jpg","jpeg","tiff","bmp","webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        td = tempfile.mkdtemp()
        tp = os.path.join(td, uploaded.name)
        with open(tp, "wb") as f:
            f.write(uploaded.read())
        st.session_state.update({"tmp_path": tp, "tmp_suffix": suffix, "tmp_dir": td})

        if suffix in {".png",".jpg",".jpeg",".webp",".bmp"}:
            st.image(tp, use_container_width=True)
        else:
            st.success(f"📄 {uploaded.name}")

    b1, b2 = st.columns(2)
    ocr_btn  = b1.button("🔍 OCR Only",    use_container_width=True, disabled=not uploaded)
    full_btn = b2.button("⚡ Full Pipeline", use_container_width=True, type="primary", disabled=not uploaded)

    if st.session_state["ocr_done"]:
        n = len(st.session_state["ocr_text"])
        st.success(f"✅ OCR · {n:,} chars")
    if st.session_state.get("extracted") and not st.session_state["extracted"].get("_parse_error"):
        nf = len([v for v in st.session_state["extracted"].values() if v not in (None, [], {})])
        st.success(f"✅ {nf} fields extracted")


# ── Field Manager ─────────────────────────────────────────────
with col_fm:
    st.markdown("#### 🎛 Field Manager")
    st.caption("Click a field to **add** or **remove** it from extraction.")

    qa1, qa2, qa3 = st.columns(3)
    if qa1.button("✅ All",    use_container_width=True):
        st.session_state["selected_fields"] = [
            fk for cat_fields in ALL_FIELDS_REGISTRY.values() for fk,_,_,_ in cat_fields
        ]
        st.rerun()
    if qa2.button("❌ Clear",  use_container_width=True):
        st.session_state["selected_fields"] = []
        st.rerun()
    if qa3.button("↩ Reset",   use_container_width=True):
        st.session_state["selected_fields"] = list(DEFAULT_FIELDS)
        st.rerun()

    selected_set = set(st.session_state["selected_fields"])
    total_fields = sum(len(v) for v in ALL_FIELDS_REGISTRY.values())
    st.markdown(
        f'<div style="font-size:.8rem; color:#2563eb; font-weight:600; margin-bottom:8px;">'
        f'{len(selected_set)} / {total_fields} fields selected</div>',
        unsafe_allow_html=True
    )

    cat_icons = {
        "Invoice": "🧾", "Amounts": "💰", "Supplier": "🏢",
        "Customer": "👤", "Line Items": "📋", "Locale": "🌍"
    }

    for cat, fields in ALL_FIELDS_REGISTRY.items():
        icon = cat_icons.get(cat, "•")
        st.markdown(f'<div class="fm-category-label">{icon} {cat}</div>', unsafe_allow_html=True)

        pills_html = ""
        for fk, label, _, _ in fields:
            if fk in selected_set:
                pills_html += f'<span class="pill-on">✓ {label}</span>'
            else:
                pills_html += f'<span class="pill-off">+ {label}</span>'
        st.markdown(f'<div style="line-height:2.2">{pills_html}</div>', unsafe_allow_html=True)

        with st.expander(f"Toggle {cat}", expanded=False):
            for fk, label, _, _ in fields:
                cur = fk in selected_set
                new = st.checkbox(label, value=cur, key=f"ck_{fk}")
                if new != cur:
                    if new:
                        st.session_state["selected_fields"].append(fk)
                    else:
                        st.session_state["selected_fields"].remove(fk)
                    st.rerun()


# ─────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if (ocr_btn or full_btn) and st.session_state.get("tmp_path"):
    tp     = st.session_state["tmp_path"]
    suffix = st.session_state["tmp_suffix"]
    td     = st.session_state["tmp_dir"]

    with st.spinner("Running DocTR OCR…"):
        try:
            pages_dir = os.path.join(td, "pages")
            if suffix == ".pdf":
                image_paths = convert_pdf_to_images(tp, pages_dir, dpi=dpi)
            else:
                image_paths = [tp]

            ocr_text = run_ocr(image_paths)
            st.session_state.update({
                "ocr_text": ocr_text, "ocr_done": True,
                "extracted": {}, "qa_history": []
            })
        except Exception as e:
            st.error(f"**OCR Error:** {e}")
            st.stop()

    if full_btn and do_extract and st.session_state["selected_fields"]:
        if not _check_ollama(ollama_host):
            st.warning("⚠️ Ollama offline — start with `ollama serve`")
        else:
            with st.spinner(f"Extracting {len(st.session_state['selected_fields'])} fields with **{llm_model}**…"):
                try:
                    result = extract_fields_with_llm(
                        st.session_state["ocr_text"],
                        provider="ollama",
                        model=llm_model,
                        selected_keys=st.session_state["selected_fields"],
                    )
                    st.session_state["extracted"] = result
                except Exception as e:
                    st.error(f"**LLM Error:** {e}")
    st.rerun()


# ─────────────────────────────────────────────────────────────
# Results tabs
# ─────────────────────────────────────────────────────────────
if st.session_state["ocr_done"]:
    tab_ocr, tab_fields, tab_qa = st.tabs(["📄 OCR Text", "🗂 Extracted Fields", "💬 Q&A"])

    # ── Tab 1: OCR ────────────────────────────────────────────
    with tab_ocr:
        st.text_area("", st.session_state["ocr_text"], height=420,
                     label_visibility="collapsed")
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Download OCR Text", st.session_state["ocr_text"],
                           file_name="ocr_output.txt", use_container_width=True)

        if not st.session_state.get("extracted") and do_extract:
            if c2.button("🤖 Extract Fields Now", use_container_width=True):
                if not _check_ollama(ollama_host):
                    st.error("Start Ollama: `ollama serve`")
                else:
                    with st.spinner("Extracting…"):
                        try:
                            st.session_state["extracted"] = extract_fields_with_llm(
                                st.session_state["ocr_text"],
                                provider="ollama", model=llm_model,
                                selected_keys=st.session_state["selected_fields"],
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

    # ── Tab 2: Extracted Fields ───────────────────────────────
    with tab_fields:
        data = st.session_state.get("extracted", {})

        if not data:
            st.info("Run **Full Pipeline** or click **Extract Fields Now** in the OCR tab.")

        elif data.get("_parse_error"):
            raw = data.get("_raw_response", "")
            st.warning("⚠️ LLM returned extra text around the JSON. Attempting auto-recovery…")
            st.text_area("Raw LLM response", raw, height=280)
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    recovered = json.loads(m.group(0))
                    st.success("✅ Recovered JSON — displaying below:")
                    data = recovered
                    st.session_state["extracted"] = recovered
                except Exception:
                    st.error("Could not recover JSON. Try re-running or switching model.")
                    st.stop()

        if data and not data.get("_parse_error"):
            # ── Top metrics ───────────────────────────────────
            top_keys = [
                ("invoice_number","Invoice #"), ("date","Date"),
                ("total_amount","Total"),       ("due_date","Due Date"),
            ]
            active_top = [(k,l) for k,l in top_keys if k in st.session_state["selected_fields"]]
            if active_top:
                cols = st.columns(len(active_top))
                for i,(k,l) in enumerate(active_top):
                    cols[i].markdown(_metric(l, data.get(k)), unsafe_allow_html=True)

            # ── Totals chips ──────────────────────────────────
            sel = set(st.session_state["selected_fields"])
            if any(k in sel for k in ("total_amount","total_net","total_tax")):
                chips = []
                for k, lbl in [("total_amount","Total Amount"),("total_net","Total Net"),("total_tax","Total Tax")]:
                    if k in sel:
                        v = _v(data.get(k)); nv = "nv" if v=="None" else ""
                        chips.append(f'<div class="total-chip"><div class="lbl">{lbl}</div><div class="val {nv}">{v}</div></div>')
                st.markdown(f'<div class="totals-row">{"".join(chips)}</div>', unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ── Supplier + Customer cards ─────────────────────
            sup_fields = {"supplier_name","supplier_phone","supplier_email","supplier_website",
                          "sup_reg_type","sup_reg_phone","supplier_address",
                          "sup_iban","sup_swift","sup_account","sup_routing"}
            cust_fields = {"customer_name","customer_id","customer_address",
                           "shipping_address","billing_address","cust_reg_type","cust_reg_phone"}
            inv_fields  = {"document_type","reference_number","purchase_order","payment_date",
                           "tax_rate","tax_base","tax_amount_val","locale_language","locale_country","locale_currency"}

            has_sup  = bool(sel & sup_fields)
            has_cust = bool(sel & cust_fields)
            has_inv  = bool(sel & inv_fields)

            sup_reg  = data.get("supplier_company_registration") or {}
            sup_pay  = data.get("supplier_payment_details") or {}
            sup_addr = data.get("supplier_address") or {}
            cust_addr= data.get("customer_address") or {}
            cust_reg = data.get("customer_company_registration") or {}
            taxes    = data.get("taxes") or {}
            locale   = data.get("locale") or {}

            if has_sup or has_cust:
                sc1, sc2 = st.columns(2)

                if has_sup:
                    with sc1:
                        rows = ""
                        if "supplier_name"    in sel: rows += _field("Name",         data.get("supplier_name"))
                        if "supplier_phone"   in sel: rows += _field("Phone",        data.get("supplier_phone"))
                        if "supplier_email"   in sel: rows += _field("Email",        data.get("supplier_email"))
                        if "supplier_website" in sel: rows += _field("Website",      data.get("supplier_website"))
                        if "sup_reg_type"     in sel: rows += _field("Co. Type",     sup_reg.get("type"))
                        if "sup_reg_phone"    in sel: rows += _field("Co. Phone",    sup_reg.get("phone_no"))
                        if "supplier_address" in sel: rows += _field("Address",      _addr(sup_addr))
                        pay_rows = ""
                        if "sup_iban"    in sel: pay_rows += _field("IBAN",        sup_pay.get("iban"))
                        if "sup_swift"   in sel: pay_rows += _field("SWIFT / BIC", sup_pay.get("swift"))
                        if "sup_account" in sel: pay_rows += _field("Account No.", sup_pay.get("account_number"))
                        if "sup_routing" in sel: pay_rows += _field("Routing No.", sup_pay.get("routing_number"))
                        pay_block = ""
                        if pay_rows:
                            pay_block = (f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #f1f5f9;">'
                                         f'<div class="fkey" style="margin-bottom:6px">Payment Details</div>{pay_rows}</div>')
                        st.markdown(f'<div class="section-card"><div class="section-title">🏢 Supplier</div>{rows}{pay_block}</div>',
                                    unsafe_allow_html=True)

                if has_cust:
                    with sc2:
                        rows = ""
                        if "customer_name"    in sel: rows += _field("Name",          data.get("customer_name"))
                        if "customer_id"      in sel: rows += _field("ID",            data.get("customer_id"))
                        if "cust_reg_type"    in sel: rows += _field("Co. Type",      cust_reg.get("type"))
                        if "cust_reg_phone"   in sel: rows += _field("Co. Phone",     cust_reg.get("phone_no"))
                        if "customer_address" in sel: rows += _field("Address",       _addr(cust_addr))
                        if "shipping_address" in sel: rows += _field("Shipping",      data.get("shipping_address"))
                        if "billing_address"  in sel: rows += _field("Billing",       data.get("billing_address"))
                        st.markdown(f'<div class="section-card"><div class="section-title">👤 Customer</div>{rows}</div>',
                                    unsafe_allow_html=True)

            if has_inv:
                rows = ""
                if "document_type"   in sel: rows += _field("Document Type",  data.get("document_type"))
                if "reference_number"in sel: rows += _field("Reference No.",  data.get("reference_number"))
                if "purchase_order"  in sel: rows += _field("Purchase Order", data.get("purchase_order"))
                if "payment_date"    in sel: rows += _field("Payment Date",   data.get("payment_date"))
                if "tax_rate"        in sel: rows += _field("Tax Rate",       taxes.get("rate"))
                if "tax_base"        in sel: rows += _field("Tax Base",       taxes.get("base"))
                if "tax_amount_val"  in sel: rows += _field("Tax Amount",     taxes.get("amount"))
                if "locale_language" in sel: rows += _field("Language",       locale.get("language"))
                if "locale_country"  in sel: rows += _field("Country",        locale.get("country"))
                if "locale_currency" in sel: rows += _field("Currency",       locale.get("currency"))
                if rows:
                    st.markdown(f'<div class="section-card"><div class="section-title">📋 Invoice Details</div>{rows}</div>',
                                unsafe_allow_html=True)

            # ── Line items ────────────────────────────────────
            if "line_items" in sel:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 📋 Line Items")
                items = data.get("line_items") or []
                if items and isinstance(items, list) and len(items) > 0 and isinstance(items[0], dict):
                    import pandas as pd
                    df = pd.DataFrame(items).fillna("None")
                    df = df.rename(columns={
                        "description":"Description","quantity":"Qty",
                        "unit_price":"Unit Price","total_price":"Total Price",
                        "product_code":"Product Code","tax_amount":"Tax Amt",
                        "tax_rate":"Tax Rate","unit_measure":"Unit"
                    })
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.markdown('<span style="color:#94a3b8;font-size:.88rem">No line items found — None</span>',
                                unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇ Download JSON",
                               json.dumps(data, indent=2, ensure_ascii=False),
                               file_name="invoice_extracted.json",
                               mime="application/json", use_container_width=True)
            with d2.expander("🔍 Raw JSON"):
                st.json(data)

    # ── Tab 3: Q&A ────────────────────────────────────────────
    with tab_qa:
        st.markdown("#### 💬 Ask about this invoice")
        if not _check_ollama(ollama_host):
            st.error("Ollama is offline. Start: `ollama serve`")
        else:
            with st.form("qa_form", clear_on_submit=True):
                qc, bc = st.columns([5, 1])
                question = qc.text_input("", placeholder="What is the total tax amount?",
                                         label_visibility="collapsed")
                asked = bc.form_submit_button("Ask →")
            if asked and question:
                with st.spinner("Thinking…"):
                    try:
                        ans = answer_question(
                            question,
                            st.session_state.get("extracted", {}),
                            st.session_state["ocr_text"],
                            provider="ollama", model=llm_model,
                        )
                        st.session_state["qa_history"].insert(0, {"q": question, "a": ans})
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        if not st.session_state.get("qa_history"):
            st.caption("Suggestions:")
            for sq in ["What is the total amount?","Who is the supplier?",
                       "List all line items","What is the tax rate applied?"]:
                st.markdown(f"→ *{sq}*")

        for item in st.session_state.get("qa_history", []):
            with st.chat_message("user"):    st.write(item["q"])
            with st.chat_message("assistant"): st.write(item["a"])

else:
    st.markdown("""
<div style="text-align:center;padding:50px 20px;">
  <div style="font-size:3rem">🧾</div>
  <div style="font-size:1.15rem;font-weight:600;color:#1e293b;margin-top:10px">Upload an invoice to get started</div>
  <div style="font-size:.88rem;color:#64748b;margin-top:5px">PDF · PNG · JPG · TIFF · BMP · WebP</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Install**\n```bash\npip install pymupdf python-doctr[torch] requests streamlit pandas\n```")
    with c2:
        st.markdown("**Start Ollama**\n```bash\nollama serve\nollama pull llama3\n```")
    with c3:
        st.markdown("**Run**\n```bash\nstreamlit run app.py\n```")
