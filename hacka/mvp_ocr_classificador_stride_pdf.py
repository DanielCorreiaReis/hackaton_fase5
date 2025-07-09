"""
================================================================================
RELATÓRIO STRIDE AUTOMÁTICO – formatação melhorada (negrito + Unicode)
================================================================================
"""

import os
import time
import json
import unicodedata
from difflib import SequenceMatcher

import joblib
import requests
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth

from azure.identity import EnvironmentCredential
from azure.keyvault.secrets import SecretClient

# ──────────────────────────────────────────────────────────────────────────────
# 1) REGISTRA A MELHOR FONTE DISPONÍVEL
# ──────────────────────────────────────────────────────────────────────────────
CANDIDATAS = [
    ("DejaVuSans",       "./DejaVuSans.ttf"),                     # dir. atual
    ("ArialUnicodeMS",   r"C:\Windows\Fonts\ARIALUNI.TTF"),       # Windows
    ("SegoeUI",          r"C:\Windows\Fonts\segoeui.ttf"),        # Windows
]

FONT_NAME = None
BOLD_FONT_NAME = None
picked_path = None

for nome, caminho in CANDIDATAS:
    if os.path.isfile(caminho):
        pdfmetrics.registerFont(TTFont(nome, caminho))
        FONT_NAME = nome
        picked_path = caminho
        break

# tenta registrar variante bold se a fonte escolhida for DejaVuSans
if FONT_NAME == "DejaVuSans":
    bold_path = picked_path.replace(".ttf", "-Bold.ttf")
    if os.path.isfile(bold_path):
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold_path))
        BOLD_FONT_NAME = "DejaVuSans-Bold"

# Arial Unicode e Segoe UI já vêm como Regular; usa Helvetica-Bold como bold
if FONT_NAME in {"ArialUnicodeMS", "SegoeUI"}:
    BOLD_FONT_NAME = "Helvetica-Bold"

# fallback total: Helvetica + sanitização
if FONT_NAME is None:
    FONT_NAME = "Helvetica"
    BOLD_FONT_NAME = "Helvetica-Bold"
    print("Fonte Unicode não encontrada; usando Helvetica (+ sanitização).")
else:
    print(f"Fonte registrada: {FONT_NAME}")

# ──────────────────────────────────────────────────────────────────────────────
# 2) SANITIZAÇÃO (apenas se Helvetica)
# ──────────────────────────────────────────────────────────────────────────────
def safe(text: str) -> str:
    if FONT_NAME != "Helvetica":
        return text
    BAD = dict.fromkeys(range(0x2010, 0x2015), "-")
    BAD[0x2212] = "-"
    return unicodedata.normalize("NFKC", text).translate(BAD)

# ──────────────────────────────────────────────────────────────────────────────
# 3) CONFIGURAÇÕES GERAIS
# ──────────────────────────────────────────────────────────────────────────────

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
key_vault_name = os.getenv('KEY_VAULT_NAME')
kv_url = f"https://{key_vault_name}.vault.azure.net/"

# Autenticação com Azure usando EnvironmentCredential
credential = EnvironmentCredential()
client = SecretClient(vault_url=kv_url, credential=credential)

# Recuperar um segredo do Key Vault
secret_key_name = "KEY"
retrieved_secret_key = client.get_secret(secret_key_name)
AZURE_OCR_KEY = retrieved_secret_key.value

secret_region_name = "ENDPOINT"
retrieved_secret_endpoint = client.get_secret(secret_region_name)
AZURE_OCR_ENDPOINT = retrieved_secret_endpoint.value

IMAGE_PATH  = "azure.png"
MODEL_PATH  = "modelo_treinado.pkl"
VECT_PATH   = "vectorizer.pkl"
STRIDE_JSON = "modelo_stride.json"

FONT_SIZE    = 12
LEADING      = 16
MARGIN_X     = 40
MARGIN_Y     = 40
THRESHOLD    = 0.60
USE_JSON_FALLBACK = False

# ──────────────────────────────────────────────────────────────────────────────
# 4) OCR
# ──────────────────────────────────────────────────────────────────────────────
def extrair_texto_ocr(img_path: str) -> list[str]:
    url = f"{AZURE_OCR_ENDPOINT}vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_OCR_KEY,
        "Content-Type": "application/octet-stream"
    }
    with open(img_path, "rb") as arq:
        resp = requests.post(url, headers=headers, data=arq)
    resp.raise_for_status()
    op_url = resp.headers["Operation-Location"]

    while True:
        res = requests.get(op_url, headers={"Ocp-Apim-Subscription-Key": AZURE_OCR_KEY}).json()
        if res["status"] in {"succeeded", "failed"}:
            break
        time.sleep(1)

    words = []
    for page in res["analyzeResult"]["readResults"]:
        for line in page["lines"]:
            words.append(
                line["text"].lower().strip().replace("-", "_").replace(" ", "_")
            )
    return words

# ──────────────────────────────────────────────────────────────────────────────
# 5) CLASSIFICAÇÃO
# ──────────────────────────────────────────────────────────────────────────────
def prever_tipo_arquitetura(tokens: list[str]) -> str:
    model = joblib.load(MODEL_PATH)
    vect  = joblib.load(VECT_PATH)
    return model.predict(vect.transform([" ".join(tokens)]))[0]

# ──────────────────────────────────────────────────────────────────────────────
# 6) MATCHING INTELIGENTE
# ──────────────────────────────────────────────────────────────────────────────
def _norm(txt: str) -> str:
    return txt.lower().replace("-", "_").replace(" ", "_")

def _score(a: str, b: str) -> float:
    if a in b or b in a:
        return 1.0
    tok_a, tok_b = set(a.split("_")), set(b.split("_"))
    jaccard = len(tok_a & tok_b) / max(len(tok_a | tok_b), 1)
    seq     = SequenceMatcher(None, a, b).ratio()
    return max(jaccard, seq)

def carregar_stride_filtrado(tipo: str, tokens: list[str]) -> dict:
    with open(STRIDE_JSON, encoding="utf-8") as f:
        stride_all = json.load(f).get(tipo, {})

    tokens_norm = [_norm(t) for t in tokens]
    comps_json  = stride_all.get("componentes", {})

    filtrados = {
        nome: dados
        for nome, dados in comps_json.items()
        if max(_score(_norm(nome), t) for t in tokens_norm) >= THRESHOLD
    }
    return {
        "descricao_full": stride_all.get("descricao"),
        "componentes": filtrados
    }

# ──────────────────────────────────────────────────────────────────────────────
# 7) UTIL
# ──────────────────────────────────────────────────────────────────────────────
def humanize(comp: str) -> str:
    return comp.replace("_", " ").title()

def gerar_descricao(tipo: str, comps: list[str], fallback: str | None) -> str:
    if comps:
        leg = ", ".join(map(humanize, comps))
        return f"Arquitetura \"{tipo}\" com os seguintes componentes identificados no diagrama: {leg}."
    if USE_JSON_FALLBACK and fallback:
        return fallback
    return f"Arquitetura \"{tipo}\" (nenhum componente reconhecido no diagrama)."

# ──────────────────────────────────────────────────────────────────────────────
# 8) WRAP
# ──────────────────────────────────────────────────────────────────────────────
def _wrap(text, max_width, font_name):
    text = safe(text)
    words, cur, lines = text.split(), "", []
    for w in words:
        test = f"{cur} {w}".strip()
        if stringWidth(test, font_name, FONT_SIZE) <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# ──────────────────────────────────────────────────────────────────────────────
# 9) PDF
# ──────────────────────────────────────────────────────────────────────────────
def gerar_pdf(tipo: str, stride: dict, saida="relatorio_stride.pdf"):
    c = canvas.Canvas(saida, pagesize=A4)
    W, H = A4
    maxw = W - 2*MARGIN_X
    y = H - MARGIN_Y

    comps = list(stride.get("componentes", {}).keys())
    desc  = gerar_descricao(tipo, comps, stride.get("descricao_full"))

    # --- título (bold) ---
    c.setFont(BOLD_FONT_NAME, FONT_SIZE)
    for ln in _wrap(f"📘 Relatório STRIDE – Arquitetura: {tipo}", maxw, BOLD_FONT_NAME):
        c.drawString(MARGIN_X, y, ln); y -= LEADING

    # --- descrição (regular) ---
    c.setFont(FONT_NAME, FONT_SIZE)
    y -= 6
    for ln in _wrap(f"Descrição: {desc}", maxw, FONT_NAME):
        c.drawString(MARGIN_X, y, ln); y -= LEADING
    y -= LEADING

    if not comps:
        c.drawString(MARGIN_X, y, "Nenhum componente mapeado no JSON para este diagrama.")
        c.save(); print(f"PDF gerado sem itens STRIDE ({saida})."); return

    # --- loop componentes ---
    for comp, dados in stride["componentes"].items():
        if y < MARGIN_Y + 4*LEADING:
            c.showPage(); y = H - MARGIN_Y
        # componente em bold
        c.setFont(BOLD_FONT_NAME, FONT_SIZE)
        for ln in _wrap(f"🔧 Componente: {humanize(comp)}", maxw, BOLD_FONT_NAME):
            c.drawString(MARGIN_X, y, ln); y -= LEADING
        c.setFont(FONT_NAME, FONT_SIZE)

        # ameaças / contramedidas (bullet em bold):
        for ame, desc_a in dados.get("ameacas", {}).items():
            contra = dados.get("contramedidas", {}).get(ame, "N/A")
            linha1 = f"• Ameaça ({ame}): {desc_a}"
            linha2 = f"Contramedida: {contra}"

            # — primeira linha com bullet em bold —
            bullet, resto = "•", linha1[1:]  # remove primeiro char
            c.setFont(BOLD_FONT_NAME, FONT_SIZE)
            c.drawString(MARGIN_X + 20, y, bullet)
            x_offset = stringWidth(bullet + " ", BOLD_FONT_NAME, FONT_SIZE)
            c.setFont(FONT_NAME, FONT_SIZE)
            for ln in _wrap(resto.strip(), maxw - x_offset - 20, FONT_NAME):
                c.drawString(MARGIN_X + 20 + x_offset, y, ln); y -= LEADING

            # — segunda linha (contramedida) —
            for ln in _wrap(linha2, maxw - 40, FONT_NAME):
                c.drawString(MARGIN_X + 40, y, ln); y -= LEADING
            y -= 4
        y -= LEADING / 2

    c.save()
    print(f"✅ Relatório salvo: {saida}")

# ──────────────────────────────────────────────────────────────────────────────
# 10) PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def executar_pipeline():
    print("Extraindo texto do diagrama...")
    tokens = extrair_texto_ocr(IMAGE_PATH)
    print("   → Componentes detectados:", tokens)

    print("Classificando arquitetura...")
    tipo = prever_tipo_arquitetura(tokens)
    print("   → Arquitetura prevista:", tipo)

    print("Filtrando STRIDE...")
    stride = carregar_stride_filtrado(tipo, tokens)

    print("Gerando PDF...")
    gerar_pdf(tipo, stride)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    executar_pipeline()
