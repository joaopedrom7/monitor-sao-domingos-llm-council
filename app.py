"""
Claude x GPT - Roteador Inteligente (IAedu)
================================================
Roteia cada pedido para o modelo mais forte (Claude Opus 4.7 vs GPT-5.5),
com modos Duelo e Colaboracao. Inclui:
  - Login com password (PBKDF2, stdlib)
  - Memoria "master" por utilizador (resumo persistente do perfil + temas passados)
  - Upload de PDF / PPTX / imagem (com OCR) + RAG com embeddings

Correr:
  pip install -r requirements.txt
  # OCR precisa do binario do sistema:  sudo apt-get install tesseract-ocr tesseract-ocr-por
  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import io
import json
import os
import queue
import re
import secrets
import threading
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Dependencias opcionais (extracao/OCR) - degradacao graciosa se faltarem
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

try:
    from pptx import Presentation
    HAS_PPTX = True
except Exception:
    HAS_PPTX = False

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

OCR_LANG = "eng"
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    langs = set(pytesseract.get_languages(config=""))
    OCR_LANG = "+".join([l for l in ("por", "eng") if l in langs]) or "eng"
    HAS_OCR = HAS_PIL
except Exception:
    HAS_OCR = False

# ---------------------------------------------------------------------------
# Dependencias opcionais para RAG com embeddings
# ---------------------------------------------------------------------------
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None
    HAS_NUMPY = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    CrossEncoder = None
    HAS_SENTENCE_TRANSFORMERS = False

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
USERS_FILE = APP_DIR / "users.json"
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")

PER_FILE_CAP = 20000
TOTAL_CAP = 40000
SESSION_TRANSCRIPT_CAP = 24000  # janela maxima da conversa atual enviada aos modelos
MEMORY_UPDATE_EVERY = 3   # turnos entre atualizacoes automaticas da memoria

# RAG com embeddings para anexos grandes/persistentes da sessao
ATTACHMENT_SEARCH_TRIGGER = 8000
ATTACHMENT_CHUNK_SIZE = 1800
ATTACHMENT_CHUNK_OVERLAP = 250
ATTACHMENT_CONTEXT_CAP = 36000
ATTACHMENT_TOP_K_DEFAULT = 6
ATTACHMENT_CANDIDATE_MULTIPLIER = 6
EMBEDDING_MODEL_OPTIONS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large-instruct",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]
RERANKER_MODEL_OPTIONS = [
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

# ---------------------------------------------------------------------------
# Configuracao dos dois agentes (do notebook IA_EDU.ipynb)
# ---------------------------------------------------------------------------
MODELS = {
    "claude": {
        "label": "Claude Opus 4.7", "short": "Opus 4.7", "color": "#D2A24C",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmoss7l0f658oko01vk2egfpg/stream",
        "channel_id": "cmpws0w210p11i601iwo0bxpe",
        "env": "IAEDU_CLAUDE_KEY", "price_in": 5.0, "price_out": 25.0,
    },
    "gpt": {
        "label": "GPT-5.5", "short": "GPT-5.5", "color": "#46B98C",
        "endpoint": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmor5objoex9gfp01vm7p95jh/stream",
        "channel_id": "cmpzy2p8j05wanr010emnsh2w",
        "env": "IAEDU_GPT_KEY", "price_in": 5.0, "price_out": 30.0,
    },
}

# ---------------------------------------------------------------------------
# Regras de roteamento (derivadas do comparativo Opus 4.7 vs GPT-5.5)
# ---------------------------------------------------------------------------
ROUTING_RULES = [
    {"id": "terminal", "model": "gpt", "label": "Agentes de terminal / CLI / DevOps",
     "bench": "Terminal-Bench 2.0 - 82.7% vs 69.4%",
     "keywords": ["terminal", "shell", "bash", "cli", "comando", "linha de comando", "devops",
                  "infra", "infraestrutura", "docker", "kubernetes", "ssh", "ci/cd",
                  "pipeline ci", "cron", "ansible", "makefile"]},
    {"id": "web", "model": "gpt", "label": "Pesquisa web / browsing / info atual",
     "bench": "BrowseComp - 90.1% (Pro) vs 79.3%",
     "keywords": ["pesquisa web", "pesquisar na web", "browse", "navegar", "internet",
                  "noticias", "notícias", "atual", "ultimo", "último", "recente", "cotacao",
                  "cotação", "preco atual", "preço atual", "search the web", "google", "hoje em dia"]},
    {"id": "longcontext", "model": "gpt", "label": "Contexto longo (>128K tokens)",
     "bench": "MRCR v2 512K-1M - 74.0% vs 32.2%",
     "keywords": ["contexto longo", "documento inteiro", "monorepo", "corpus", "ficheiro gigante",
                  "long context", "1m tokens", "ano de registos", "logs inteiros",
                  "analisa este documento longo", "milhares de paginas"]},
    {"id": "math", "model": "gpt", "label": "Matematica de fronteira / competicao",
     "bench": "FrontierMath Tier 4 - 35.4% vs 22.9%",
     "keywords": ["prova matematica", "prova matemática", "demonstra que", "teorema", "olimpiada",
                  "olimpíada", "competicao matematica", "integral", "equacao diferencial",
                  "equação diferencial", "algebra abstrata", "prove that", "calculo avancado", "lema"]},
    {"id": "abstract", "model": "gpt", "label": "Raciocinio abstrato / padroes",
     "bench": "ARC-AGI-2 - 85.0% vs 75.8%",
     "keywords": ["padrao", "padrão", "puzzle", "sequencia logica", "sequência lógica",
                  "raciocinio abstrato", "arc-agi", "quebra-cabeca", "adivinha a regra",
                  "qual o proximo", "qual o próximo"]},
    {"id": "office", "model": "gpt", "label": "Trabalho de escritorio / documentos",
     "bench": "OfficeQA Pro - 54.1% vs 43.6%  ·  GDPval 84.9% vs 80.3%",
     "keywords": ["powerpoint", "slides", "excel", "folha de calculo", "folha de cálculo",
                  "relatorio de escritorio", "memorando", "ata", "documento word"]},
    {"id": "pr_code", "model": "claude", "label": "Resolucao de PRs / refactor / codebase",
     "bench": "SWE-Bench Pro - 64.3% vs 58.6%",
     "keywords": ["bug", "corrige", "fix", "refactor", "refatora", "pull request", "code review",
                  "reve o codigo", "revê o código", "codebase", "repositorio", "repositório",
                  "stack trace", "erro no codigo", "erro no código", "debugar", "depurar",
                  "migracao de codigo", "traceback", "exception"]},
    {"id": "mcp", "model": "claude", "label": "Orquestracao de ferramentas / MCP / agentes",
     "bench": "MCP Atlas - 79.1% vs 75.3%",
     "keywords": ["mcp", "orquestra", "tool use", "agente multi-step", "chama a api", "encadeia",
                  "workflow de agentes", "function calling", "multiplas ferramentas",
                  "múltiplas ferramentas", "fastapi"]},
    {"id": "finance", "model": "claude", "label": "Analise financeira",
     "bench": "FinanceAgent v1.1 - 64.4% vs 60.0%",
     "keywords": ["financeiro", "financas", "finanças", "investimento", "balanco", "balanço",
                  "dcf", "valuation", "fluxo de caixa", "acoes", "ações", "portfolio", "portfólio",
                  "analise financeira", "demonstracoes financeiras"]},
    {"id": "academic", "model": "claude", "label": "Raciocinio academico profundo",
     "bench": "Humanity's Last Exam - 54.7% vs 52.2% (c/ ferramentas)",
     "keywords": ["tese", "paper", "artigo cientifico", "artigo científico", "investigacao",
                  "investigação", "academico", "académico", "prova conceptual",
                  "explica em profundidade", "raciocinio profundo", "dissertacao", "dissertação",
                  "estado da arte", "review critica"]},
]

DEFAULT_MODEL = "claude"
DEFAULT_REASON = ("Sem sinal forte de categoria &mdash; Claude por defeito "
                  "(saida ~17% mais barata e forte em raciocinio geral / codigo).")
LONG_INPUT_CHARS = 6000


# ---------------------------------------------------------------------------
# Roteador
# ---------------------------------------------------------------------------
def route(message: str, total_len: int | None = None) -> dict:
    text = message.lower()
    scores: dict[str, int] = {}
    for rule in ROUTING_RULES:
        hits = sum(1 for kw in rule["keywords"] if kw in text)
        if hits:
            scores[rule["id"]] = hits
    if (total_len or len(message)) > LONG_INPUT_CHARS:
        scores["longcontext"] = scores.get("longcontext", 0) + 2

    if not scores:
        return {"model": DEFAULT_MODEL, "rule_id": None, "label": "Geral",
                "bench": "-", "reason": DEFAULT_REASON, "scores": {}}

    best_id, best = None, -1
    for rule in ROUTING_RULES:
        s = scores.get(rule["id"], 0)
        if s > best:
            best, best_id = s, rule["id"]

    rule = next(r for r in ROUTING_RULES if r["id"] == best_id)
    reason = (f"Categoria detetada: <b>{rule['label']}</b> &rarr; "
              f"{MODELS[rule['model']]['label']} ({rule['bench']})")
    return {"model": rule["model"], "rule_id": rule["id"], "label": rule["label"],
            "bench": rule["bench"], "reason": reason, "scores": scores}


# ---------------------------------------------------------------------------
# Autenticacao (PBKDF2 + users.json)
# ---------------------------------------------------------------------------
def _load_users() -> dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), "utf-8")


def _hash_pw(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


def create_user(username: str, name: str, password: str):
    users = _load_users()
    salt = secrets.token_hex(16)
    users[username] = {"salt": salt, "hash": _hash_pw(password, salt),
                       "name": name, "created": dt.datetime.now().isoformat(timespec="seconds")}
    _save_users(users)


def verify_user(username: str, password: str) -> bool:
    u = _load_users().get(username)
    if not u:
        return False
    return hmac.compare_digest(u["hash"], _hash_pw(password, u["salt"]))


# ---------------------------------------------------------------------------
# Memoria do utilizador
# ---------------------------------------------------------------------------
def _user_dir(username: str) -> Path:
    d = DATA_DIR / username
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_memory(username: str) -> dict:
    f = _user_dir(username) / "memory.json"
    if f.exists():
        try:
            return json.loads(f.read_text("utf-8"))
        except Exception:
            pass
    return {"master": "", "updated_at": None, "turns_since": 0}


def save_memory(username: str, mem: dict):
    (_user_dir(username) / "memory.json").write_text(
        json.dumps(mem, indent=2, ensure_ascii=False), "utf-8")


def append_log(username: str, prompt: str, mode: str):
    line = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
            "mode": mode, "prompt": prompt[:1500]}
    with open(_user_dir(username) / "log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def recent_log(username: str, n: int = 12) -> list[str]:
    f = _user_dir(username) / "log.jsonl"
    if not f.exists():
        return []
    lines = f.read_text("utf-8").splitlines()[-n:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln)["prompt"])
        except Exception:
            pass
    return out


def summarize_memory(username: str):
    """Atualiza o resumo master usando o Claude (best-effort)."""
    mem = load_memory(username)
    recents = recent_log(username, 14)
    if not recents:
        return
    prompt = (
        "Es um sistema de memoria de longo prazo. Atualiza um resumo CONCISO (PT-PT, "
        "no maximo 160 palavras) sobre o utilizador: quem parece ser, areas e temas "
        "recorrentes, e o tipo de pedidos que costuma fazer. Combina o RESUMO ATUAL com "
        "as INTERACOES RECENTES. Nao inventes nada; se nao houver dados, mantem vago. "
        "Devolve apenas o resumo, sem preambulos.\n\n"
        f"=== RESUMO ATUAL ===\n{mem.get('master') or '(vazio)'}\n\n"
        f"=== INTERACOES RECENTES ===\n- " + "\n- ".join(recents)
    )
    try:
        text = "".join(respond("claude", prompt, new_thread_id(), api_key_for("claude")))
        text = text.strip()
        if text and not text.lower().startswith(("> erro", "> falha", "> limite")):
            mem["master"] = text
            mem["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            mem["turns_since"] = 0
            save_memory(username, mem)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Extracao de ficheiros (PDF / PPTX / imagem + OCR)
# ---------------------------------------------------------------------------
def ocr_image(img) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang=OCR_LANG)


def extract_pdf(data: bytes) -> str:
    if not HAS_FITZ:
        return "(PyMuPDF nao instalado - pip install pymupdf)"
    doc = fitz.open(stream=data, filetype="pdf")
    out = []
    for i, page in enumerate(doc):
        t = page.get_text().strip()
        if len(t) < 10 and HAS_OCR:  # provavelmente pagina digitalizada
            pix = page.get_pixmap(dpi=200)
            t = ocr_image(Image.open(io.BytesIO(pix.tobytes("png")))).strip()
        if t:
            out.append(f"[pag {i + 1}]\n{t}")
    return "\n\n".join(out)


def extract_pptx(data: bytes) -> str:
    if not HAS_PPTX:
        return "(python-pptx nao instalado - pip install python-pptx)"
    prs = Presentation(io.BytesIO(data))
    out = []
    for i, slide in enumerate(prs.slides):
        parts = []
        for shp in slide.shapes:
            if shp.has_text_frame and shp.text_frame.text.strip():
                parts.append(shp.text_frame.text.strip())
            if getattr(shp, "has_table", False) and shp.has_table:
                for row in shp.table.rows:
                    parts.append(" | ".join(c.text for c in row.cells))
            if HAS_OCR and shp.shape_type == 13:  # PICTURE
                try:
                    o = ocr_image(Image.open(io.BytesIO(shp.image.blob))).strip()
                    if o:
                        parts.append("[imagem] " + o)
                except Exception:
                    pass
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append("[notas] " + slide.notes_slide.notes_text_frame.text.strip())
        if parts:
            out.append(f"[slide {i + 1}]\n" + "\n".join(parts))
    return "\n\n".join(out)


def extract_image(data: bytes) -> str:
    if not HAS_OCR:
        return "(OCR indisponivel - instala 'tesseract-ocr' e pillow)"
    return ocr_image(Image.open(io.BytesIO(data))).strip()


IMG_EXTS = ("png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif")


def _normalize_vecs(x):
    """Normaliza vetores para usar produto interno como similaridade cosseno."""
    if np is None:
        return x
    x = np.asarray(x, dtype="float32")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    """Carrega o modelo de embeddings uma vez por processo Streamlit."""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError(
            "sentence-transformers nao esta instalado. Instala com: "
            "pip install sentence-transformers"
        )
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_reranker_model(model_name: str):
    """Carrega o reranker neural uma vez por processo Streamlit."""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError(
            "sentence-transformers nao esta instalado. Instala com: "
            "pip install sentence-transformers"
        )
    return CrossEncoder(model_name)


def _embed_documents(model, texts: list[str]):
    """Gera embeddings de documentos/chunks, usando APIs especializadas quando existem."""
    if hasattr(model, "encode_document"):
        try:
            return _normalize_vecs(model.encode_document(texts, convert_to_numpy=True, show_progress_bar=False))
        except TypeError:
            return _normalize_vecs(model.encode_document(texts))
    try:
        return _normalize_vecs(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
    except TypeError:
        return _normalize_vecs(model.encode(texts, convert_to_numpy=True, show_progress_bar=False))


def _embed_query(model, query: str):
    """Gera embedding da pergunta, usando APIs especializadas quando existem."""
    if hasattr(model, "encode_query"):
        try:
            return _normalize_vecs(model.encode_query([query], convert_to_numpy=True, show_progress_bar=False))
        except TypeError:
            return _normalize_vecs(model.encode_query([query]))
    try:
        return _normalize_vecs(model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
    except TypeError:
        return _normalize_vecs(model.encode([query], convert_to_numpy=True, show_progress_bar=False))


def _split_text_chunks(text: str, chunk_size: int = ATTACHMENT_CHUNK_SIZE,
                       overlap: int = ATTACHMENT_CHUNK_OVERLAP) -> list[tuple[int, int, str]]:
    """Divide texto em chunks com sobreposicao, devolvendo (inicio, fim, texto)."""
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        # Tenta terminar numa fronteira natural para preservar frases/tabelas.
        if end < len(text):
            cut_candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
            ]
            cut = max(cut_candidates)
            if cut > start + chunk_size * 0.60:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= len(text):
            break
        start = max(0, end - overlap)
        if start >= end:
            start = end
    return chunks


def _doc_hash(name: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(name.encode("utf-8", "ignore"))
    h.update(b"\0")
    h.update((text or "").encode("utf-8", "ignore")[:200000])
    return h.hexdigest()[:16]


def _ensure_session_rag_docs():
    if "rag_docs" not in st.session_state:
        st.session_state.rag_docs = []
    if "rag_doc_hashes" not in st.session_state:
        st.session_state.rag_doc_hashes = set()


def add_docs_to_session_rag(docs: list[dict]) -> int:
    """Adiciona documentos extraidos ao índice lógico da sessão, evitando duplicados."""
    _ensure_session_rag_docs()
    added = 0
    for doc in docs:
        dh = _doc_hash(doc["name"], doc["text"])
        if dh in st.session_state.rag_doc_hashes:
            continue
        d = dict(doc)
        d["hash"] = dh
        st.session_state.rag_docs.append(d)
        st.session_state.rag_doc_hashes.add(dh)
        added += 1
    return added


def clear_session_rag():
    st.session_state.rag_docs = []
    st.session_state.rag_doc_hashes = set()


def _build_faiss_index(embeddings):
    if not HAS_FAISS:
        raise RuntimeError("faiss-cpu nao esta instalado. Instala com: pip install faiss-cpu")
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


def _retrieve_attachment_chunks_embeddings(
    docs: list[dict],
    query: str,
    top_k: int,
    embedding_model_name: str,
    use_reranker: bool = False,
    reranker_model_name: str = RERANKER_MODEL_OPTIONS[0],
    candidate_multiplier: int = ATTACHMENT_CANDIDATE_MULTIPLIER,
) -> tuple[str, dict]:
    """RAG real: chunks -> embeddings -> FAISS -> top-k -> reranker opcional."""
    if not docs:
        return "", {"mode": "empty", "selected": [], "total_chunks": 0}
    if not HAS_NUMPY:
        raise RuntimeError("numpy nao esta instalado. Instala com: pip install numpy")
    if not HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError("sentence-transformers nao esta instalado. Instala com: pip install sentence-transformers")
    if not HAS_FAISS:
        raise RuntimeError("faiss-cpu nao esta instalado. Instala com: pip install faiss-cpu")

    candidates = []
    for doc_i, doc in enumerate(docs):
        for chunk_i, (start, end, chunk) in enumerate(_split_text_chunks(doc.get("text", ""))):
            if len(chunk) < 40:
                continue
            candidates.append({
                "doc_i": doc_i,
                "file": doc.get("name", f"doc_{doc_i}"),
                "hash": doc.get("hash", ""),
                "chunk_i": chunk_i,
                "start": start,
                "end": end,
                "text": chunk,
            })

    if not candidates:
        return "", {"mode": "empty", "selected": [], "total_chunks": 0}

    model = load_embedding_model(embedding_model_name)
    chunk_texts = [c["text"] for c in candidates]
    doc_emb = _embed_documents(model, chunk_texts)
    query_emb = _embed_query(model, query)

    index = _build_faiss_index(doc_emb)
    n_candidates = min(len(candidates), max(top_k, top_k * candidate_multiplier))
    scores, idxs = index.search(query_emb.astype("float32"), n_candidates)

    ranked = []
    seen = set()
    for rank, (idx, score) in enumerate(zip(idxs[0].tolist(), scores[0].tolist()), 1):
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        item = dict(candidates[idx])
        item["semantic_rank"] = rank
        item["semantic_score"] = float(score)
        ranked.append(item)

    # Reranking neural opcional: mais lento, mas normalmente melhora respostas em documentos longos.
    if use_reranker and ranked:
        try:
            reranker = load_reranker_model(reranker_model_name)
            pairs = [[query, r["text"]] for r in ranked]
            rr_scores = reranker.predict(pairs)
            for r, rr in zip(ranked, rr_scores):
                r["rerank_score"] = float(rr)
            ranked = sorted(ranked, key=lambda r: r.get("rerank_score", 0.0), reverse=True)
        except Exception as e:
            for r in ranked:
                r["rerank_error"] = str(e)

    selected = ranked[:top_k]

    parts = [
        "[RAG COM EMBEDDINGS EM ANEXOS]",
        "Os trechos abaixo foram recuperados por similaridade semantica a partir dos anexos da sessao.",
        f"Modelo de embeddings: {embedding_model_name}",
        f"Reranker neural: {'ativo (' + reranker_model_name + ')' if use_reranker else 'inativo'}",
        "Usa estes trechos como evidencia. Se a resposta nao estiver nos trechos recuperados, diz isso explicitamente.",
        "Quando possivel, refere o ficheiro e o numero do trecho usado.",
        "[FIM NOTA RAG]",
    ]

    total = sum(len(d.get("text", "")) for d in docs)
    header = (
        f"Texto total indexado na sessao: {total:,} caracteres · "
        f"chunks avaliados: {len(candidates)} · candidatos reordenados: {len(ranked)} · "
        f"trechos enviados: {len(selected)}"
    )
    parts.append(header.replace(",", " "))

    used_chars = 0
    for n, c in enumerate(selected, 1):
        score_bits = [f"semantic_score: {c.get('semantic_score', 0):.4f}"]
        if "rerank_score" in c:
            score_bits.append(f"rerank_score: {c['rerank_score']:.4f}")
        if "rerank_error" in c:
            score_bits.append(f"reranker_error: {c['rerank_error'][:120]}")
        block = (
            f"\n--- TRECHO RAG {n} | ficheiro: {c['file']} | "
            f"chunk: {c['chunk_i'] + 1} | chars: {c['start']}-{c['end']} | "
            + " | ".join(score_bits)
            + " ---\n"
            + c["text"]
        )
        if used_chars + len(block) > ATTACHMENT_CONTEXT_CAP:
            remaining = max(0, ATTACHMENT_CONTEXT_CAP - used_chars)
            if remaining > 1000:
                parts.append(block[:remaining] + "\n...[trecho cortado por limite de contexto]")
            break
        parts.append(block)
        used_chars += len(block)

    meta = {
        "mode": "embedding_rag",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name if use_reranker else None,
        "reranker_active": bool(use_reranker),
        "selected": [
            {k: c.get(k) for k in (
                "file", "chunk_i", "start", "end", "semantic_rank", "semantic_score", "rerank_score", "rerank_error"
            )}
            for c in selected
        ],
        "total_chunks": len(candidates),
        "total_chars": total,
    }
    return "\n".join(parts).strip(), meta


def extract_files(
    files,
    query: str = "",
    use_attachment_search: bool = True,
    attachment_top_k: int = ATTACHMENT_TOP_K_DEFAULT,
    embedding_model_name: str = EMBEDDING_MODEL_OPTIONS[0],
    use_reranker: bool = False,
    reranker_model_name: str = RERANKER_MODEL_OPTIONS[0],
) -> tuple[str, list[str], list[str], dict]:
    """Devolve (texto_contexto, nomes, avisos, meta).

    - Extrai texto de PDF/PPTX/imagem/TXT.
    - Guarda os documentos numa base RAG persistente na sessao.
    - Se o RAG estiver ativo, recupera semanticamente os trechos mais relevantes.
    - Se o RAG estiver inativo, usa o modo antigo com limites/truncagem.
    """
    docs, names, warnings = [], [], []
    for f in files:
        name = f.name
        names.append(name)
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        data = f.getvalue()

        if ext == "pdf" and not HAS_FITZ:
            warnings.append(f"{name}: instala 'pymupdf' para ler PDF.")
            continue
        if ext in ("pptx", "ppt") and not HAS_PPTX:
            warnings.append(f"{name}: instala 'python-pptx' para ler PPTX.")
            continue
        if ext in IMG_EXTS and not HAS_OCR:
            warnings.append(f"{name}: OCR indisponivel (instala 'tesseract-ocr' + pillow).")
            continue

        try:
            if ext == "pdf":
                txt = extract_pdf(data)
            elif ext in ("pptx", "ppt"):
                txt = extract_pptx(data)
            elif ext in IMG_EXTS:
                txt = extract_image(data)
            else:
                txt = data.decode("utf-8", "ignore")
        except Exception as e:
            warnings.append(f"{name}: falha a ler ({e}).")
            continue

        txt = (txt or "").strip()
        if not txt:
            warnings.append(f"{name}: sem texto extraido (PDF digitalizado? liga o OCR).")
            continue
        docs.append({"name": name, "text": txt})

    if not docs:
        return "", names, warnings, {"mode": "empty"}

    added = add_docs_to_session_rag(docs)
    if added:
        warnings.append(f"{added} documento(s) adicionados ao índice RAG da sessão.")

    raw_total = sum(len(d["text"]) for d in docs)

    if use_attachment_search:
        ctx, meta = _retrieve_attachment_chunks_embeddings(
            st.session_state.rag_docs,
            query,
            max(1, attachment_top_k),
            embedding_model_name,
            use_reranker=use_reranker,
            reranker_model_name=reranker_model_name,
        )
        return ctx, names, warnings, meta

    # Modo antigo: direto/truncado.
    chunks = []
    for doc in docs:
        txt = doc["text"]
        if len(txt) > PER_FILE_CAP:
            txt = txt[:PER_FILE_CAP] + "\n...[ficheiro truncado]"
        chunks.append(f"--- {doc['name']} ---\n{txt}")

    full = "\n\n".join(chunks)
    if len(full) > TOTAL_CAP:
        full = full[:TOTAL_CAP] + "\n...[total truncado]"
    return full, names, warnings, {"mode": "direct", "total_chars": raw_total}


def retrieve_session_rag_context(
    query: str,
    attachment_top_k: int = ATTACHMENT_TOP_K_DEFAULT,
    embedding_model_name: str = EMBEDDING_MODEL_OPTIONS[0],
    use_reranker: bool = False,
    reranker_model_name: str = RERANKER_MODEL_OPTIONS[0],
) -> tuple[str, dict]:
    """Recupera contexto da base RAG da sessão mesmo sem novos anexos."""
    _ensure_session_rag_docs()
    if not st.session_state.rag_docs:
        return "", {"mode": "empty"}
    return _retrieve_attachment_chunks_embeddings(
        st.session_state.rag_docs,
        query,
        max(1, attachment_top_k),
        embedding_model_name,
        use_reranker=use_reranker,
        reranker_model_name=reranker_model_name,
    )


# ---------------------------------------------------------------------------
# Chaves de API
# ---------------------------------------------------------------------------
def api_key_for(model_key: str) -> str:
    cfg = MODELS[model_key]
    env_value = os.getenv(cfg["env"])
    if env_value:
        return env_value.strip()
    try:
        if cfg["env"] in st.secrets:
            return str(st.secrets[cfg["env"]]).strip()
    except Exception:
        pass
    st.error(f"Secret em falta: configura `{cfg['env']}` no ambiente de deploy.")
    st.stop()


def new_thread_id() -> str:
    return secrets.token_urlsafe(16)


# ---------------------------------------------------------------------------
# Rate limiting: retry com backoff + throttle global
# ---------------------------------------------------------------------------
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BASE_DELAY = 2.0
# Chunk pequeno para o requests entregar eventos assim que chegam.
# Se estiver None, alguns servidores/proxies acumulam e o texto so aparece no fim.
STREAM_CHUNK_SIZE = 64
_min_interval = [1.2]
_throttle_lock = threading.Lock()
_last_request_ts = [0.0]


def _throttle():
    with _throttle_lock:
        wait = _min_interval[0] - (time.time() - _last_request_ts[0])
        if wait > 0:
            time.sleep(wait)
        _last_request_ts[0] = time.time()


def _is_rate_limit(text) -> bool:
    t = str(text).lower()
    return any(s in t for s in ("429", "rate limit", "rate-limit", "too many requests"))


# ---------------------------------------------------------------------------
# Streaming do agente IAedu
# ---------------------------------------------------------------------------
def _decode_stream(cfg: dict, api_key: str, thread_id: str, message: str):
    fields = {
        "channel_id": (None, cfg["channel_id"]),
        "thread_id": (None, thread_id),
        "user_info": (None, "{}"),
        "message": (None, message),
    }
    headers = {"x-api-key": api_key, "Accept": "text/event-stream"}
    decoder = json.JSONDecoder()
    buffer = ""
    got_tokens = False
    fallback_msg = None

    _throttle()
    with requests.post(cfg["endpoint"], headers=headers, files=fields,
                       stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_content(chunk_size=STREAM_CHUNK_SIZE, decode_unicode=True):
            if not raw:
                continue
            buffer += raw
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                try:
                    event, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                buffer = buffer[idx:]
                if not isinstance(event, dict):
                    continue
                etype = event.get("type")
                if etype == "token":
                    tok = event.get("content", "")
                    if tok:
                        got_tokens = True
                        yield ("token", tok)
                elif etype == "message":
                    content = event.get("content", {})
                    if isinstance(content, dict):
                        fallback_msg = content.get("content")
                elif etype == "context_limit":
                    yield ("context_limit", None)
                    return
                elif etype == "error":
                    yield ("error", str(event.get("content", "erro desconhecido")))
                    return

    if not got_tokens and fallback_msg:
        yield ("token", fallback_msg)


def respond(model_key: str, message: str, thread_id: str, api_key: str, persist: bool = False):
    cfg = MODELS[model_key]
    tid = thread_id
    ctx_retried = False
    rl_attempts = 0
    while True:
        yielded_any = False
        hit_ctx = False
        rate_limited = False
        try:
            for kind, payload in _decode_stream(cfg, api_key, tid, message):
                if kind == "token":
                    yielded_any = True
                    yield payload
                elif kind == "context_limit":
                    hit_ctx = True
                    tid = new_thread_id()
                    if persist:
                        st.session_state.threads[model_key] = tid
                    break
                elif kind == "error":
                    if _is_rate_limit(payload) and not yielded_any:
                        rate_limited = True
                        break
                    yield f"\n\n> Erro do agente: {payload}"
                    return
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429 and not yielded_any:
                rate_limited = True
            else:
                yield f"\n\n> Erro HTTP ({cfg['label']}): {e}"
                return
        except Exception as e:
            yield f"\n\n> Falha de ligacao ({cfg['label']}): {e}"
            return

        if rate_limited:
            if rl_attempts < RATE_LIMIT_RETRIES:
                rl_attempts += 1
                delay = RATE_LIMIT_BASE_DELAY * (2 ** (rl_attempts - 1))
                yield (f"\n\n_(limite de pedidos atingido - nova tentativa em "
                       f"{delay:.0f}s . {rl_attempts}/{RATE_LIMIT_RETRIES})_\n\n")
                time.sleep(delay)
                continue
            yield (f"\n\n> Limite de pedidos do IAedu (429) persistente apos "
                   f"{RATE_LIMIT_RETRIES} tentativas. Espera um pouco e tenta de novo.")
            return

        if hit_ctx and not ctx_retried:
            ctx_retried = True
            yield "\n\n_(contexto cheio - recomecei a conversa deste modelo)_\n\n"
            continue
        return


# ---------------------------------------------------------------------------
# Estimativas
# ---------------------------------------------------------------------------
def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def est_cost(prompt: str, output: str, model_key: str) -> float:
    cfg = MODELS[model_key]
    return (est_tokens(prompt) / 1_000_000 * cfg["price_in"]
            + est_tokens(output) / 1_000_000 * cfg["price_out"])


# ---------------------------------------------------------------------------
# Composicao da mensagem (memoria + transcricao da sessao + anexos + pergunta)
# ---------------------------------------------------------------------------
def _compact_for_context(text: str, cap: int = 6000) -> str:
    """Limita blocos muito longos sem rebentar o contexto."""
    text = (text or "").strip()
    if len(text) <= cap:
        return text
    head = cap // 2
    tail = cap - head
    return text[:head] + "\n...[texto intermédio omitido]...\n" + text[-tail:]


def _assistant_msg_to_text(msg: dict) -> str:
    """Converte os vários modos da UI para texto utilizável como contexto."""
    mode = msg.get("mode")
    if mode == "single":
        model = MODELS.get(msg.get("model", ""), {}).get("label", msg.get("model", "assistant"))
        return f"Assistente ({model}):\n{msg.get('content', '')}"

    if mode == "duel":
        data = msg.get("data", {})
        return (
            "Assistente (Duelo):\n"
            f"[Claude]\n{data.get('claude', '')}\n\n"
            f"[GPT]\n{data.get('gpt', '')}"
        )

    if mode == "collab":
        ex = MODELS.get(msg.get("executor", ""), {}).get("label", msg.get("executor", "executor"))
        rv = MODELS.get(msg.get("reviewer", ""), {}).get("label", msg.get("reviewer", "reviewer"))
        parts = [
            f"Assistente (Colaboracao; executor={ex}; revisor={rv}):",
            f"[Rascunho]\n{msg.get('draft', '')}",
            f"[Revisao]\n{msg.get('critique', '')}",
        ]
        if msg.get("synthesis"):
            parts.append(f"[Versao final]\n{msg.get('synthesis', '')}")
        return "\n\n".join(parts)

    return f"Assistente:\n{msg.get('content', '')}"


def session_transcript(max_chars: int = SESSION_TRANSCRIPT_CAP) -> str:
    """Transcricao da conversa atual, em janela deslizante.

    Usa st.session_state.messages, que ja alimenta o render_history().
    A janela e montada do fim para o inicio para preservar os turnos mais recentes.
    """
    messages = st.session_state.get("messages", [])
    blocks = []

    for msg in messages:
        if msg.get("role") == "user":
            extra = ""
            if msg.get("attachments"):
                extra = "\n[anexos neste turno: " + ", ".join(msg["attachments"]) + "]"
            blocks.append("Utilizador:\n" + msg.get("content", "") + extra)
        elif msg.get("role") == "assistant":
            blocks.append(_assistant_msg_to_text(msg))

    selected = []
    total = 0
    for block in reversed(blocks):
        block = _compact_for_context(block, cap=6000)
        add = len(block) + 2
        if selected and total + add > max_chars:
            break
        selected.append(block)
        total += add

    selected.reverse()
    return "\n\n---\n\n".join(selected).strip()


def build_agent_message(
    user_prompt: str,
    username: str,
    use_memory: bool,
    attach_text: str,
    use_session_context: bool = True,
) -> str:
    parts = []

    if use_memory:
        mem = load_memory(username).get("master", "").strip()
        if mem:
            parts.append("[MEMORIA DO UTILIZADOR - contexto de fundo, nao e a pergunta]\n"
                         + mem + "\n[FIM MEMORIA]")

    if use_session_context:
        hist = session_transcript()
        if hist:
            parts.append(
                "[TRANSCRICAO DA CONVERSA ATUAL - usa para manter continuidade; "
                "nao repitas a menos que seja util]\n"
                + hist +
                "\n[FIM TRANSCRICAO]"
            )

    if attach_text:
        parts.append("[FICHEIROS ANEXADOS NESTE TURNO]\n" + attach_text + "\n[FIM FICHEIROS]")

    parts.append("[PERGUNTA ATUAL]\n" + user_prompt)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tema / CSS / Hero
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{
  --bg:#0B0E14; --bg-2:#11151F; --bg-3:#161B26;
  --line:#222A38; --line-2:#2E3850;
  --text:#E7EBF3; --muted:#8B94A7; --faint:#5B6577;
  --claude:#D2A24C; --gpt:#46B98C; --accent:#D2A24C;
  --font-display:'Fraunces',Georgia,serif;
  --font-ui:'IBM Plex Sans',system-ui,sans-serif;
  --font-mono:'JetBrains Mono',ui-monospace,monospace;
}
.stApp,[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 560px at 82% -12%, rgba(210,162,76,0.07), transparent 60%),
    radial-gradient(900px 520px at -8% 6%, rgba(70,185,140,0.055), transparent 55%),
    var(--bg);
  color:var(--text); font-family:var(--font-ui);
}
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stMainBlockContainer"],.block-container{ max-width:1080px; padding-top:1.1rem; padding-bottom:5rem; }
body,p,li,span,div{ font-family:var(--font-ui); }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,var(--bg-2),#0E121B); border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container{ padding-top:1.4rem; }
.eyebrow{ font-family:var(--font-mono); text-transform:uppercase; letter-spacing:.22em; font-size:.64rem;
  color:var(--faint); margin:1.1rem 0 .55rem; display:flex; align-items:center; gap:.6rem; }
.eyebrow::after{ content:""; height:1px; flex:1; background:var(--line); }
[data-testid="stChatMessage"]{ background:var(--bg-2); border:1px solid var(--line); border-radius:16px;
  padding:1.05rem 1.25rem; box-shadow:0 1px 0 rgba(255,255,255,.02),0 14px 34px rgba(0,0,0,.30); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:transparent; border:1px dashed var(--line-2); box-shadow:none; }
[data-testid^="stChatMessageAvatar"]{ display:none; }
[data-testid="stChatMessage"] > div:first-child{ gap:0 !important; }
[data-testid="stChatMessage"] p{ line-height:1.62; }
[data-testid="stChatMessage"] code{ font-family:var(--font-mono); background:#0c1119; border:1px solid var(--line);
  padding:.05rem .35rem; border-radius:6px; font-size:.86em; }
[data-testid="stChatMessage"] pre{ background:#0c1119 !important; border:1px solid var(--line); border-radius:12px; }
a{ color:var(--accent); }
[data-testid="stChatInput"]{ background:var(--bg-2); border:1px solid var(--line); border-radius:14px; }
[data-testid="stChatInput"]:focus-within{ border-color:var(--accent); box-shadow:0 0 0 3px rgba(210,162,76,.14); }
.stButton > button{ background:transparent; color:var(--text); border:1px solid var(--line-2); border-radius:10px;
  font-family:var(--font-mono); text-transform:uppercase; letter-spacing:.1em; font-size:.72rem;
  padding:.5rem .9rem; transition:all .15s ease; }
.stButton > button:hover{ border-color:var(--accent); color:var(--accent); background:rgba(210,162,76,.06); }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:12px; background:var(--bg-2); }
[data-testid="stExpander"] summary{ font-family:var(--font-mono); font-size:.78rem; letter-spacing:.04em; }
[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--bg-3); border-radius:14px; }
[data-testid="stFileUploaderDropzone"]{ background:var(--bg-3); border:1px dashed var(--line-2); border-radius:12px; }
.badge{ display:inline-flex; align-items:center; gap:.55rem; font-family:var(--font-mono); font-size:.72rem;
  padding:.32rem .7rem; border:1px solid var(--line-2); border-radius:999px; background:var(--bg-3); color:var(--text); }
.badge .dot{ width:8px; height:8px; border-radius:50%; box-shadow:0 0 10px 0 currentColor; }
.badge--claude{ border-color:rgba(210,162,76,.4); } .badge--claude .dot{ background:var(--claude); color:var(--claude); }
.badge--gpt{ border-color:rgba(70,185,140,.4); } .badge--gpt .dot{ background:var(--gpt); color:var(--gpt); }
.badge .sub{ color:var(--faint); margin-left:.35rem; }
.reason{ margin:.55rem 0 .2rem; padding:.6rem .85rem; border-left:2px solid var(--line-2); background:var(--bg-3);
  border-radius:0 10px 10px 0; color:var(--muted); font-size:.85rem; line-height:1.5; }
.reason b{ color:var(--text); font-weight:600; }
.reason--claude{ border-left-color:var(--claude); } .reason--gpt{ border-left-color:var(--gpt); }
.metricline{ font-family:var(--font-mono); font-size:.7rem; color:var(--muted); display:flex; gap:.55rem;
  flex-wrap:wrap; margin-top:.5rem; } .metricline .sep{ color:var(--faint); }
.step{ font-family:var(--font-mono); text-transform:uppercase; letter-spacing:.14em; font-size:.7rem;
  color:var(--muted); margin:1.1rem 0 .35rem; display:flex; align-items:center; gap:.6rem; }
.step .n{ width:1.35rem; height:1.35rem; border-radius:6px; background:var(--text); color:var(--bg);
  display:inline-flex; align-items:center; justify-content:center; font-weight:600; font-size:.72rem; }
.col-head{ display:flex; align-items:center; justify-content:space-between; margin:.2rem 0 .5rem; }
.routerule{ display:flex; gap:.55rem; align-items:flex-start; padding:.35rem 0; font-size:.8rem;
  color:var(--text); border-bottom:1px solid var(--line); }
.routerule:last-child{ border-bottom:none; }
.routerule .d{ width:8px; height:8px; border-radius:50%; margin-top:.42rem; flex:none; }
.routerule .b{ display:block; font-family:var(--font-mono); font-size:.66rem; color:var(--faint); margin-top:.1rem; }
.attach-note{ font-family:var(--font-mono); font-size:.68rem; color:var(--faint); margin-top:.4rem; }
.userbox{ display:flex; align-items:center; gap:.6rem; padding:.55rem .7rem; border:1px solid var(--line);
  border-radius:12px; background:var(--bg-3); margin-bottom:.4rem; }
.userbox .av{ width:30px; height:30px; border-radius:8px; background:linear-gradient(135deg,var(--claude),var(--gpt));
  display:flex; align-items:center; justify-content:center; font-family:var(--font-mono); font-weight:600; color:#0B0E14; }
.userbox .n{ font-size:.85rem; } .userbox .r{ font-family:var(--font-mono); font-size:.62rem; color:var(--faint); }
.authwrap{ max-width:420px; margin:.5rem auto 0; }
.authwrap h3{ font-family:var(--font-display); font-weight:600; font-size:1.4rem; margin-bottom:.2rem; }
.authwrap p.s{ color:var(--muted); font-size:.85rem; margin-bottom:.4rem; }
</style>
"""

HERO_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box;}
html,body{height:100%;overflow:hidden;background:#0B0E14;}
#c{position:absolute;inset:0;}
.wrap{position:relative;height:100%;display:flex;flex-direction:column;justify-content:center;
  padding:26px 30px;font-family:'IBM Plex Sans',sans-serif;color:#E7EBF3;}
.eye{font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.34em;
  font-size:11px;color:#5B6577;margin-bottom:12px;}
h1{font-family:'Fraunces',serif;font-weight:600;font-size:46px;line-height:1;letter-spacing:-.5px;}
h1 .x{color:#D2A24C;font-style:italic;margin:0 10px;font-weight:500;}
.sub{margin-top:14px;font-family:'JetBrains Mono',monospace;font-size:13px;color:#8B94A7;height:18px;}
.sub .cur{color:#D2A24C;}
.pills{position:absolute;top:26px;right:30px;display:flex;gap:8px;}
.pill{font-family:'JetBrains Mono',monospace;font-size:11px;color:#E7EBF3;border:1px solid #2E3850;
  border-radius:999px;padding:5px 11px;display:flex;align-items:center;gap:7px;background:rgba(22,27,38,.6);}
.pill .d{width:7px;height:7px;border-radius:50%;}
.line{position:absolute;left:0;right:0;bottom:0;height:1px;
  background:linear-gradient(90deg,transparent,#D2A24C55,#46B98C55,transparent);}
</style></head><body>
<canvas id="c"></canvas>
<div class="pills">
  <div class="pill"><span class="d" style="background:#D2A24C;box-shadow:0 0 8px #D2A24C"></span>Opus 4.7</div>
  <div class="pill"><span class="d" style="background:#46B98C;box-shadow:0 0 8px #46B98C"></span>GPT-5.5</div>
</div>
<div class="wrap">
  <div class="eye">IAedu &middot; Multi-Model Router</div>
  <h1>Claude<span class="x">&times;</span>GPT</h1>
  <div class="sub"><span id="t"></span><span class="cur">_</span></div>
</div>
<div class="line"></div>
<script>
var canvas=document.getElementById('c'),ctx=canvas.getContext('2d'),W,H,pts=[];
function size(){W=canvas.width=canvas.offsetWidth;H=canvas.height=canvas.offsetHeight;
  pts=[];var n=Math.floor(W/55);for(var i=0;i<n;i++){pts.push({x:Math.random()*W,y:Math.random()*H,
  vx:(Math.random()-.5)*.25,vy:(Math.random()-.5)*.25});}}
size();window.addEventListener('resize',size);
function draw(){ctx.clearRect(0,0,W,H);
  for(var i=0;i<pts.length;i++){var p=pts[i];p.x+=p.vx;p.y+=p.vy;
    if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;}
  for(var i=0;i<pts.length;i++){for(var j=i+1;j<pts.length;j++){
    var a=pts[i],b=pts[j],dx=a.x-b.x,dy=a.y-b.y,d=Math.sqrt(dx*dx+dy*dy);
    if(d<130){ctx.strokeStyle='rgba(139,148,167,'+(.12*(1-d/130))+')';ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}}}
  for(var i=0;i<pts.length;i++){ctx.fillStyle='rgba(210,162,76,.5)';
    ctx.beginPath();ctx.arc(pts[i].x,pts[i].y,1.3,0,6.3);ctx.fill();}
  requestAnimationFrame(draw);}
draw();
var phrases=["Roteamento inteligente por tarefa","Memoria do utilizador entre sessoes",
  "Upload de PDF, PPTX e imagem com OCR","Duelo e colaboracao entre modelos"];
var pi=0,ci=0,del=false,el=document.getElementById('t');
function type(){var w=phrases[pi];el.textContent=w.substring(0,ci);
  if(!del){ci++;if(ci>w.length){del=true;setTimeout(type,1600);return;}}
  else{ci--;if(ci<0){del=false;pi=(pi+1)%phrases.length;ci=0;}}
  setTimeout(type,del?28:55);}
type();
</script></body></html>
"""


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------
def model_badge(k: str, sub: str | None = None) -> str:
    m = MODELS[k]
    sub_html = f"<span class='sub'>{sub}</span>" if sub else ""
    return f"<span class='badge badge--{k}'><span class='dot'></span>{m['label']}{sub_html}</span>"


def reason_card(k: str, text: str) -> str:
    return f"<div class='reason reason--{k}'>{text}</div>"


def metric_html(k: str, output: str, secs: float, prompt: str) -> str:
    return (f"<div class='metricline'><span>~{est_tokens(output)} tok</span>"
            f"<span class='sep'>/</span><span>{secs:.1f}s</span>"
            f"<span class='sep'>/</span><span>~${est_cost(prompt, output, k):.4f}</span></div>")


def step_html(n: int, text: str) -> str:
    return f"<div class='step'><span class='n'>{n}</span>{text}</div>"


def attach_note(names: list[str]) -> str:
    return f"<div class='attach-note'>anexos: {', '.join(names)}</div>"


# ---------------------------------------------------------------------------
# Render de historico
# ---------------------------------------------------------------------------
def render_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
                if msg.get("attachments"):
                    st.markdown(attach_note(msg["attachments"]), unsafe_allow_html=True)
            continue
        with st.chat_message("assistant"):
            mode = msg.get("mode")
            if mode == "single":
                head = model_badge(msg["model"])
                if msg.get("reason"):
                    head += reason_card(msg["model"], msg["reason"])
                st.markdown(head, unsafe_allow_html=True)
                st.markdown(msg["content"])
            elif mode == "duel":
                cols = st.columns(2)
                for col, k in zip(cols, ["claude", "gpt"]):
                    with col:
                        st.markdown(f"<div class='col-head'>{model_badge(k)}</div>", unsafe_allow_html=True)
                        with st.container(border=True):
                            st.markdown(msg["data"].get(k, ""))
                        if msg["meta"].get(k):
                            st.markdown(msg["meta"][k], unsafe_allow_html=True)
            elif mode == "collab":
                ex, rv = msg["executor"], msg["reviewer"]
                st.markdown("Executor " + model_badge(ex) + " &nbsp; Revisor " + model_badge(rv),
                            unsafe_allow_html=True)
                st.markdown(step_html(1, f"Rascunho - {MODELS[ex]['label']}"), unsafe_allow_html=True)
                st.markdown(msg["draft"])
                st.markdown(step_html(2, f"Revisao critica - {MODELS[rv]['label']}"), unsafe_allow_html=True)
                st.markdown(msg["critique"])
                if msg.get("synthesis"):
                    st.markdown(step_html(3, f"Versao final - {MODELS[ex]['label']}"), unsafe_allow_html=True)
                    st.markdown(msg["synthesis"])


# ---------------------------------------------------------------------------
# Modos de resposta
# ---------------------------------------------------------------------------
def handle_single(send_text, display_text, forced_model, show_reason):
    if forced_model:
        model_key, reason = forced_model, "Modelo selecionado manualmente."
    else:
        decision = route(display_text, total_len=len(send_text))
        model_key, reason = decision["model"], decision["reason"]
    with st.chat_message("assistant"):
        head = model_badge(model_key)
        if show_reason:
            head += reason_card(model_key, reason)
        st.markdown(head, unsafe_allow_html=True)
        full = st.write_stream(respond(model_key, send_text,
                                       st.session_state.threads[model_key],
                                       api_key_for(model_key), persist=True))
    st.session_state.messages.append({"role": "assistant", "mode": "single", "model": model_key,
                                      "reason": reason if show_reason else "", "content": full})


def handle_duel(send_text, display_text):
    order = ["claude", "gpt"]
    with st.chat_message("assistant"):
        cols = st.columns(2)
        ph, metaph = {}, {}
        for col, k in zip(cols, order):
            with col:
                st.markdown(f"<div class='col-head'>{model_badge(k)}</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    ph[k] = st.empty()
                metaph[k] = st.empty()
        tids = {k: st.session_state.threads[k] for k in order}
        keys = {k: api_key_for(k) for k in order}
        qs = {k: queue.Queue() for k in order}
        results = {k: "" for k in order}
        starts = {k: time.time() for k in order}
        times = {k: 0.0 for k in order}

        def worker(k):
            try:
                for tok in respond(k, send_text, tids[k], keys[k], persist=False):
                    qs[k].put(tok)
            except Exception as e:  # noqa
                qs[k].put(f"\n\n> Erro: {e}")
            finally:
                qs[k].put(None)

        threads = {k: threading.Thread(target=worker, args=(k,), daemon=True) for k in order}
        for t in threads.values():
            t.start()
        done = {k: False for k in order}
        while not all(done.values()):
            for k in order:
                updated = False
                try:
                    while True:
                        item = qs[k].get_nowait()
                        if item is None:
                            done[k] = True
                            times[k] = time.time() - starts[k]
                        else:
                            results[k] += item
                            updated = True
                except queue.Empty:
                    pass
                if updated:
                    ph[k].markdown(results[k] + " |")
            time.sleep(0.03)
        meta = {}
        for k in order:
            ph[k].markdown(results[k])
            meta[k] = metric_html(k, results[k], times[k], send_text)
            metaph[k].markdown(meta[k], unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "mode": "duel", "data": results, "meta": meta})


def handle_collab(send_text, display_text, do_synthesis):
    decision = route(display_text, total_len=len(send_text))
    executor = decision["model"]
    reviewer = "gpt" if executor == "claude" else "claude"
    ex_key, rv_key = api_key_for(executor), api_key_for(reviewer)
    with st.chat_message("assistant"):
        st.markdown("Executor " + model_badge(executor) + " &nbsp; Revisor " + model_badge(reviewer),
                    unsafe_allow_html=True)
        st.markdown(reason_card(executor, decision["reason"]), unsafe_allow_html=True)
        st.markdown(step_html(1, f"Rascunho - {MODELS[executor]['label']}"), unsafe_allow_html=True)
        draft = st.write_stream(respond(executor, send_text, new_thread_id(), ex_key))
        critique_prompt = (
            "Es um revisor tecnico critico e rigoroso. Outro assistente respondeu ao pedido "
            "abaixo. Identifica erros factuais, lacunas, ambiguidades, riscos e suposicoes "
            "frageis, e propoe melhorias CONCRETAS e acionaveis. Se direto; nao repitas tudo.\n\n"
            f"=== PEDIDO ===\n{display_text}\n\n=== RESPOSTA A REVER ===\n{draft}"
        )
        st.markdown(step_html(2, f"Revisao critica - {MODELS[reviewer]['label']}"), unsafe_allow_html=True)
        critique = st.write_stream(respond(reviewer, critique_prompt, new_thread_id(), rv_key))
        synthesis = ""
        if do_synthesis:
            synth_prompt = (
                "Tens o teu rascunho e uma revisao critica. Produz a VERSAO FINAL melhorada, "
                "integrando o feedback valido. Entrega so a resposta final.\n\n"
                f"=== PEDIDO ===\n{display_text}\n\n=== RASCUNHO ===\n{draft}\n\n"
                f"=== REVISAO ===\n{critique}"
            )
            st.markdown(step_html(3, f"Versao final - {MODELS[executor]['label']}"), unsafe_allow_html=True)
            synthesis = st.write_stream(respond(executor, synth_prompt, new_thread_id(), ex_key))
    st.session_state.messages.append({"role": "assistant", "mode": "collab", "executor": executor,
                                      "reviewer": reviewer, "draft": draft, "critique": critique,
                                      "synthesis": synthesis})


# ---------------------------------------------------------------------------
# Sessao / Auth UI
# ---------------------------------------------------------------------------
def reset_session():
    st.session_state.messages = []
    st.session_state.threads = {k: new_thread_id() for k in MODELS}
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    clear_session_rag()


def render_auth():
    users = _load_users()
    st.markdown("<div class='authwrap'>", unsafe_allow_html=True)
    if not users:
        st.markdown("<h3>Criar conta de administrador</h3>"
                    "<p class='s'>Primeiro arranque: define a tua conta.</p>", unsafe_allow_html=True)
        with st.form("setup"):
            name = st.text_input("Nome")
            username = st.text_input("Utilizador (a-z, 0-9, _ . -)")
            p1 = st.text_input("Password", type="password")
            p2 = st.text_input("Confirmar password", type="password")
            ok = st.form_submit_button("Criar conta", use_container_width=True)
        if ok:
            username = (username or "").strip().lower()
            if not USERNAME_RE.match(username):
                st.error("Utilizador invalido (3-32 chars: a-z, 0-9, _ . -).")
            elif len(p1) < 6:
                st.error("Password com pelo menos 6 caracteres.")
            elif p1 != p2:
                st.error("As passwords nao coincidem.")
            else:
                create_user(username, name or username, p1)
                st.session_state.auth = {"user": username, "name": name or username}
                reset_session()
                st.rerun()
    else:
        st.markdown("<h3>Entrar</h3><p class='s'>Acede com as tuas credenciais.</p>",
                    unsafe_allow_html=True)
        with st.form("login"):
            username = st.text_input("Utilizador")
            password = st.text_input("Password", type="password")
            ok = st.form_submit_button("Entrar", use_container_width=True)
        if ok:
            username = (username or "").strip().lower()
            if verify_user(username, password):
                st.session_state.auth = {"user": username, "name": _load_users()[username]["name"]}
                reset_session()
                st.rerun()
            else:
                st.error("Credenciais invalidas.")
        with st.expander("Criar nova conta"):
            with st.form("register"):
                name = st.text_input("Nome", key="r_name")
                u = st.text_input("Utilizador", key="r_user")
                p1 = st.text_input("Password", type="password", key="r_p1")
                p2 = st.text_input("Confirmar", type="password", key="r_p2")
                rok = st.form_submit_button("Registar", use_container_width=True)
            if rok:
                u = (u or "").strip().lower()
                if not USERNAME_RE.match(u):
                    st.error("Utilizador invalido.")
                elif u in users:
                    st.error("Esse utilizador ja existe.")
                elif len(p1) < 6 or p1 != p2:
                    st.error("Password fraca ou nao coincide.")
                else:
                    create_user(u, name or u, p1)
                    st.success("Conta criada. Ja podes entrar.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------------
def sidebar(username):
    with st.sidebar:
        u = st.session_state.auth
        initial = (u["name"][:1] or "?").upper()
        st.markdown(f"<div class='userbox'><div class='av'>{initial}</div>"
                    f"<div><div class='n'>{u['name']}</div><div class='r'>@{u['user']}</div></div></div>",
                    unsafe_allow_html=True)
        if st.button("Terminar sessao", use_container_width=True):
            st.session_state.auth = None
            reset_session()
            st.rerun()

        st.markdown("<div class='eyebrow'>Modo de operacao</div>", unsafe_allow_html=True)
        mode = st.radio("Modo", ["Auto (Roteador)", "So Claude", "So GPT",
                                 "Duelo (lado a lado)", "Colaboracao"],
                        index=0, label_visibility="collapsed")
        show_reason = st.toggle("Mostrar justificacao do roteamento", value=True)
        do_synthesis = st.toggle("Sintese final (colaboracao)", value=True) if mode == "Colaboracao" else False

        st.markdown("<div class='eyebrow'>Anexos (PDF, PPTX, imagem)</div>", unsafe_allow_html=True)
        files = st.file_uploader("Anexar", type=["pdf", "pptx", "ppt", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
                                 accept_multiple_files=True, label_visibility="collapsed",
                                 key=f"uploader_{st.session_state.uploader_key}")
        if files:
            st.caption(f"{len(files)} ficheiro(s) sera(o) anexado(s) ao proximo pedido.")
        if not HAS_OCR:
            st.caption("OCR off: instala 'tesseract-ocr' p/ ler imagens e PDFs digitalizados.")

        st.markdown("<div class='eyebrow'>RAG dos anexos</div>", unsafe_allow_html=True)
        use_attachment_search = st.toggle("Usar RAG com embeddings", value=True)
        attachment_top_k = st.slider("Trechos RAG a enviar", 2, 16, ATTACHMENT_TOP_K_DEFAULT, 1)
        embedding_model_name = st.selectbox("Modelo de embeddings", EMBEDDING_MODEL_OPTIONS, index=0)
        use_reranker = st.toggle("Reranker neural", value=False, help="Mais preciso em documentos longos, mas mais lento e pesado.")
        reranker_model_name = st.selectbox("Modelo reranker", RERANKER_MODEL_OPTIONS, index=0, disabled=not use_reranker)
        _ensure_session_rag_docs()
        if st.session_state.rag_docs:
            total_chars = sum(len(d.get("text", "")) for d in st.session_state.rag_docs)
            st.caption(f"Índice da sessão: {len(st.session_state.rag_docs)} doc(s), {total_chars:,} caracteres".replace(",", " "))
            if st.button("Limpar índice RAG", use_container_width=True):
                clear_session_rag()
                st.rerun()
        missing = []
        if not HAS_SENTENCE_TRANSFORMERS:
            missing.append("sentence-transformers")
        if not HAS_FAISS:
            missing.append("faiss-cpu")
        if not HAS_NUMPY:
            missing.append("numpy")
        if missing:
            st.caption("RAG embeddings indisponível: instala " + ", ".join(missing))

        st.markdown("<div class='eyebrow'>Memoria</div>", unsafe_allow_html=True)
        use_memory = st.toggle("Usar memoria master no contexto", value=True)
        use_session_context = st.toggle("Usar conversa atual como contexto", value=True)
        auto_mem = st.toggle("Atualizar memoria automaticamente", value=True)
        mem = load_memory(username)
        with st.expander("Ver / editar memoria master"):
            edited = st.text_area("Resumo", value=mem.get("master", ""), height=160,
                                  label_visibility="collapsed")
            c1, c2 = st.columns(2)
            if c1.button("Guardar", use_container_width=True):
                mem["master"] = edited
                mem["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
                save_memory(username, mem)
                st.toast("Memoria guardada.")
            if c2.button("Atualizar agora", use_container_width=True):
                with st.spinner("A resumir..."):
                    summarize_memory(username)
                st.rerun()
            if mem.get("updated_at"):
                st.caption(f"Atualizada: {mem['updated_at']}")
            if st.button("Limpar memoria", use_container_width=True):
                save_memory(username, {"master": "", "updated_at": None, "turns_since": 0})
                st.rerun()

        st.markdown("<div class='eyebrow'>Sessao</div>", unsafe_allow_html=True)
        if st.button("Nova conversa", use_container_width=True):
            reset_session()
            st.rerun()

        st.markdown("<div class='eyebrow'>Avancado</div>", unsafe_allow_html=True)
        _min_interval[0] = st.slider("Intervalo minimo entre pedidos (s)", 0.0, 5.0, _min_interval[0], 0.1)

        with st.expander("Como funciona o roteamento"):
            rows = ""
            for r in ROUTING_RULES:
                c = MODELS[r["model"]]["color"]
                rows += (f"<div class='routerule'><span class='d' style='background:{c}'></span>"
                         f"<div>{r['label']}<span class='b'>{MODELS[r['model']]['label']} &middot; {r['bench']}</span></div></div>")
            c = MODELS[DEFAULT_MODEL]["color"]
            rows += (f"<div class='routerule'><span class='d' style='background:{c}'></span>"
                     f"<div>Geral / sem sinal<span class='b'>{MODELS[DEFAULT_MODEL]['label']} &middot; default</span></div></div>")
            st.markdown(rows, unsafe_allow_html=True)

    return (mode, show_reason, do_synthesis, files, use_memory, use_session_context, auto_mem,
            use_attachment_search, attachment_top_k, embedding_model_name, use_reranker, reranker_model_name)


def main():
    st.set_page_config(page_title="Claude x GPT - Roteador", page_icon="*", layout="wide")
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "auth" not in st.session_state:
        st.session_state.auth = None
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    components.html(HERO_HTML, height=240, scrolling=False)

    if not st.session_state.auth:
        render_auth()
        return

    username = st.session_state.auth["user"]
    if "messages" not in st.session_state:
        reset_session()

    (mode, show_reason, do_synthesis, files, use_memory, use_session_context, auto_mem,
     use_attachment_search, attachment_top_k, embedding_model_name, use_reranker, reranker_model_name) = sidebar(username)

    render_history()

    prompt = st.chat_input("Escreve o teu pedido...")
    if not prompt:
        return

    # Anexos
    attach_text, attach_names, attach_meta = "", [], {"mode": "empty"}
    if files:
        with st.spinner("A extrair texto dos ficheiros..."):
            attach_text, attach_names, warns, attach_meta = extract_files(
                files,
                query=prompt,
                use_attachment_search=use_attachment_search,
                attachment_top_k=attachment_top_k,
                embedding_model_name=embedding_model_name,
                use_reranker=use_reranker,
                reranker_model_name=reranker_model_name,
            )
        for w in warns:
            st.warning(w)
        if attach_meta.get("mode") in ("embedding_rag", "retrieval"):
            st.info(
                f"RAG dos anexos: {attach_meta.get('total_chunks', 0)} chunks avaliados; "
                f"{len(attach_meta.get('selected', []))} enviados ao modelo."
            )
    elif use_attachment_search and st.session_state.get("rag_docs"):
        with st.spinner("A recuperar trechos relevantes do índice RAG da sessão..."):
            try:
                attach_text, attach_meta = retrieve_session_rag_context(
                    prompt,
                    attachment_top_k=attachment_top_k,
                    embedding_model_name=embedding_model_name,
                    use_reranker=use_reranker,
                    reranker_model_name=reranker_model_name,
                )
                attach_names = [d.get("name", "documento") for d in st.session_state.rag_docs]
                if attach_meta.get("mode") == "embedding_rag":
                    st.info(
                        f"RAG dos anexos: {attach_meta.get('total_chunks', 0)} chunks avaliados; "
                        f"{len(attach_meta.get('selected', []))} enviados ao modelo."
                    )
            except Exception as e:
                st.warning(f"Falha no RAG de sessão: {e}")
                attach_text, attach_meta = "", {"mode": "error", "error": str(e)}

    display_text = prompt
    send_text = build_agent_message(prompt, username, use_memory, attach_text, use_session_context)

    st.session_state.messages.append({"role": "user", "content": prompt, "attachments": attach_names})
    with st.chat_message("user"):
        st.markdown(prompt)
        if attach_names:
            st.markdown(attach_note(attach_names), unsafe_allow_html=True)

    if mode == "Auto (Roteador)":
        handle_single(send_text, display_text, None, show_reason)
    elif mode == "So Claude":
        handle_single(send_text, display_text, "claude", show_reason)
    elif mode == "So GPT":
        handle_single(send_text, display_text, "gpt", show_reason)
    elif mode == "Duelo (lado a lado)":
        handle_duel(send_text, display_text)
    elif mode == "Colaboracao":
        handle_collab(send_text, display_text, do_synthesis)

    # Memoria: log + atualizacao automatica
    append_log(username, display_text, mode)
    mem = load_memory(username)
    mem["turns_since"] = mem.get("turns_since", 0) + 1
    save_memory(username, mem)
    if auto_mem and mem["turns_since"] >= MEMORY_UPDATE_EVERY:
        with st.spinner("A atualizar memoria..."):
            summarize_memory(username)

    # Limpa o uploader para o proximo turno
    if attach_names:
        st.session_state.uploader_key += 1
        st.rerun()


if __name__ == "__main__":
    main()
