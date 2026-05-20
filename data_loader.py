import bisect
import logging
import uuid
import fitz  # pymupdf
import ollama
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
PARENT_GROUP_SIZE = 4  # semantic chunks grouped into one parent

logger = logging.getLogger(__name__)

_embed_model = SentenceTransformer(EMBED_MODEL)
_llama_embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
_bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def _describe_image(image_bytes: bytes, ext: str) -> str:
    """Send image bytes to llava via ollama and return a concise description."""
    logger.debug("Describing image (ext=%s, size=%d bytes)", ext, len(image_bytes))
    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": (
                "Describe this image concisely. Focus on charts, diagrams, tables, "
                "or meaningful visual data. If purely decorative, respond with exactly: decorative"
            ),
            "images": [image_bytes],
        }],
    )
    desc = response["message"]["content"].strip()
    logger.debug("Image description: %s", desc[:100])
    return desc


def extract_image_descriptions(path: str) -> list[str]:
    """Extract and describe non-decorative images from a PDF using llava; skip images under 100x100px."""
    logger.info("Extracting image descriptions from %s", path)
    doc = fitz.open(path)
    descriptions = []
    seen_xrefs: set[int] = set()

    for page_num, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_image = doc.extract_image(xref)
            w, h = base_image.get("width", 0), base_image.get("height", 0)
            if w < 100 or h < 100:
                logger.debug("Skipping small image xref=%d (%dx%d) on page %d", xref, w, h, page_num + 1)
                continue

            try:
                desc = _describe_image(base_image["image"], base_image.get("ext", "png"))
                if "decorative" not in desc.lower():
                    descriptions.append(f"[Image on page {page_num + 1}]: {desc}")
                else:
                    logger.debug("Skipping decorative image xref=%d on page %d", xref, page_num + 1)
            except Exception as e:
                logger.warning("Failed to describe image xref=%d on page %d: %s", xref, page_num + 1, e)

    doc.close()
    logger.info("Extracted %d image descriptions from %s", len(descriptions), path)
    return descriptions


def _find_page(text: str, full_text: str, page_starts: list[int], page_nums: list[int]) -> int:
    """Map a text snippet back to its 1-based page number using precomputed character offsets."""
    sample = text[:80].strip()
    if not sample or not page_starts:
        return 1
    pos = full_text.find(sample)
    if pos == -1:
        return page_nums[0]
    idx = bisect.bisect_right(page_starts, pos) - 1
    return page_nums[max(0, idx)]


def load_and_chunk_pdf(path: str) -> list[dict]:
    """Load a PDF, extract image descriptions, semantically chunk the text, and return child chunks with parent context."""
    logger.info("Loading PDF: %s", path)
    docs = PDFReader().load_data(file=path)

    page_starts: list[int] = []
    page_nums: list[int] = []
    parts: list[str] = []
    pos = 0
    for i, d in enumerate(docs):
        text = getattr(d, "text", None)
        if not text:
            continue
        page_starts.append(pos)
        page_nums.append(i + 1)
        parts.append(text)
        pos += len(text) + 2  # "\n\n" separator

    full_text = "\n\n".join(parts)
    logger.info("Loaded %d pages from %s", len(parts), path)

    img_descs = extract_image_descriptions(path)
    if img_descs:
        full_text += "\n\n" + "\n\n".join(img_descs)

    splitter = SemanticSplitterNodeParser(
        embed_model=_llama_embed,
        breakpoint_percentile_threshold=95,
    )
    logger.info("Running semantic splitter on %d chars", len(full_text))
    nodes = splitter.get_nodes_from_documents([Document(text=full_text)])
    logger.info("Semantic splitter produced %d nodes", len(nodes))

    chunks = []
    for i in range(0, len(nodes), PARENT_GROUP_SIZE):
        group = nodes[i:i + PARENT_GROUP_SIZE]
        parent_text = "\n\n".join(n.get_content() for n in group)

        for node in group:
            node_text = node.get_content()
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": node_text,
                "parent_text": parent_text,
                "page_num": _find_page(node_text, full_text, page_starts, page_nums),
            })

    logger.info("Produced %d chunks from %s", len(chunks), path)
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts into 384-dim float vectors using all-MiniLM-L6-v2."""
    logger.debug("Embedding %d texts", len(texts))
    return _embed_model.encode(texts, convert_to_numpy=True).tolist()


def bm25_embed_texts(texts: list[str]) -> list[dict]:
    """Return BM25 sparse vectors as list of {indices, values} dicts using Qdrant/bm25."""
    logger.debug("BM25 embedding %d texts", len(texts))
    return [
        {"indices": e.indices.tolist(), "values": e.values.tolist()}
        for e in _bm25_model.embed(texts)
    ]
