import bisect
import uuid
import fitz  # pymupdf
import ollama
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
PARENT_GROUP_SIZE = 4  # semantic chunks grouped into one parent

_embed_model = SentenceTransformer(EMBED_MODEL)
_llama_embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)


def _describe_image(image_bytes: bytes, ext: str) -> str:
    """Send image bytes to llava via ollama and return a concise description."""
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
    return response["message"]["content"].strip()


def extract_image_descriptions(path: str) -> list[str]:
    """Extract and describe non-decorative images from a PDF using llava; skip images under 100x100px."""
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
                continue

            try:
                desc = _describe_image(base_image["image"], base_image.get("ext", "png"))
                if "decorative" not in desc.lower():
                    descriptions.append(f"[Image on page {page_num + 1}]: {desc}")
            except Exception:
                pass

    doc.close()
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

    img_descs = extract_image_descriptions(path)
    if img_descs:
        full_text += "\n\n" + "\n\n".join(img_descs)

    splitter = SemanticSplitterNodeParser(
        embed_model=_llama_embed,
        breakpoint_percentile_threshold=95,
    )
    nodes = splitter.get_nodes_from_documents([Document(text=full_text)])

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

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts into 384-dim float vectors using all-MiniLM-L6-v2."""
    return _embed_model.encode(texts, convert_to_numpy=True).tolist()
