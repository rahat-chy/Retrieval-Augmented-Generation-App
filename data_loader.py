import fitz  # pymupdf
import ollama
from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_embed_model = SentenceTransformer(EMBED_MODEL)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def _describe_image(image_bytes: bytes, ext: str) -> str:
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
            if w < 100 or h < 100:  # skip icons / decorative thumbnails
                continue

            try:
                desc = _describe_image(base_image["image"], base_image.get("ext", "png"))
                if "decorative" not in desc.lower():
                    descriptions.append(f"[Image on page {page_num + 1}]: {desc}")
            except Exception:
                pass

    doc.close()
    return descriptions


def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)
    full_text = "\n".join(d.text for d in docs if getattr(d, "text", None))
    chunks = splitter.split_text(full_text)

    chunks.extend(extract_image_descriptions(path))

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _embed_model.encode(texts, convert_to_numpy=True).tolist()
