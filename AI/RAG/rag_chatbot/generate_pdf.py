"""
generate_pdf.py
────────────────
Generates a professionally formatted PDF of the Project Problem Statement
for the AI-Powered Multi-Document Chatbot (RAG System) project.

Uses ReportLab to produce a print-ready A4 document with:
    - SevenMentor header with institution name and document metadata
    - Styled section titles, body text, tables, and code blocks
    - Colour-coded table rows and source citations
    - Footer with page numbers

Output:
    Project_Problem_Statement_RAG_Chatbot.pdf  (in the project root)

Usage:
    python generate_pdf.py

Requirements:
    pip install reportlab
"""

import sys
from pathlib import Path
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

# ── Colour palette ────────────────────────────────────────────────────────────
BRAND_BLUE    = colors.HexColor("#1a73e8")
BRAND_DARK    = colors.HexColor("#202124")
BRAND_GREY    = colors.HexColor("#5f6368")
BRAND_LIGHT   = colors.HexColor("#f8f9fa")
BRAND_ALT_ROW = colors.HexColor("#e8f0fe")
BRAND_GREEN   = colors.HexColor("#34a853")
WHITE         = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "Project_Problem_Statement_RAG_Chatbot.pdf"


# ── Styles ────────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "cover_institution": ParagraphStyle(
            "cover_institution",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=BRAND_BLUE,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            fontName="Helvetica",
            fontSize=13,
            textColor=BRAND_GREY,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Helvetica-Bold",
            fontSize=17,
            textColor=BRAND_DARK,
            alignment=TA_CENTER,
            spaceAfter=8,
            leading=22,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            fontName="Helvetica",
            fontSize=10,
            textColor=BRAND_GREY,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=BRAND_BLUE,
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=BRAND_DARK,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=BRAND_DARK,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=10,
            textColor=BRAND_DARK,
            leading=14,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Courier",
            fontSize=9,
            textColor=BRAND_DARK,
            backColor=BRAND_LIGHT,
            leading=13,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="Helvetica",
            fontSize=9,
            textColor=BRAND_DARK,
            leading=13,
        ),
        "table_cell_bold": ParagraphStyle(
            "table_cell_bold",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=BRAND_DARK,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=BRAND_GREY,
            alignment=TA_CENTER,
        ),
    }
    return styles


# ── Header / Footer callbacks ─────────────────────────────────────────────────

def on_page(canvas, doc):
    canvas.saveState()
    page_num = doc.page

    # Top rule
    canvas.setStrokeColor(BRAND_BLUE)
    canvas.setLineWidth(1.5)
    canvas.line(MARGIN, PAGE_H - 1.3 * cm, PAGE_W - MARGIN, PAGE_H - 1.3 * cm)

    if page_num > 1:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(BRAND_GREY)
        canvas.drawString(MARGIN, PAGE_H - 1.1 * cm, "SevenMentor — RAG Chatbot Project")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.1 * cm, "Project Problem Statement")

    # Bottom rule + page number
    canvas.setStrokeColor(BRAND_BLUE)
    canvas.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(BRAND_GREY)
    canvas.drawCentredString(PAGE_W / 2, 0.9 * cm, f"Page {page_num}")
    canvas.restoreState()


# ── Table helpers ─────────────────────────────────────────────────────────────

def make_table(data, col_widths, styles_map, has_header=True):
    tbl = Table(data, colWidths=col_widths, repeatRows=1 if has_header else 0)
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",  (0, 0), (-1, 0), BRAND_BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        # Alternating rows
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BRAND_ALT_ROW]),
        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#dadce0")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return tbl


# ── Content builders ──────────────────────────────────────────────────────────

def build_cover(styles):
    s = styles
    usable_w = PAGE_W - 2 * MARGIN
    today = date.today().strftime("%B %Y")

    elements = [
        Spacer(1, 2.5 * cm),
        Paragraph("SevenMentor", s["cover_institution"]),
        Paragraph("Artificial Intelligence & Machine Learning Programme", s["cover_subtitle"]),
        Spacer(1, 1 * cm),
        HRFlowable(width=usable_w, thickness=2, color=BRAND_BLUE, spaceAfter=20),
        Paragraph("Project Problem Statement", s["cover_subtitle"]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "AI-Powered Multi-Document Chatbot<br/>"
            "A Retrieval-Augmented Generation (RAG) System<br/>"
            "for Enterprise Document Intelligence",
            s["cover_title"],
        ),
        HRFlowable(width=usable_w, thickness=2, color=BRAND_BLUE, spaceBefore=20, spaceAfter=20),
        Spacer(1, 0.5 * cm),
        Paragraph(f"Document Version: 1.0", s["cover_meta"]),
        Paragraph(f"Date: {today}", s["cover_meta"]),
        Paragraph("Document Type: Formal Capstone Project Statement", s["cover_meta"]),
        Paragraph("Classification: Academic — For Student Use", s["cover_meta"]),
    ]
    return elements


def build_section1(styles):
    s = styles
    usable_w = PAGE_W - 2 * MARGIN

    elements = [
        Paragraph("1. Project Title &amp; Overview", s["h1"]),
        HRFlowable(width=usable_w, thickness=1, color=BRAND_BLUE, spaceAfter=8),

        Paragraph("<b>Project Title</b>", s["h2"]),
        Paragraph(
            "AI-Powered Multi-Document Chatbot: A Retrieval-Augmented Generation (RAG) "
            "System for Enterprise Document Intelligence",
            s["body"],
        ),

        Paragraph("<b>Overview</b>", s["h2"]),
        Paragraph(
            "Organizations today manage large volumes of unstructured documents—reports, "
            "manuals, policies, and knowledge bases—yet employees and stakeholders struggle "
            "to find accurate, contextual answers quickly. Traditional search returns keyword "
            "matches; generic chatbots lack access to proprietary content. This project "
            "addresses that gap by building a <b>Retrieval-Augmented Generation (RAG)</b> "
            "system that allows users to upload multiple documents, store them securely on "
            "AWS S3, and converse with the content using natural language. The system combines "
            "document ingestion, intelligent chunking, vector embeddings, and a large language "
            "model (LLM) to deliver precise, source-grounded answers.",
            s["body"],
        ),
        Paragraph(
            "The solution is designed for scalability and real-world deployment. Students will "
            "implement file upload and storage on AWS S3, process text using NLP techniques "
            "(tokenization, chunking), generate embeddings and store them in a vector store "
            "(FAISS), and integrate Llama 2 via AWS SageMaker JumpStart. A web interface "
            "built with Streamlit provides a simple, professional frontend for uploading "
            "files and chatting with the content. By the end of the project, students will "
            "have built an end-to-end AI application that aligns with industry practices in "
            "data engineering, NLP, deep learning, and MLOps.",
            s["body"],
        ),
    ]
    return elements


def build_section2(styles):
    s = styles
    usable_w = PAGE_W - 2 * MARGIN
    col_w = [3.2 * cm, 3.5 * cm, usable_w - 6.7 * cm]

    header = [
        Paragraph("Syllabus Section", s["table_header"]),
        Paragraph("Topic", s["table_header"]),
        Paragraph("Project Tasks &amp; Mapping", s["table_header"]),
    ]
    rows = [
        [
            Paragraph("Section 4", s["table_cell_bold"]),
            Paragraph("Data Collection", s["table_cell_bold"]),
            Paragraph(
                "<b>File uploads and S3 storage:</b> Design and implement a secure file "
                "upload mechanism (PDF, TXT, DOCX). Use Boto3 to store uploaded files in "
                "AWS S3 with bucket policies, naming conventions, and versioning. Validate "
                "file types and sizes; maintain metadata (filename, timestamp, session ID).",
                s["table_cell"],
            ),
        ],
        [
            Paragraph("Section 6 &amp; 7", s["table_cell_bold"]),
            Paragraph("Deep Learning &amp; Transformers", s["table_cell_bold"]),
            Paragraph(
                "<b>LLM integration via SageMaker:</b> Deploy and invoke Llama 2 7B Chat "
                "from AWS JumpStart on a SageMaker endpoint. Build and format Llama 2 chat "
                "prompts. Understand context window limits, temperature, and sampling "
                "parameters. Parse structured JSON responses from the model.",
                s["table_cell"],
            ),
        ],
        [
            Paragraph("Section 7", s["table_cell_bold"]),
            Paragraph("NLP &amp; Text Preprocessing", s["table_cell_bold"]),
            Paragraph(
                "<b>Tokenization, chunking, and embeddings:</b> Extract text from uploaded "
                "documents. Apply RecursiveCharacterTextSplitter (LangChain) with chunk_size=500 "
                "and chunk_overlap=50. Generate L2-normalized vector embeddings using "
                "SentenceTransformers (all-MiniLM-L6-v2) and store them in a FAISS index.",
                s["table_cell"],
            ),
        ],
        [
            Paragraph("Section 8", s["table_cell_bold"]),
            Paragraph("Deployment &amp; MLOps", s["table_cell_bold"]),
            Paragraph(
                "<b>Hosting and frontend:</b> Deploy Llama 2 to SageMaker (ml.g5.2xlarge). "
                "Build a Streamlit web app with file upload sidebar and chat UI. Implement "
                "MLOps practices: structured logging, error handling, session management, "
                "and S3-backed index persistence for stateless restarts.",
                s["table_cell"],
            ),
        ],
    ]

    data = [header] + rows
    elements = [
        Paragraph("2. Alignment with Course Syllabus", s["h1"]),
        HRFlowable(width=usable_w, thickness=1, color=BRAND_BLUE, spaceAfter=8),
        Paragraph(
            "Every major deliverable in this project is explicitly mapped to a section of "
            "the SevenMentor AI/ML curriculum, ensuring that practical engineering tasks "
            "reinforce classroom concepts.",
            s["body"],
        ),
        make_table(data, col_w, s),
    ]
    return elements


def build_section3(styles):
    s = styles
    usable_w = PAGE_W - 2 * MARGIN
    col_w = [3.0 * cm, 4.0 * cm, usable_w - 7.0 * cm]

    header = [
        Paragraph("Category", s["table_header"]),
        Paragraph("Technology", s["table_header"]),
        Paragraph("Purpose", s["table_header"]),
    ]
    rows = [
        ["Language",         "Python 3.x",                     "Core development, scripting, and API logic"],
        ["Cloud &amp; ML",   "AWS SageMaker",                  "Llama 2 hosting/invocation via JumpStart"],
        ["Storage",          "AWS S3",                         "Secure storage for uploaded documents"],
        ["SDK",              "Boto3",                          "AWS API calls (S3, SageMaker Runtime)"],
        ["Vector Store",     "FAISS (faiss-cpu)",              "Store and search document chunk embeddings"],
        ["Framework",        "LangChain",                      "Text splitting (RecursiveCharacterTextSplitter)"],
        ["Embeddings",       "SentenceTransformers",           "all-MiniLM-L6-v2: 384-dim semantic vectors"],
        ["Frontend",         "Streamlit",                      "Web UI: file upload sidebar + chat interface"],
        ["Doc Parsing",      "PyPDF2, python-docx",            "Extract text from PDF and DOCX files"],
        ["Config",           "python-dotenv",                  "Load .env credentials without hardcoding"],
        ["PDF Generation",   "ReportLab",                      "Generate this Problem Statement PDF"],
    ]

    data = [header] + [
        [Paragraph(r[0], s["table_cell_bold"]),
         Paragraph(r[1], s["table_cell_bold"]),
         Paragraph(r[2], s["table_cell"])]
        for r in rows
    ]

    elements = [
        Paragraph("3. Tools &amp; Technologies", s["h1"]),
        HRFlowable(width=usable_w, thickness=1, color=BRAND_BLUE, spaceAfter=8),
        make_table(data, col_w, s),
    ]
    return elements


def build_section4(styles):
    s = styles
    usable_w = PAGE_W - 2 * MARGIN

    pipeline_steps = [
        ("1. Ingestion", "User uploads PDF/DOCX/TXT → validated → stored in AWS S3 under raw/<session_id>/<filename>."),
        ("2. Preprocessing", "Text is extracted (PyPDF2 / python-docx) → cleaned (whitespace, null bytes) → split into overlapping chunks (size=500, overlap=50)."),
        ("3. Embedding", "Each chunk is encoded into a 384-dimension vector using SentenceTransformers all-MiniLM-L6-v2 with L2 normalization."),
        ("4. Indexing", "All chunk vectors are stored in a FAISS IndexFlatIP. The index + metadata JSON are persisted to S3 for stateless restarts."),
        ("5. Retrieval", "User asks a question → query is embedded → FAISS returns top-5 nearest chunks → chunks carry source + page metadata."),
        ("6. Generation", "Retrieved chunks + user query → formatted Llama 2 chat prompt → sent to SageMaker endpoint → model generates a grounded answer."),
        ("7. Deployment", "Streamlit app runs on any host. SageMaker endpoint handles the LLM. S3 stores all documents and indexes."),
    ]

    why_items = [
        ("<b>S3 Ingestion:</b>", "Centralized, durable, and cost-effective storage. Integrates natively with SageMaker and decouples storage from compute."),
        ("<b>Chunking:</b>", "LLMs have a context window limit (~4096 tokens for Llama 2). Smaller focused chunks improve retrieval precision and avoid truncation."),
        ("<b>Embeddings:</b>", "Dense vectors capture semantic meaning, enabling 'refund policy' to match 'how to return a product' even with different words."),
        ("<b>FAISS:</b>", "Sub-millisecond similarity search over millions of vectors — far faster than brute-force comparison loops."),
        ("<b>RAG vs. fine-tuning:</b>", "RAG grounds the LLM in your specific documents without retraining, reduces hallucination, and allows real-time updates by re-indexing."),
        ("<b>Streamlit:</b>", "Minimal Python frontend that lets students focus on the AI pipeline rather than web development complexity."),
    ]

    elements = [
        Paragraph("4. The Pipeline: How It Works and Why", s["h1"]),
        HRFlowable(width=usable_w, thickness=1, color=BRAND_BLUE, spaceAfter=8),

        Paragraph("4.1 Pipeline Steps", s["h2"]),
    ]

    for title, description in pipeline_steps:
        elements.append(
            Paragraph(f"<b>{title}:</b> {description}", s["bullet"])
        )

    elements += [
        Spacer(1, 0.4 * cm),
        Paragraph("4.2 Conceptual Flow", s["h2"]),
        Paragraph(
            "User → Upload Files → [S3]  →  Extract → Clean → Chunk  →  "
            "Embed → [FAISS Index]",
            s["code"],
        ),
        Paragraph(
            "User → Ask Question → Embed Query → Retrieve Top-K Chunks  →  "
            "[Llama 2 on SageMaker]  →  Answer → User",
            s["code"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("4.3 Design Decisions Explained", s["h2"]),
    ]

    for label, reason in why_items:
        elements.append(Paragraph(f"{label} {reason}", s["bullet"]))

    return elements


def build_section5(styles):
    s = styles
    usable_w = PAGE_W - 2 * MARGIN

    objectives = [
        "Design and implement a document ingestion pipeline using AWS S3 and Boto3, including file type validation and metadata tagging.",
        "Apply NLP and text preprocessing techniques (tokenization, chunking with overlap) to prepare documents for embedding and retrieval.",
        "Build a semantic retrieval system using SentenceTransformer embeddings and a FAISS vector store to find relevant document chunks.",
        "Integrate a large language model (Llama 2 via AWS SageMaker JumpStart) and implement a complete RAG workflow that combines retrieval and generation.",
        "Develop a production-style web application using Streamlit with file upload, multi-turn chat, and collapsible source citations.",
        "Deploy and operate the system on AWS, applying MLOps practices: structured logging, error handling, session management, and S3 index persistence.",
        "Articulate the full RAG pipeline to a technical audience — explaining the role of each component (ingestion, chunking, embedding, retrieval, generation) and the trade-offs involved.",
    ]

    elements = [
        Paragraph("5. Learning Objectives", s["h1"]),
        HRFlowable(width=usable_w, thickness=1, color=BRAND_BLUE, spaceAfter=8),
        Paragraph(
            "Upon successful completion of this project, students will be able to:",
            s["body"],
        ),
    ]

    for i, obj in enumerate(objectives, start=1):
        elements.append(Paragraph(f"{i}.  {obj}", s["bullet"]))

    elements += [
        Spacer(1, 0.6 * cm),
        HRFlowable(width=usable_w, thickness=1, color=BRAND_BLUE, spaceAfter=8),
        Paragraph(
            "This document is intended for academic use at SevenMentor. "
            "All code and infrastructure described herein is provided as a teaching resource. "
            "Students are expected to implement, test, and extend the system independently.",
            s["body"],
        ),
    ]
    return elements


# ── PDF Assembly ──────────────────────────────────────────────────────────────

def generate_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    styles = build_styles()

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
    )

    frame = Frame(
        MARGIN, MARGIN,
        PAGE_W - 2 * MARGIN, PAGE_H - 4.4 * cm,
        id="main",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=on_page)])

    story = []

    # Cover
    story.extend(build_cover(styles))
    story.append(Spacer(1, 1.5 * cm))

    # Sections
    for builder in [
        build_section1,
        build_section2,
        build_section3,
        build_section4,
        build_section5,
    ]:
        story.append(Spacer(1, 0.3 * cm))
        story.extend(builder(styles))

    doc.build(story)
    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    output = generate_pdf()
    print(f"Done. Open: {output}")
