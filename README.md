# RAG-Based Research Paper & IEEE Generator

This project is a Streamlit application that lets you:

- Upload IEEE-style research papers in PDF format
- Build a Retrieval-Augmented Generation (RAG) index over those papers
- Generate:
  - A unified literature review
  - Research gaps and novel research ideas
  - A full IEEE-style research paper (grounded in the uploaded corpus)
  - A DOCX version of the generated paper
- Browse extracted figures from PDFs
- Ask arbitrary questions over all uploaded papers

The app combines **LangChain + FAISS** for retrieval, **Groq LLMs** for analysis, and **Gemini** for long-form IEEE-style paper generation.

---

## Features

- **RAG-backed literature analysis**
  - Upload multiple IEEE PDFs and build an in-memory FAISS vector store
  - Semantic chunking with overlapping windows for better recall
  - Similarity search powered by `sentence-transformers/all-MiniLM-L6-v2`

- **Groq-powered insights**
  - Unified literature review across all uploaded papers
  - Research gaps and 3+ novel research ideas
  - Free-form Q&A over the indexed corpus

- **Gemini IEEE paper generator**
  - Multi-step prompting to produce a structured IEEE paper:
    - Title & Abstract
    - I. Introduction
    - II. Literature Survey
    - III. Proposed Methodology
    - IV. Expected Results
    - V. Conclusion
    - VI. References (grounded in uploaded papers)

- **DOCX export**
  - Automatically converts the generated paper into a `.docx` file
  - Attempts to infer headings and body text
  - Optionally appends extracted figures in an "Appendix" section

- **Visual figures view**
  - Extracts sufficiently large images from PDFs and displays them in the UI

---

## Tech Stack

- **Frontend / UI**: Streamlit
- **LLMs**:
  - Groq: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`
  - Google Gemini: `gemini-2.5-flash`
- **RAG / Retrieval**:
  - LangChain `RecursiveCharacterTextSplitter`
  - `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`
  - FAISS in-memory vector store
- **PDF Parsing**:
  - `pypdf` (or fallback `PyPDF2`)
  - `Pillow` for image extraction
- **Export**:
  - `python-docx` for DOCX generation
- **Config**:
  - `python-dotenv` for local environment variable loading

---

## Project Structure

- [app.py](app.py) – Main Streamlit application and RAG/LLM orchestration
- [requirements.txt](requirements.txt) – Python dependencies

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>.git
cd paper_generator
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

You need API keys for **Groq** and **Google Gemini**.

Create a `.env` file in the project root (same folder as `app.py`) with:

```env
GROQ_API_KEY="your_groq_key_here"
GEMINI_API_KEY="your_gemini_key_here"
```

Alternatively, if deploying on Streamlit Cloud or another managed platform, you can set:

- `GROQ_API_KEY` and `GEMINI_API_KEY` in Streamlit **Secrets**.

The app resolves keys in this order:

1. `st.secrets["GROQ_API_KEY"]` / `st.secrets["GEMINI_API_KEY"]`
2. Fallback to environment variables: `GROQ_API_KEY`, `GEMINI_API_KEY`

---

## Running the App

From the project root:

```bash
streamlit run app.py
```

This will open the app in your default browser (usually at `http://localhost:8501`).

---

## Usage Guide

### 1. Upload papers & build the RAG index

1. Start the app.
2. Use the **"Upload IEEE Research Papers (PDF)"** widget to upload one or more PDFs.
3. On first upload, the app will:
   - Extract text from each PDF page
   - Chunk text with `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)`
   - Embed chunks with `all-MiniLM-L6-v2`
   - Build an in-memory FAISS vector store stored in `st.session_state.vectorstore`

You’ll see a spinner labeled **"Building RAG index..."** and a success message once indexing is complete.

> Note: The index is held in memory per Streamlit session; it is not persisted across restarts.

### 2. Tabs Overview

The app exposes five main tabs:

#### 📑 Review

- Button: **"Generate Literature Review"**
- Pipeline:
  1. Calls `rag_retrieve` with a fixed query:
     - "Summarize problem statements, methods, datasets, and metrics used across all papers."
  2. Concatenates top-k returned chunks into a context string.
  3. Sends the context and instruction prompt to a Groq LLM via `groq_generate`.
  4. Displays the resulting unified literature review.

#### 💡 Gaps & Ideas

- Button: **"Generate Research Gaps & Novel Ideas"**
- Pipeline:
  1. Retrieval query: "Identify limitations, future work, and unexplored research directions."
  2. Groq LLM is asked to output:
     - A list of research gaps
     - 3 novel research ideas (titles + descriptions)

#### 📄 IEEE Paper

- Text input: **"Select / Paste Research Idea Title"**
- Button: **"🚀 Generate IEEE Paper"**
- Pipeline:
  1. Retrieve **literature context** relevant to the topic.
  2. Retrieve **references context** that includes citation-like content.
  3. Call `generate_ieee_paper(topic, literature_ctx, references_ctx)` which:
     - Configures Gemini with `GEMINI_API_KEY`.
     - Runs three consecutive prompts:
       - Part 1: Title, Abstract, I. Introduction
       - Part 2: II. Literature Survey, III. Proposed Methodology
       - Part 3: IV. Expected Results, V. Conclusion, VI. References
     - Uses only the provided contexts (literature and references) as grounding.
  4. Stores the full paper in `st.session_state.paper`.
  5. Shows a preview and enables **DOCX download** via `create_docx_report`.

#### 📊 Figures

- Extracts images from all uploaded PDFs via `extract_images_from_pdf`:
  - Uses `pypdf` page images API and `Pillow` to filter and normalize images.
  - Filters out images below a minimum width (default 300 px).
- Displays each image with a caption indicating its source page.

#### 🔍 Ask Papers

- Free-form question input: **"Ask a question across all papers"**
- Pipeline:
  1. Uses `rag_retrieve` with the user’s question.
  2. Sends the concatenated retrieved context to Groq via `groq_generate`.
  3. Displays the model’s answer.

---

## RAG Architecture (High-Level)

1. **Document ingestion**
   - PDFs uploaded via Streamlit.
   - `extract_text_from_pdf` reads pages and uses `PdfReader.extract_text()`.
   - All page texts are concatenated per file.

2. **Chunking**
   - `RecursiveCharacterTextSplitter` with:
     - `chunk_size=800`
     - `chunk_overlap=150`
   - Output: short, overlapping text chunks.

3. **Embedding & indexing**
   - `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`
   - `FAISS.from_documents(docs, embeddings)` builds an in-memory FAISS index.
   - The index and documents are stored in Streamlit session state.

4. **Query-time retrieval**
   - A query string is embedded with the same model.
   - FAISS similarity search returns top-k similar chunks.
   - Chunks are concatenated and injected into LLM prompts as grounding.

5. **LLM generation**
   - Groq models for:
     - Literature review
     - Gaps & ideas
     - Q&A
   - Gemini for long-form IEEE papers.

6. **Export**
   - Generated paper text is converted to DOCX via `python-docx`.
   - Headings are inferred from lines starting with `I.`, `II.`, `III.`, `Abstract`, etc.
   - Optional images are added in an "Appendix" section.

---

## Environment & Security Notes

- **API keys** are never hard-coded; they are read from:
  - Streamlit secrets (`st.secrets`), or
  - Environment variables loaded via `python-dotenv`.
- Do **not** commit your `.env` file or API keys to Git.
- PDFs are processed in memory in this implementation; for production, consider:
  - Adding file type/size validation
  - Antivirus scanning
  - Persistent storage with proper access control

---

## Extending / Customizing

Some ideas for extending the project:

- Swap FAISS for a hosted vector database (Pinecone, Qdrant, etc.).
- Add more metadata to chunks (e.g., section headers, page numbers) and surface it in the UI.
- Improve PDF parsing using layout-aware tools for better section and reference detection.
- Introduce hybrid retrieval (BM25 + embeddings) and/or re-ranking.
- Add user accounts and project-level document collections to turn this into a multi-tenant SaaS.

---

## Troubleshooting

- **GROQ_API_KEY missing in .env**
  - Ensure `.env` has `GROQ_API_KEY` set and that you restarted the app.

- **GEMINI_API_KEY missing in .env file.**
  - Add `GEMINI_API_KEY` to `.env` or Streamlit secrets.

- **PDF text extraction is poor**
  - Some PDFs (especially scanned ones) may require OCR.
  - Consider pre-processing PDFs externally or integrating an OCR pipeline.

- **High memory usage**
  - Limit the number/size of uploaded PDFs.
  - Clear session state or restart the app between experiments.

---

## License

Add your preferred license here (e.g., MIT, Apache 2.0) before publishing.
