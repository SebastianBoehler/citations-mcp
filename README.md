# Research Citations MCP Server

A Model Context Protocol (MCP) server for searching and citing research papers using Retrieval-Augmented Generation (RAG). This server helps you find relevant citations, search for specific passages, and answer research questions based on your collection of PDF research papers.

## Features

- 🔍 **Semantic Search**: Find relevant passages across all your research papers
- 📚 **Citation Finder**: Get properly formatted citations with source references
- 💡 **Question Answering**: Ask questions and get answers backed by your papers
- 📝 **Paper Summarization**: Generate summaries with custom prompts or pre-defined focus areas
- 🔬 **Methodology Extraction**: Extract structured methodology details from papers
- 📖 **Bibliography Extraction**: Get APA-formatted citations from paper reference sections
- 🏷️ **Automatic Metadata Extraction**: During index rebuild, automatically extracts authors, year, title, and APA citation from first 1-2 pages using GPT-4o
- 📄 **PDF Processing**: Automatic extraction and chunking of text from PDFs
- 🚀 **SSE Transport**: Remote access via Server-Sent Events
- 🎯 **Vector Search**: ChromaDB-powered semantic similarity search
- 🤖 **LangChain Integration**: Built on battle-tested RAG frameworks

## Architecture

```
┌─────────────────────────────────────────────────┐
│           MCP Client (Claude, etc.)             │
└───────────────────┬─────────────────────────────┘
                    │ SSE Transport
┌───────────────────▼─────────────────────────────┐
│              FastAPI + Starlette                │
│                 MCP Server                       │
├─────────────────────────────────────────────────┤
│              RAG Engine                         │
│  ┌─────────────┐        ┌──────────────┐       │
│  │  LangChain  │◄──────►│   ChromaDB   │       │
│  │   Retrieval │        │ Vector Store │       │
│  └─────────────┘        └──────────────┘       │
│         │                       ▲               │
│         ▼                       │               │
│  ┌─────────────┐        ┌──────────────┐       │
│  │ OpenAI LLM  │        │ PDF Processor│       │
│  │ & Embeddings│        │  (PyPDFLoader)│      │
│  └─────────────┘        └──────┬───────┘       │
└────────────────────────────────┼───────────────┘
                                 │
                        ┌────────▼────────┐
                        │  Research Papers │
                        │   (PDF Files)    │
                        └──────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- UV package manager (recommended) or pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd citations-mcp
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and set:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PAPERS_DIRECTORY`: Path to your folder containing PDF research papers
   - Other optional settings (see `.env.example`)

## Usage

### Starting the Server

```bash
# Using uv
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

# Or using uvicorn directly
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

The server will:
1. Start on `http://127.0.0.1:8000`
2. Automatically process all PDFs in your papers directory
3. Build a vector store index (cached for future runs)
4. Expose MCP tools via SSE at `http://127.0.0.1:8000/mcp/sse`

### Available MCP Tools

#### 1. `search_papers`
Search for relevant passages in your research papers.

```python
{
  "query": "machine learning for natural language processing",
  "num_results": 5
}
```

#### 2. `search_in_paper`
Search for relevant passages within a specific paper only.

```python
{
  "query": "transformer architecture",
  "filename": "attention_is_all_you_need.pdf",
  "num_results": 5
}
```

#### 3. `find_citation`
Find relevant citations for a specific topic, grouped by source paper.

```python
{
  "topic": "transformer architecture",
  "num_citations": 3
}
```

#### 4. `answer_question`
Ask a research question and get an answer with sources.

```python
{
  "question": "What are the main challenges in few-shot learning?"
}
```

#### 5. `list_papers`
List all indexed research papers.

```python
{}
```

#### 6. `get_paper_info`
Get information about a specific paper.

```python
{
  "filename": "attention_is_all_you_need.pdf"
}
```

#### 7. `rebuild_index`
Rebuild the vector store index (use after adding new papers). During rebuild, the server automatically:
- Extracts metadata from the first 1-2 pages of each paper using GPT-4o
- Captures authors, publication year, title, journal/conference, DOI, and generates APA citation
- Attaches this metadata to all chunks for easy citation reference

```python
{
  "force": true
}
```

**Note**: Metadata extraction uses GPT-4o API calls (one per paper), so rebuilding with many papers may incur API costs.

#### 8. `extract_methodology`
Extract structured methodology details from a research paper.

```python
{
  "filename": "attention_is_all_you_need.pdf"
}
```

Returns: Research approach, datasets, models, evaluation metrics, experimental setup, baselines, and implementation details.

#### 9. `summarize_paper`
Generate a summary of a research paper with custom prompts or pre-defined focus.

```python
{
  "filename": "attention_is_all_you_need.pdf",
  "focus": "key_findings"  // Options: "general", "key_findings", "methodology", "limitations", "contributions"
}
```

Or with a custom prompt:

```python
{
  "filename": "attention_is_all_you_need.pdf",
  "custom_prompt": "Summarize the experimental results and their statistical significance"
}
```

#### 10. `extract_bibliography`
Extract the bibliography/references from a paper in APA format.

```python
{
  "filename": "attention_is_all_you_need.pdf"
}
```

Returns: APA-formatted list of all citations used in the paper.

### Connecting from Claude Desktop

Add this to your Claude Desktop MCP configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "research-citations": {
      "url": "http://127.0.0.1:8000/mcp/sse"
    }
  }
}
```

Restart Claude Desktop and you'll see the research tools available.

### Public Access via Ngrok (for ChatGPT and Remote Access)

To expose your server publicly via ngrok:

```bash
# Install ngrok (if not already installed)
brew install ngrok

# Configure your ngrok auth token
export NGROK_AUTHTOKEN="your_ngrok_token"

# Start the public server
./start_public.sh
```

This will give you a public URL like `https://abc123.ngrok-free.app/mcp/sse` that you can use with ChatGPT or other remote MCP clients.

See [NGROK_SETUP.md](NGROK_SETUP.md) for detailed instructions.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `PAPERS_DIRECTORY` | Path to research papers folder (required) | - |
| `VECTOR_DB_PATH` | Path to store vector database | `./vector_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `research_papers` |
| `CHUNK_SIZE` | Text chunk size for processing | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `LLM_MODEL` | OpenAI LLM model | `gpt-4o-mini` |
| `HOST` | Server host | `127.0.0.1` |
| `PORT` | Server port | `8000` |

### Adding Papers

1. Place PDF files in your `PAPERS_DIRECTORY`
2. Either:
   - Restart the server (auto-detects new papers)
   - Call the `rebuild_index` tool with `force: true`

## Development

### Project Structure

```
citations-mcp/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── pdf_processor.py    # PDF loading and chunking
│   ├── rag_engine.py       # RAG pipeline and search
│   ├── mcp_server.py       # MCP tool definitions
│   ├── sse_transport.py    # SSE transport layer
│   └── main.py             # FastAPI application
├── pyproject.toml          # Dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

### Running Tests

```bash
# Using uv
uv run pytest

# Or using pytest directly
pytest
```

### Adding New Tools

Edit `src/mcp_server.py` and add a new function decorated with `@mcp.tool()`:

```python
@mcp.tool()
async def my_new_tool(param: str) -> Dict[str, Any]:
    """Tool description."""
    # Implementation
    return {"result": "value"}
```

## Troubleshooting

### Vector store not initializing

- Check that `PAPERS_DIRECTORY` exists and contains PDF files
- Ensure `OPENAI_API_KEY` is valid
- Check logs for specific error messages

### PDFs not being processed

- Verify PDFs are valid and readable
- Check file permissions
- Look for processing errors in server logs

### Poor search results

- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env`
- Try different `EMBEDDING_MODEL` options
- Rebuild index with `force: true`

## Performance Tips

- **First Run**: Initial indexing takes time proportional to number of papers
- **Caching**: Vector store is persisted and reused on subsequent runs
- **Embeddings**: `text-embedding-3-small` is fast and cost-effective
- **Chunks**: Smaller chunks (500-1000) work better for precise citations

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Uses [Model Context Protocol](https://modelcontextprotocol.io/)
- Powered by [OpenAI](https://openai.com/) embeddings and LLMs
- PDF processing via [pypdf](https://pypdf.readthedocs.io/)
- Vector storage with [ChromaDB](https://www.trychroma.com/)
