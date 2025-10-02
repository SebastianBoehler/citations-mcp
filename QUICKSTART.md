

# Quick Start Guide

Get your Research Citations MCP Server running in 5 minutes!

## Step 1: Install Dependencies

```bash
# Clone the repository
cd citations-mcp

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and set your values
nano .env
```

**Required settings:**
- `OPENAI_API_KEY`: Your OpenAI API key
- `PAPERS_DIRECTORY`: Full path to your folder with PDF research papers

Example `.env`:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
PAPERS_DIRECTORY=/Users/you/Documents/research/papers
```

## Step 3: Prepare Your Papers

Place your PDF research papers in the directory you specified:

```bash
mkdir -p ~/Documents/research/papers
# Copy your PDFs to this folder
```

## Step 4: Build the Index

```bash
# Using the CLI
uv run python cli.py build-index

# Or force rebuild
uv run python cli.py build-index --force
```

This will:
- Scan all PDFs in your papers directory
- Extract text from each page
- Create text chunks
- Generate embeddings
- Store in ChromaDB vector database

## Step 5: Test It Works

```bash
# List indexed papers
uv run python cli.py list-papers

# Search for a topic
uv run python cli.py search "machine learning"

# Find citations
uv run python cli.py cite "transformer architecture"

# Ask a question
uv run python cli.py ask "What are the main challenges in NLP?"
```

## Step 6: Start the MCP Server

```bash
# Start the server
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

The server will be available at:
- API: `http://127.0.0.1:8000`
- Health check: `http://127.0.0.1:8000/health`
- MCP SSE endpoint: `http://127.0.0.1:8000/mcp/sse`

## Step 7: Connect from Claude Desktop

Edit your Claude config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "research-citations": {
      "url": "http://127.0.0.1:8000/mcp/sse"
    }
  }
}
```

Restart Claude Desktop and your research tools will be available!

## Common Commands

```bash
# Check server status
uv run python cli.py status

# Rebuild index after adding new papers
uv run python cli.py build-index --force

# Search for specific content
uv run python cli.py search "your query here" -k 10

# Start server in production mode
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

## Troubleshooting

**"No PDF files found"**
- Check `PAPERS_DIRECTORY` path is correct
- Ensure PDFs are in the directory
- Check file permissions

**"OpenAI API error"**
- Verify `OPENAI_API_KEY` is valid
- Check you have API credits
- Ensure key has access to embeddings

**"Vector store not found"**
- Run `uv run python cli.py build-index` first
- Check `VECTOR_DB_PATH` in .env

**Server won't start**
- Check port 8000 is not already in use
- Verify all dependencies are installed
- Check server logs for errors

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore all available MCP tools
- Customize chunk size and embeddings models
- Add more research papers and rebuild the index

Happy researching! 🎓
