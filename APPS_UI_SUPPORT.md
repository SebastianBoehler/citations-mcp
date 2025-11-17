# ChatGPT Apps UI Support

Concise checklist for keeping the `find_citation` tool rendering correctly inside ChatGPT.

## 1. Server & Metadata

- Ensure `FastMCP` server overrides `list_tools`, `list_resources`, and `list_resource_templates` to inject `_meta` values mirroring OpenAI Apps SDK examples.
- `WIDGET_META` must include:
  - `openai/outputTemplate`: `ui://widget/find-citations.html`
  - `openai/toolInvocation/invoking`: "Gathering citations"
  - `openai/toolInvocation/invoked`: "Rendered citations"
  - `openai/widgetAccessible`: `true`
  - `openai/resultCanProduceWidget`: `true`
- Register the widget HTML as an MCP resource at `ui://widget/find-citations.html` with `text/html` MIME type.

## 2. Tool Response Contract

- Always return Apps SDK payload structure:
  ```json
  {
    "content": [{"type": "text", "text": "Found N source(s) for '<topic>'."}],
    "structuredContent": {
      "topic": "<topic>",
      "total_sources": <int>,
      "citations": {
        "<filename>": {
          "bibliography": {
            "title": "...",
            "authors": "...",
            "year": "...",
            "apa_citation": "..."
          },
          "relevant_excerpts": [{
            "page": <int>,
            "excerpt": "...",
            "relevance_score": <float>
          }]
        }
      }
    },
    "_meta": WIDGET_META
  }
  ```
- Wrap error cases with the same envelope but place details under `structuredContent` (e.g., `{ "error": "...", "topic": "..." }`).
- Initialize the vector store during app startup so the tool never returns empty/error payloads due to unbuilt embeddings.

## 3. Widget Behavior (`public/find-citations.html`)

- `window.openai.toolOutput` and `openai:set_globals` listeners feed data into `render()`.
- `coerceStructured()` must unwrap both:
  1. Direct `{ structuredContent, ... }` payloads.
  2. Nested `{ result: { structuredContent, ... } }` wrappers (added recursion handles this).
- Empty states trigger when `payload.citations` is missing or empty; successful responses iterate over `Object.entries(payload.citations)` to draw cards.

## 4. Debugging Workflow

- Use console logs baked into the widget (`[citations-widget] ...`) to trace each step:
  - `structuredContent wrapper detected`
  - `result wrapper detected`
  - `updating payload`
  - `rendering entries`
  - `no citations present` / `payload error`
- If the UI shows "no citations" while data exists, confirm ChatGPT delivered either a `result` wrapper or direct `structuredContent` and adjust the coercion logic accordingly (already recursive).
- Ensure browser DevTools console is open when testing inside ChatGPT to capture the logs.

## 5. Tests & Verification

- `tests/test_new_features.py` asserts:
  - `find_citation` tool advertises the widget via annotations/metadata.
  - Widget resource is registered with correct URI.
  - Tool response includes Apps SDK structured payload schema.
- Run the test suite plus a manual ChatGPT session after any changes touching:
  - `src/mcp_server.py`
  - `public/find-citations.html`
  - RAG/vector-store initialization paths.

Following this checklist prevents the "widget renders but stays empty" regressions we previously hit.
