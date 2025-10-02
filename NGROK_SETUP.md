# Ngrok Public Access Setup

This guide shows how to expose your Research Citations MCP Server publicly via ngrok, allowing you to use it with ChatGPT or other remote MCP clients.

## Prerequisites

### 1. Install ngrok

```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/download
```

### 2. Get ngrok Auth Token

1. Sign up at [https://ngrok.com](https://ngrok.com)
2. Go to [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
3. Copy your auth token

### 3. Configure ngrok

```bash
# Option 1: Set environment variable
export NGROK_AUTHTOKEN="your_ngrok_auth_token_here"

# Option 2: Configure ngrok directly
ngrok config add-authtoken your_ngrok_auth_token_here
```

## Usage

### Start the Public Server

```bash
./start_public.sh
```

This will:
1. Start the MCP server on port 8000
2. Create an ngrok tunnel
3. Display the public URL

### Example Output

```
[citations-mcp] starting Research Citations MCP server on port 8000
[citations-mcp] starting ngrok tunnel
[citations-mcp] ✅ Server is running!
[citations-mcp] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[citations-mcp] Public URL: https://abc123.ngrok-free.app
[citations-mcp] MCP SSE Endpoint: https://abc123.ngrok-free.app/mcp/sse
[citations-mcp] Health Check: https://abc123.ngrok-free.app/health
[citations-mcp] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[citations-mcp]
[citations-mcp] Add this to ChatGPT MCP config:
[citations-mcp]   URL: https://abc123.ngrok-free.app/mcp/sse
[citations-mcp]
[citations-mcp] Press Ctrl+C to stop the server
```

## Connect from ChatGPT

### Method 1: Via ChatGPT UI

1. Go to ChatGPT settings
2. Navigate to MCP Servers
3. Add new server:
   - **Type**: Remote SSE
   - **URL**: `https://your-ngrok-url.ngrok-free.app/mcp/sse`

### Method 2: Via Configuration File

If ChatGPT supports config files for remote servers, add:

```json
{
  "mcpServers": {
    "research-citations": {
      "url": "https://your-ngrok-url.ngrok-free.app/mcp/sse",
      "type": "sse"
    }
  }
}
```

## Configuration Options

You can customize the script with environment variables:

```bash
# Change the local port
MCP_SSE_LOCAL_PORT=9000 ./start_public.sh

# Use custom ngrok binary
MCP_SSE_NGROK_BIN=/path/to/ngrok ./start_public.sh

# Set auth token via env var
NGROK_AUTHTOKEN="your_token" ./start_public.sh
```

## Security Considerations

⚠️ **Important Security Notes:**

1. **Authentication**: The current setup does not include authentication. Anyone with the ngrok URL can access your MCP server.

2. **API Keys**: Your OpenAI API key is used server-side, so it's not exposed to clients.

3. **Data Privacy**: Consider what research papers you're exposing - they'll be searchable via the public endpoint.

4. **Rate Limiting**: Consider adding rate limiting for production use.

### Adding Basic Authentication (Recommended)

To add basic auth, you can:

1. **Use ngrok basic auth:**
   ```bash
   ngrok http 8000 --basic-auth="username:password"
   ```

2. **Modify the script:**
   Edit `start_public.sh` and add auth to the ngrok command:
   ```bash
   ngrok_cmd+=(--basic-auth "myuser:mypassword")
   ```

3. **Add FastAPI middleware:**
   Edit `src/main.py` to add authentication middleware.

## Troubleshooting

### ngrok command not found
```bash
# Install ngrok
brew install ngrok
# Or download from ngrok.com
```

### jq command not found
```bash
brew install jq
```

### Tunnel not starting
- Check your ngrok auth token is valid
- Ensure port 8000 is not already in use
- Check ngrok dashboard for account limits

### Can't connect from ChatGPT
- Verify the URL is accessible in a browser: `https://your-url/health`
- Check that `/mcp/sse` endpoint exists
- Look at server logs for errors

## Stopping the Server

Press `Ctrl+C` in the terminal where the script is running. This will:
1. Stop the ngrok tunnel
2. Shutdown the MCP server
3. Clean up temporary files

## Alternative: Local Development

For local testing without ngrok:

```bash
# Start server locally
uv run uvicorn src.main:app --reload

# Access at http://127.0.0.1:8000
```

## Production Deployment

For production, consider:
- Using a proper cloud hosting service (AWS, GCP, Azure)
- Setting up HTTPS with proper certificates
- Implementing authentication and authorization
- Adding rate limiting
- Using a process manager (systemd, supervisord)
- Setting up monitoring and logging

Example production alternatives:
- Deploy to Railway, Render, or Fly.io
- Use Cloudflare Tunnels instead of ngrok
- Deploy as a container with proper reverse proxy
