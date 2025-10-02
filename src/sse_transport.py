"""SSE transport layer for MCP server using Starlette."""

import logging
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


def create_sse_server(mcp: FastMCP) -> Starlette:
    """
    Create a Starlette app that handles SSE connections for MCP.
    
    Args:
        mcp: FastMCP server instance
        
    Returns:
        Starlette application configured for SSE transport
    """
    # Create SSE transport
    transport = SseServerTransport("/messages/")
    
    # Define SSE connection handler
    async def handle_sse(request: Request):
        """Handle incoming SSE connections."""
        async with transport.connect_sse(
            request.scope,
            request.receive,
            request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0],
                streams[1],
                mcp._mcp_server.create_initialization_options()
            )
        return Response()
    
    # Create routes - use Mount for the message handler
    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/messages", app=transport.handle_post_message),
    ]
    
    # Create Starlette application
    app = Starlette(routes=routes)
    logger.info("SSE transport layer created")
    
    return app
