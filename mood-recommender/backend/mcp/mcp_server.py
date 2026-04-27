"""
mcp_server.py
─────────────
Exposes the Mood-Adaptive Recommender pipeline as an MCP-compatible tool.

The MCP (Model-Callable Protocol) server wraps the /recommend endpoint so that
other AI agents (LangChain agents, AutoGPT, etc.) can discover and invoke the
recommendation tool programmatically.

Tool definition:
    Name  : get_movie_recommendations
    Input : { "input": str, "session_id": str (optional) }
    Output: structured recommendation response (same as /recommend)

Usage:
    python -m backend.mcp.mcp_server   # runs standalone HTTP server on :8001
    — OR import MCPServer and mount it on the main FastAPI app.
"""

import logging
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Default backend URL ───────────────────────────────────────────────────────
_BACKEND_URL = "http://localhost:8000"


# ══════════════════════════════════════════════════════════════════════════════
# MCP Tool schemas
# ══════════════════════════════════════════════════════════════════════════════

class MCPToolInput(BaseModel):
    """Input schema for the get_movie_recommendations MCP tool."""

    input: str = Field(
        ...,
        description="User mood or free-text input. Any form accepted.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID to maintain context across calls.",
    )


class MCPToolManifest(BaseModel):
    """MCP tool manifest returned from /mcp/tools."""

    name: str
    description: str
    input_schema: dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI MCP app
# ══════════════════════════════════════════════════════════════════════════════

mcp_app = FastAPI(
    title="Mood Recommender MCP Server",
    description="MCP-compatible wrapper exposing the recommendation pipeline as a callable tool.",
    version="1.0.0",
)

# ── Tool manifest endpoint ────────────────────────────────────────────────────

@mcp_app.get("/mcp/tools", summary="List available MCP tools")
async def list_tools() -> list[MCPToolManifest]:
    """
    Return the manifest of all available MCP tools.

    Returns:
        list[MCPToolManifest]: Tool descriptions with input schemas.
    """
    return [
        MCPToolManifest(
            name="get_movie_recommendations",
            description=(
                "Interprets any mood or free-text input and returns personalized "
                "movie/show recommendations grounded in semantic vector search. "
                "Accepts vague, emotional, emoji, or natural language input."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "User mood or free-text input.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID for context continuity.",
                        "nullable": True,
                    },
                },
                "required": ["input"],
            },
        )
    ]


# ── Tool invocation endpoint ──────────────────────────────────────────────────

@mcp_app.post("/mcp/tools/get_movie_recommendations", summary="Invoke recommendation tool")
async def invoke_recommendation_tool(tool_input: MCPToolInput) -> dict[str, Any]:
    """
    Invoke the get_movie_recommendations tool.

    Proxies the request to the main backend's /recommend endpoint.

    Args:
        tool_input (MCPToolInput): Tool input with user mood text.

    Returns:
        dict[str, Any]: Recommendation response from the backend pipeline.

    Raises:
        HTTPException 502: If the backend is unreachable.
        HTTPException 500: If the backend returns an error.
    """
    try:
        resp = requests.post(
            f"{_BACKEND_URL}/recommend",
            json={
                "input": tool_input.input,
                "session_id": tool_input.session_id,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    except requests.ConnectionError as exc:
        logger.error("Backend unreachable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Backend recommender service is unreachable.",
        ) from exc

    except requests.HTTPError as exc:
        logger.error("Backend returned error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backend error: {exc.response.text}",
        ) from exc


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(mcp_app, host="0.0.0.0", port=8001, reload=False)
