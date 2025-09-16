import asyncio
import json
from typing import Optional, List, Dict, Any
import oracledb
from mcp import types
from mcp.server import Server
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.requests import Request
from starlette.routing import Route
import uvicorn

# ---- Oracle Config ----
ORACLE_USER = "your_user"
ORACLE_PASSWORD = "your_password"
ORACLE_DSN = "host:port/service_name"  # e.g. "127.0.0.1:1521/XEPDB1"

# Create MCP server
server = Server("oracle-db")


# ----------------------
# Tool Functions
# ----------------------
async def run_query(sql: str) -> List[Dict[str, Any]]:
    """
    Execute a SQL query against Oracle DB.
    Returns results as a list of dictionaries.
    """
    try:
        with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                cols = [c[0] for c in cursor.description]
                rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
                return rows
    except Exception as e:
        return [{"error": str(e)}]


async def get_schema(owner: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Retrieve schema information: tables, columns, and data types.
    Optionally filter by schema/owner.
    """
    try:
        with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN) as conn:
            with conn.cursor() as cursor:
                sql = """
                    SELECT owner, table_name, column_name, data_type
                    FROM all_tab_columns
                """
                if owner:
                    sql += " WHERE owner = :owner"
                    cursor.execute(sql, {"owner": owner.upper()})
                else:
                    cursor.execute(sql)

                rows = [
                    {"owner": row[0], "table": row[1], "column": row[2], "type": row[3]}
                    for row in cursor.fetchall()
                ]
                return rows
    except Exception as e:
        return [{"error": str(e)}]


# ----------------------
# MCP Server Handlers
# ----------------------
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="run_query",
            description="Execute a SQL query against Oracle DB",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["sql"]
            }
        ),
        types.Tool(
            name="get_schema",
            description="Retrieve schema information: tables, columns, and data types",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Schema owner (optional)"}
                }
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    if name == "run_query":
        sql = arguments.get("sql", "")
        result = await run_query(sql)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "get_schema":
        owner = arguments.get("owner")
        result = await get_schema(owner)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


# ----------------------
# SSE Endpoint
# ----------------------
async def sse_endpoint(request: Request):
    """SSE endpoint for MCP communication"""
    
    async def event_stream():
        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"
        
        # Keep connection alive
        try:
            while True:
                await asyncio.sleep(1)
                yield f"data: {json.dumps({'type': 'ping', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
        except asyncio.CancelledError:
            break
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


# ----------------------
# HTTP POST endpoint for MCP messages
# ----------------------
async def handle_mcp_request(request: Request):
    """Handle MCP JSON-RPC requests via HTTP"""
    try:
        # Use the MCP server to handle the request
        from mcp.server.session import ServerSession
        from mcp.shared.session import RequestHandlers
        
        # Create a simple transport for HTTP
        session = ServerSession(server, RequestHandlers())
        
        # Get the JSON-RPC message
        message = await request.json()
        
        # Process through MCP server
        response = await session.handle_message(message)
        
        return response
        
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32603, "message": str(e)}
        }


# ----------------------
# Run MCP Server (HTTP/SSE)
# ----------------------
async def main() -> None:
    app = Starlette(
        routes=[
            Route("/sse", endpoint=sse_endpoint, methods=["GET"]),
            Route("/mcp", endpoint=handle_mcp_request, methods=["POST"]),
        ]
    )
    
    config = uvicorn.Config(app, host="0.0.0.0", port=5300, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


if __name__ == "__main__":
    asyncio.run(main())