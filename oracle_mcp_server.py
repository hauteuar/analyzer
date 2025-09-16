import asyncio
import json
from typing import Optional, List, Dict, Any
import oracledb
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
# Tool 1: Run SQL Query
# ----------------------
@server.tool()
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


# ----------------------
# Tool 2: Get Schema Info
# ----------------------
@server.tool()
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
    """Handle MCP JSON-RPC requests"""
    try:
        data = await request.json()
        
        # Simple routing based on method
        method = data.get("method", "")
        params = data.get("params", {})
        
        if method == "tools/list":
            # Return available tools
            result = {
                "tools": [
                    {
                        "name": "run_query",
                        "description": "Execute a SQL query against Oracle DB",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "sql": {"type": "string", "description": "SQL query to execute"}
                            },
                            "required": ["sql"]
                        }
                    },
                    {
                        "name": "get_schema",
                        "description": "Retrieve schema information",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "owner": {"type": "string", "description": "Schema owner (optional)"}
                            }
                        }
                    }
                ]
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            
            if tool_name == "run_query":
                result = await run_query(tool_args.get("sql", ""))
            elif tool_name == "get_schema":
                result = await get_schema(tool_args.get("owner"))
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
        else:
            result = {"error": f"Unknown method: {method}"}
        
        response = {
            "jsonrpc": "2.0",
            "id": data.get("id"),
            "result": result
        }
        
        return {"jsonrpc": "2.0", "id": data.get("id"), "result": result}
        
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": data.get("id", None),
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