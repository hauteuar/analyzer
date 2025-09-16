import asyncio
import json
from typing import Optional, List, Dict, Any
import oracledb
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

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


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources (required by MCP protocol)"""
    return []


@server.list_prompts() 
async def handle_list_prompts() -> list[types.Prompt]:
    """List available prompts (required by MCP protocol)"""
    return []


# ----------------------
# Run MCP Server (stdio)
# ----------------------
async def main() -> None:
    # Run with stdio transport (standard for MCP)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())