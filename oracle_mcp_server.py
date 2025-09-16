import asyncio
import oracledb
from mcp.server.fastmcp import FastMCPServer
from mcp.server.sse import sse_server

# ---- Oracle Config ----
ORACLE_USER = "your_user"
ORACLE_PASSWORD = "your_password"
ORACLE_DSN = "host:port/service_name"  # e.g. "127.0.0.1:1521/XEPDB1"

# Create MCP server
mcp = FastMCPServer("oracle-db")


# ----------------------
# Tool 1: Run SQL Query
# ----------------------
@mcp.tool()
async def run_query(sql: str) -> list[dict]:
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
@mcp.tool()
async def get_schema(owner: str = None) -> list[dict]:
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
# Run MCP Server (HTTP/SSE)
# ----------------------
async def main():
    # Expose via SSE/HTTP (what Copilot requires)
    async with sse_server(mcp, host="0.0.0.0", port=5300, path="/sse"):
        await mcp.serve()


if __name__ == "__main__":
    asyncio.run(main())
