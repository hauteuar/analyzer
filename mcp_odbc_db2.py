#!/usr/bin/env python3
"""
DB2 MCP Server (ODBC Version) - A Model Context Protocol server for DB2 database operations.
This server provides tools to query DB2 databases, list tables, and get schemas using ODBC.
"""

import asyncio
import json
import os
from typing import Any, Optional
import pyodbc

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


class DB2MCPServer:
    def __init__(self):
        self.server = Server("db2-mcp-server")
        self.connection = None
        self.conn_string = None
        
        # Register handlers
        self.server.list_tools()(self.list_tools)
        self.server.call_tool()(self.call_tool)
        
    def connect_db2(self) -> bool:
        """Establish connection to DB2 database via ODBC."""
        try:
            # Option 1: Use DSN (Data Source Name)
            dsn = os.getenv("DB2_DSN")
            
            if dsn:
                # Connect using DSN
                db_user = os.getenv("DB2_USER")
                db_pass = os.getenv("DB2_PASSWORD")
                self.conn_string = f"DSN={dsn};UID={db_user};PWD={db_pass}"
            else:
                # Option 2: Use connection string without DSN
                driver = os.getenv("DB2_DRIVER", "{IBM DB2 ODBC DRIVER}")
                db_host = os.getenv("DB2_HOST", "localhost")
                db_port = os.getenv("DB2_PORT", "50000")
                db_name = os.getenv("DB2_DATABASE")
                db_user = os.getenv("DB2_USER")
                db_pass = os.getenv("DB2_PASSWORD")
                
                if not all([db_name, db_user, db_pass]):
                    raise ValueError("Missing required DB2 credentials in environment variables")
                
                # Create ODBC connection string
                self.conn_string = (
                    f"DRIVER={driver};"
                    f"DATABASE={db_name};"
                    f"HOSTNAME={db_host};"
                    f"PORT={db_port};"
                    f"PROTOCOL=TCPIP;"
                    f"UID={db_user};"
                    f"PWD={db_pass};"
                )
            
            # Connect to DB2 via ODBC
            self.connection = pyodbc.connect(self.conn_string)
            return True
            
        except Exception as e:
            print(f"Error connecting to DB2: {str(e)}")
            return False
    
    async def list_tools(self) -> list[Tool]:
        """List available MCP tools."""
        return [
            Tool(
                name="query_db2",
                description="Execute a SQL query against the DB2 database and return results",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="list_tables",
                description="List all tables in the DB2 database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "schema": {
                            "type": "string",
                            "description": "Optional schema name to filter tables"
                        }
                    }
                }
            ),
            Tool(
                name="get_table_schema",
                description="Get the schema (columns, types, etc.) for a specific table",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table"
                        },
                        "schema": {
                            "type": "string",
                            "description": "Optional schema name"
                        }
                    },
                    "required": ["table_name"]
                }
            ),
            Tool(
                name="get_table_info",
                description="Get detailed information about a table including row count and size",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table"
                        },
                        "schema": {
                            "type": "string",
                            "description": "Optional schema name"
                        }
                    },
                    "required": ["table_name"]
                }
            ),
            Tool(
                name="list_schemas",
                description="List all available schemas in the database",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: Any) -> list[TextContent]:
        """Handle tool execution."""
        try:
            # Ensure connection
            if not self.connection:
                if not self.connect_db2():
                    return [TextContent(
                        type="text",
                        text="Failed to connect to DB2. Check environment variables and ODBC configuration."
                    )]
            
            if name == "query_db2":
                result = await self.execute_query(arguments.get("query"))
            elif name == "list_tables":
                result = await self.list_db_tables(arguments.get("schema"))
            elif name == "get_table_schema":
                result = await self.get_schema(
                    arguments.get("table_name"),
                    arguments.get("schema")
                )
            elif name == "get_table_info":
                result = await self.get_info(
                    arguments.get("table_name"),
                    arguments.get("schema")
                )
            elif name == "list_schemas":
                result = await self.list_db_schemas()
            else:
                result = f"Unknown tool: {name}"
            
            return [TextContent(type="text", text=str(result))]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}"
            )]
    
    async def execute_query(self, query: str) -> str:
        """Execute a SQL query and return results."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith("SELECT"):
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                result = {
                    "columns": columns,
                    "rows": [[str(cell) if cell is not None else None for cell in row] for row in rows],
                    "row_count": len(rows)
                }
                return json.dumps(result, indent=2, default=str)
            else:
                self.connection.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"
                
        finally:
            cursor.close()
    
    async def list_db_tables(self, schema: Optional[str] = None) -> str:
        """List all tables in the database."""
        cursor = self.connection.cursor()
        try:
            if schema:
                query = """
                    SELECT TABSCHEMA, TABNAME, TYPE, CARD 
                    FROM SYSCAT.TABLES 
                    WHERE TABSCHEMA = ? AND TYPE = 'T'
                    ORDER BY TABNAME
                """
                cursor.execute(query, (schema.upper(),))
            else:
                query = """
                    SELECT TABSCHEMA, TABNAME, TYPE, CARD 
                    FROM SYSCAT.TABLES 
                    WHERE TYPE = 'T'
                    ORDER BY TABSCHEMA, TABNAME
                """
                cursor.execute(query)
            
            tables = cursor.fetchall()
            result = {
                "tables": [
                    {
                        "schema": row[0].strip() if row[0] else None,
                        "name": row[1].strip() if row[1] else None,
                        "type": row[2].strip() if row[2] else None,
                        "row_count": row[3]
                    }
                    for row in tables
                ]
            }
            return json.dumps(result, indent=2)
            
        finally:
            cursor.close()
    
    async def list_db_schemas(self) -> str:
        """List all schemas in the database."""
        cursor = self.connection.cursor()
        try:
            query = """
                SELECT DISTINCT TABSCHEMA 
                FROM SYSCAT.TABLES 
                WHERE TYPE = 'T'
                ORDER BY TABSCHEMA
            """
            cursor.execute(query)
            
            schemas = cursor.fetchall()
            result = {
                "schemas": [row[0].strip() for row in schemas if row[0]]
            }
            return json.dumps(result, indent=2)
            
        finally:
            cursor.close()
    
    async def get_schema(self, table_name: str, schema: Optional[str] = None) -> str:
        """Get schema information for a table."""
        cursor = self.connection.cursor()
        try:
            if schema:
                query = """
                    SELECT COLNAME, TYPENAME, LENGTH, SCALE, NULLS, DEFAULT, KEYSEQ
                    FROM SYSCAT.COLUMNS
                    WHERE TABNAME = ? AND TABSCHEMA = ?
                    ORDER BY COLNO
                """
                cursor.execute(query, (table_name.upper(), schema.upper()))
            else:
                query = """
                    SELECT COLNAME, TYPENAME, LENGTH, SCALE, NULLS, DEFAULT, KEYSEQ
                    FROM SYSCAT.COLUMNS
                    WHERE TABNAME = ?
                    ORDER BY COLNO
                """
                cursor.execute(query, (table_name.upper(),))
            
            columns = cursor.fetchall()
            result = {
                "table": table_name,
                "schema": schema,
                "columns": [
                    {
                        "name": row[0].strip() if row[0] else None,
                        "type": row[1].strip() if row[1] else None,
                        "length": row[2],
                        "scale": row[3],
                        "nullable": row[4] == 'Y',
                        "default": row[5].strip() if row[5] else None,
                        "primary_key": row[6] is not None
                    }
                    for row in columns
                ]
            }
            return json.dumps(result, indent=2)
            
        finally:
            cursor.close()
    
    async def get_info(self, table_name: str, schema: Optional[str] = None) -> str:
        """Get detailed table information."""
        cursor = self.connection.cursor()
        try:
            if schema:
                query = """
                    SELECT CARD, NPAGES, FPAGES, OVERFLOW, STATS_TIME
                    FROM SYSCAT.TABLES
                    WHERE TABNAME = ? AND TABSCHEMA = ?
                """
                cursor.execute(query, (table_name.upper(), schema.upper()))
            else:
                query = """
                    SELECT CARD, NPAGES, FPAGES, OVERFLOW, STATS_TIME
                    FROM SYSCAT.TABLES
                    WHERE TABNAME = ?
                """
                cursor.execute(query, (table_name.upper(),))
            
            info = cursor.fetchone()
            if info:
                result = {
                    "table": table_name,
                    "schema": schema,
                    "row_count": info[0],
                    "pages": info[1],
                    "formatted_pages": info[2],
                    "overflow": info[3],
                    "stats_updated": str(info[4]) if info[4] else None
                }
                return json.dumps(result, indent=2)
            else:
                return json.dumps({"error": "Table not found"})
                
        finally:
            cursor.close()
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = DB2MCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())