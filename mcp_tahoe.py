#!/usr/bin/env python3
"""
MCP Server for Tahoe/Hadoop/Hive API
Provides access to historical data through GitHub Copilot
"""

import asyncio
import os
import json
from typing import Any, Optional
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


class TahoeAPIClient:
    """Client for interacting with Tahoe/Hadoop/Hive API"""
    
    def __init__(self, api_endpoint: str, auth_code: str, timeout: int = 30):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.auth_code = auth_code
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_code}"
        }
    
    async def execute_query(self, query: str, database: Optional[str] = None) -> dict:
        """Execute a Hive SQL query"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/query",
                    json={
                        "query": query,
                        "database": database or "default"
                    },
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise Exception(f"Query execution failed: {str(e)}")
    
    async def list_databases(self) -> dict:
        """List all available databases"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.api_endpoint}/databases",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise Exception(f"Failed to list databases: {str(e)}")
    
    async def list_tables(self, database: str) -> dict:
        """List all tables in a specific database"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.api_endpoint}/databases/{database}/tables",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise Exception(f"Failed to list tables: {str(e)}")
    
    async def get_table_schema(self, database: str, table: str) -> dict:
        """Get the schema of a specific table"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.api_endpoint}/databases/{database}/tables/{table}/schema",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise Exception(f"Failed to get table schema: {str(e)}")
    
    async def get_historical_data(
        self,
        table: str,
        start_date: str,
        end_date: str,
        columns: Optional[list] = None,
        filters: Optional[dict] = None
    ) -> dict:
        """Retrieve historical data from a table within a date range"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/historical-data",
                    json={
                        "table": table,
                        "startDate": start_date,
                        "endDate": end_date,
                        "columns": columns or ["*"],
                        "filters": filters or {}
                    },
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise Exception(f"Failed to fetch historical data: {str(e)}")


# Initialize the MCP server
app = Server("tahoe-hive-server")

# Initialize API client
api_endpoint = os.getenv("TAHOE_API_ENDPOINT", "http://localhost:8080/api")
auth_code = os.getenv("TAHOE_AUTH_CODE", "")
timeout = int(os.getenv("TAHOE_TIMEOUT", "30"))

if not auth_code:
    raise ValueError("TAHOE_AUTH_CODE environment variable is required")

api_client = TahoeAPIClient(api_endpoint, auth_code, timeout)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for the MCP server"""
    return [
        Tool(
            name="execute_hive_query",
            description="Execute a Hive SQL query on the Tahoe/Hadoop cluster. Use this for custom queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Hive SQL query to execute"
                    },
                    "database": {
                        "type": "string",
                        "description": "The database to query (optional, defaults to 'default')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_databases",
            description="List all available databases in the Hive metastore",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_tables",
            description="List all tables in a specific database",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "The database name"
                    }
                },
                "required": ["database"]
            }
        ),
        Tool(
            name="get_table_schema",
            description="Get the schema/structure of a specific table",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "The database name"
                    },
                    "table": {
                        "type": "string",
                        "description": "The table name"
                    }
                },
                "required": ["database", "table"]
            }
        ),
        Tool(
            name="get_historical_data",
            description="Retrieve historical data from a table within a date range. Optimized for time-series queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "The full table name (database.table or just table)"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to retrieve (optional, defaults to all)"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Additional filters as key-value pairs (optional)"
                    }
                },
                "required": ["table", "start_date", "end_date"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool execution requests"""
    try:
        if name == "execute_hive_query":
            result = await api_client.execute_query(
                query=arguments["query"],
                database=arguments.get("database")
            )
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "list_databases":
            result = await api_client.list_databases()
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "list_tables":
            result = await api_client.list_tables(
                database=arguments["database"]
            )
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "get_table_schema":
            result = await api_client.get_table_schema(
                database=arguments["database"],
                table=arguments["table"]
            )
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "get_historical_data":
            result = await api_client.get_historical_data(
                table=arguments["table"],
                start_date=arguments["start_date"],
                end_date=arguments["end_date"],
                columns=arguments.get("columns"),
                filters=arguments.get("filters")
            )
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Main entry point for the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import sys
    print("Tahoe/Hive MCP Server running on stdio", file=sys.stderr)
    asyncio.run(main())