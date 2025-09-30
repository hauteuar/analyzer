#!/usr/bin/env python3
"""
Tableau MCP Server for GitHub Copilot
Connects Tableau Server to GitHub Copilot via MCP protocol
"""

import asyncio
import json
import sys
from typing import Any
import tableauserverclient as TSC
from mcp.server import Server
from mcp.types import Tool, TextContent

# Tableau Server Configuration
TABLEAU_SERVER_URL = "https://your-tableau-server.com"
TABLEAU_TOKEN_NAME = "your-token-name"
TABLEAU_TOKEN_VALUE = "your-token-value"
TABLEAU_SITE_ID = ""  # Leave empty for default site

# Initialize MCP Server
app = Server("tableau-mcp-server")

# Global Tableau connection
tableau_auth = None
server = None

def init_tableau_connection():
    """Initialize Tableau Server connection"""
    global tableau_auth, server
    tableau_auth = TSC.PersonalAccessTokenAuth(
        TABLEAU_TOKEN_NAME, 
        TABLEAU_TOKEN_VALUE, 
        TABLEAU_SITE_ID
    )
    server = TSC.Server(TABLEAU_SERVER_URL, use_server_version=True)

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Tableau tools"""
    return [
        Tool(
            name="list_workbooks",
            description="List all workbooks available on Tableau Server",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_views",
            description="List all views (reports) available on Tableau Server",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_view_data",
            description="Get data from a specific Tableau view/report by name",
            inputSchema={
                "type": "object",
                "properties": {
                    "view_name": {
                        "type": "string",
                        "description": "Name of the Tableau view to query",
                    }
                },
                "required": ["view_name"],
            },
        ),
        Tool(
            name="query_workbook_info",
            description="Get detailed information about a specific workbook",
            inputSchema={
                "type": "object",
                "properties": {
                    "workbook_name": {
                        "type": "string",
                        "description": "Name of the workbook to query",
                    }
                },
                "required": ["workbook_name"],
            },
        ),
        Tool(
            name="get_datasource_info",
            description="List all data sources available on Tableau Server",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls from GitHub Copilot"""
    
    try:
        if name == "list_workbooks":
            return await list_workbooks()
        elif name == "list_views":
            return await list_views()
        elif name == "get_view_data":
            return await get_view_data(arguments.get("view_name"))
        elif name == "query_workbook_info":
            return await query_workbook_info(arguments.get("workbook_name"))
        elif name == "get_datasource_info":
            return await get_datasource_info()
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def list_workbooks() -> list[TextContent]:
    """List all workbooks on Tableau Server"""
    with server.auth.sign_in(tableau_auth):
        workbooks = list(TSC.Pager(server.workbooks))
        
        result = "Available Workbooks:\n\n"
        for wb in workbooks:
            result += f"- {wb.name}\n"
            result += f"  ID: {wb.id}\n"
            result += f"  Project: {wb.project_name}\n"
            result += f"  Created: {wb.created_at}\n\n"
        
        return [TextContent(type="text", text=result)]

async def list_views() -> list[TextContent]:
    """List all views on Tableau Server"""
    with server.auth.sign_in(tableau_auth):
        views = list(TSC.Pager(server.views))
        
        result = "Available Views (Reports):\n\n"
        for view in views:
            result += f"- {view.name}\n"
            result += f"  ID: {view.id}\n"
            result += f"  Workbook: {view.workbook_id}\n\n"
        
        return [TextContent(type="text", text=result)]

async def get_view_data(view_name: str) -> list[TextContent]:
    """Get data from a specific view"""
    with server.auth.sign_in(tableau_auth):
        views = list(TSC.Pager(server.views))
        
        target_view = None
        for view in views:
            if view.name.lower() == view_name.lower():
                target_view = view
                break
        
        if not target_view:
            return [TextContent(
                type="text", 
                text=f"View '{view_name}' not found. Use list_views to see available views."
            )]
        
        # Get view image and data
        server.views.populate_csv(target_view)
        
        result = f"View: {target_view.name}\n"
        result += f"ID: {target_view.id}\n"
        result += f"Content URL: {target_view.content_url}\n"
        result += f"\nNote: CSV data has been populated. Use Tableau's API to download full data.\n"
        
        return [TextContent(type="text", text=result)]

async def query_workbook_info(workbook_name: str) -> list[TextContent]:
    """Get detailed information about a workbook"""
    with server.auth.sign_in(tableau_auth):
        workbooks = list(TSC.Pager(server.workbooks))
        
        target_wb = None
        for wb in workbooks:
            if wb.name.lower() == workbook_name.lower():
                target_wb = wb
                break
        
        if not target_wb:
            return [TextContent(
                type="text",
                text=f"Workbook '{workbook_name}' not found."
            )]
        
        # Get detailed info
        server.workbooks.populate_views(target_wb)
        server.workbooks.populate_connections(target_wb)
        
        result = f"Workbook: {target_wb.name}\n"
        result += f"ID: {target_wb.id}\n"
        result += f"Project: {target_wb.project_name}\n"
        result += f"Created: {target_wb.created_at}\n"
        result += f"Updated: {target_wb.updated_at}\n\n"
        
        result += "Views in this workbook:\n"
        for view in target_wb.views:
            result += f"  - {view.name}\n"
        
        return [TextContent(type="text", text=result)]

async def get_datasource_info() -> list[TextContent]:
    """List all data sources"""
    with server.auth.sign_in(tableau_auth):
        datasources = list(TSC.Pager(server.datasources))
        
        result = "Available Data Sources:\n\n"
        for ds in datasources:
            result += f"- {ds.name}\n"
            result += f"  ID: {ds.id}\n"
            result += f"  Type: {ds.datasource_type}\n"
            result += f"  Project: {ds.project_name}\n\n"
        
        return [TextContent(type="text", text=result)]

async def main():
    """Main entry point"""
    # Initialize Tableau connection
    init_tableau_connection()
    
    # Run the MCP server
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())