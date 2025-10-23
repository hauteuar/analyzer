"""
MCP Server Wrapper for COBOL RAG
Implements proper MCP protocol for GitHub Copilot integration
"""

import sys
import json
import os
import logging
from typing import Dict, Any, List

# Import from cobol_rag_agent
try:
    from cobol_rag_agent import COBOLIndexer, MCPServer
except ImportError:
    print(json.dumps({
        "jsonrpc": "2.0",
        "error": {
            "code": -32000,
            "message": "cobol_rag_agent.py not found. Ensure it's in the same directory."
        }
    }), flush=True)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('mcp_server.log')]
)
logger = logging.getLogger(__name__)


class MCPServerWrapper:
    """
    MCP Protocol Wrapper for COBOL RAG Agent
    Handles initialization, capabilities, and tool definitions
    """
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.initialized = False
        self.server = None
        
        logger.info(f"MCP Server initializing with index: {index_dir}")
        
        # Load indexes
        try:
            indexer = COBOLIndexer(index_dir)
            indexer.load_all()
            self.server = MCPServer(indexer.code_index, indexer.doc_index, indexer.graph)
            logger.info("COBOL RAG indexes loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load indexes: {e}")
            raise
    
    def get_server_info(self) -> Dict[str, Any]:
        """Return server information"""
        return {
            "name": "cobol-rag",
            "version": "1.0.0",
            "description": "COBOL RAG system for mainframe code analysis"
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return server capabilities"""
        return {
            "tools": {
                "search_code": {
                    "description": "Search COBOL/JCL code semantically using natural language queries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query (e.g., 'DB2 SELECT customer table')"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                },
                "graph_neighbors": {
                    "description": "Get program relationships, dependencies, and interfaces (DB2, CICS, MQ)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {
                                "type": "string",
                                "description": "Node ID (format: 'prog:PROGRAMNAME' or 'table:TABLENAME')"
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Traversal depth (default: 1)",
                                "default": 1
                            }
                        },
                        "required": ["node"]
                    }
                },
                "flow_mermaid": {
                    "description": "Generate Mermaid flow diagram showing program flow and interfaces",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {
                                "type": "string",
                                "description": "Node ID to visualize (format: 'prog:PROGRAMNAME')"
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Diagram depth (default: 2)",
                                "default": 2
                            }
                        },
                        "required": ["node"]
                    }
                },
                "resolve_dynamic_call": {
                    "description": "Resolve dynamic CALL statements using heuristics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "variable": {
                                "type": "string",
                                "description": "Variable name used in dynamic CALL"
                            },
                            "context": {
                                "type": "string",
                                "description": "Source code context containing the CALL"
                            }
                        },
                        "required": ["variable", "context"]
                    }
                },
                "combined_search": {
                    "description": "Comprehensive search across code, docs, and program graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        }
    
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        logger.info("Handling initialize request")
        self.initialized = True
        
        return {
            "protocolVersion": "1.0",
            "serverInfo": self.get_server_info(),
            "capabilities": self.get_capabilities()
        }
    
    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        logger.info("Handling tools/list request")
        
        capabilities = self.get_capabilities()
        tools = []
        
        for tool_name, tool_def in capabilities["tools"].items():
            tools.append({
                "name": tool_name,
                "description": tool_def["description"],
                "inputSchema": tool_def["parameters"]
            })
        
        return {"tools": tools}
    
    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Handling tool call: {tool_name}")
        
        # Map to internal method names
        method_map = {
            "search_code": "search_code",
            "graph_neighbors": "graph_neighbors",
            "flow_mermaid": "flow_mermaid",
            "resolve_dynamic_call": "resolve_dynamic_call",
            "combined_search": "combined_search"
        }
        
        method = method_map.get(tool_name)
        if not method:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }
        
        # Call the underlying server
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": arguments
        }
        
        response = self.server.handle_request(request)
        
        if "error" in response:
            return {"error": response["error"]}
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(response["result"], indent=2)
                }
            ]
        }
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.info(f"Received request: {method}")
        
        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_tools_list(params)
            elif method == "tools/call":
                result = self.handle_tools_call(params)
            elif method == "ping":
                result = {"status": "ok"}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    def run(self):
        """Run the MCP server on stdin/stdout"""
        logger.info("MCP Server started and listening on stdin")
        
        # Send ready notification
        ready = {
            "jsonrpc": "2.0",
            "method": "notifications/ready",
            "params": {}
        }
        print(json.dumps(ready), flush=True)
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response), flush=True)


def main():
    """Main entry point"""
    # Get index directory from environment or default
    index_dir = os.getenv('INDEX_DIR', './index')
    
    if not os.path.exists(index_dir):
        error = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": f"Index directory not found: {index_dir}. Run batch_parser.py first."
            }
        }
        print(json.dumps(error), flush=True)
        sys.exit(1)
    
    try:
        server = MCPServerWrapper(index_dir)
        server.run()
    except Exception as e:
        error = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": f"Server initialization failed: {str(e)}"
            }
        }
        print(json.dumps(error), flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()