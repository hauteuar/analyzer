"""
MCP Server Wrapper for COBOL RAG
Implements proper MCP protocol for GitHub Copilot integration
"""

import sys
import json
import os
import logging
from typing import Dict, Any

# Disable output buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import from cobol_rag_agent
try:
    from cobol_rag_agent import COBOLIndexer, MCPServer
except ImportError:
    error = {
        "jsonrpc": "2.0",
        "id": None,
        "error": {
            "code": -32000,
            "message": "cobol_rag_agent.py not found"
        }
    }
    print(json.dumps(error))
    sys.stdout.flush()
    sys.exit(1)

# Setup logging to file only (not stdout)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('mcp_server.log', mode='w')]
)
logger = logging.getLogger(__name__)


class MCPServerWrapper:
    """MCP Protocol Wrapper for COBOL RAG Agent"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.initialized = False
        self.server = None
        
        logger.info(f"Initializing MCP Server with index: {index_dir}")
        
        # Load indexes
        try:
            indexer = COBOLIndexer(index_dir)
            indexer.load_all()
            self.server = MCPServer(indexer.code_index, indexer.doc_index, indexer.graph)
            logger.info("Indexes loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load indexes: {e}", exc_info=True)
            raise
    
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        logger.info("Handling initialize request")
        self.initialized = True
        
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "cobol-rag",
                "version": "1.0.0"
            }
        }
    
    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        logger.info("Handling tools/list request")
        
        tools = [
            {
                "name": "search_code",
                "description": "Search COBOL/JCL code semantically using natural language queries. Returns relevant code chunks with program names, file locations, and similarity scores.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'DB2 SELECT customer table', 'CICS READ commands', 'MQ message processing')"
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
            {
                "name": "graph_neighbors",
                "description": "Get program relationships, dependencies, and interfaces (DB2 tables, CICS commands, MQ operations). Shows what a program calls and what calls it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node": {
                            "type": "string",
                            "description": "Node ID in format 'prog:PROGRAMNAME' (e.g., 'prog:CUSTUPDT') or 'table:TABLENAME' (e.g., 'table:CUSTOMER')"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Traversal depth - how many levels of relationships to explore (default: 1)",
                            "default": 1
                        }
                    },
                    "required": ["node"]
                }
            },
            {
                "name": "flow_mermaid",
                "description": "Generate Mermaid flow diagram showing program flow, calls, DB2 tables, CICS commands, and MQ operations. Returns diagram code that can be visualized.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node": {
                            "type": "string",
                            "description": "Program to visualize in format 'prog:PROGRAMNAME' (e.g., 'prog:CUSTUPDT')"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Diagram depth - how many levels to show (default: 2)",
                            "default": 2
                        }
                    },
                    "required": ["node"]
                }
            },
            {
                "name": "resolve_dynamic_call",
                "description": "Resolve dynamic CALL statements (CALL WS-VARIABLE) by analyzing the code context to determine possible target programs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "variable": {
                            "type": "string",
                            "description": "Variable name used in dynamic CALL (e.g., 'WS-PROGRAM-NAME')"
                        },
                        "context": {
                            "type": "string",
                            "description": "COBOL source code context containing the CALL statement and preceding MOVE statements"
                        }
                    },
                    "required": ["variable", "context"]
                }
            },
            {
                "name": "combined_search",
                "description": "Comprehensive search that combines code search, documentation search, and graph analysis. Provides the most complete context about a topic.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'customer update transaction flow')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results per search type (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        return {"tools": tools}
    
    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Handling tool call: {tool_name} with args: {arguments}")
        
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
            logger.error(f"Unknown tool: {tool_name}")
            raise Exception(f"Unknown tool: {tool_name}")
        
        # Call the underlying server
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": arguments
        }
        
        response = self.server.handle_request(request)
        logger.info(f"Tool call response: {json.dumps(response)[:200]}...")
        
        if "error" in response:
            raise Exception(response["error"]["message"])
        
        # Format result as MCP expects
        result_text = json.dumps(response["result"], indent=2)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": result_text
                }
            ]
        }
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.info(f"Received request: method={method}, id={request_id}")
        
        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "initialized":
                # Client confirms initialization
                logger.info("Client confirmed initialization")
                return None  # No response needed for notification
            elif method == "tools/list":
                result = self.handle_tools_list(params)
            elif method == "tools/call":
                result = self.handle_tools_call(params)
            elif method == "ping":
                result = {}
            else:
                logger.error(f"Unknown method: {method}")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            logger.info(f"Sending response: {json.dumps(response)[:200]}...")
            return response
        
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
        logger.info("=" * 70)
        logger.info("MCP Server started - listening on stdin/stdout")
        logger.info("=" * 70)
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            logger.info(f"Received: {line}")
            
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                
                if response is not None:  # Some notifications don't need responses
                    output = json.dumps(response)
                    print(output)
                    sys.stdout.flush()
                    logger.info(f"Sent: {output[:200]}...")
                    
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
                print(json.dumps(error_response))
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)


def main():
    """Main entry point with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MCP Server for COBOL RAG')
    parser.add_argument('command', nargs='?', default='serve', 
                       help='Command to run (default: serve)')
    parser.add_argument('--index-dir', default=None,
                       help='Index directory path')
    
    args = parser.parse_args()
    
    logger.info("Starting MCP Server Wrapper")
    
    # Get index directory from args, environment, or default
    if args.index_dir:
        index_dir = args.index_dir
    else:
        index_dir = os.getenv('INDEX_DIR', './index')
    
    # Convert relative to absolute path
    index_dir = os.path.abspath(index_dir)
    logger.info(f"Index directory: {index_dir}")
    
    if not os.path.exists(index_dir):
        error = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32000,
                "message": f"Index directory not found: {index_dir}. Run batch_parser.py first."
            }
        }
        print(json.dumps(error))
        sys.stdout.flush()
        logger.error(f"Index directory not found: {index_dir}")
        sys.exit(1)
    
    try:
        server = MCPServerWrapper(index_dir)
        server.run()
    except Exception as e:
        logger.error(f"Server initialization failed: {e}", exc_info=True)
        error = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32000,
                "message": f"Server initialization failed: {str(e)}"
            }
        }
        print(json.dumps(error))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == '__main__':
    main()


