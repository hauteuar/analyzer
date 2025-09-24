#!/usr/bin/env python3
"""
MCP File Operations Server - Python Version
Single file MCP server for GitHub Copilot stdio connection.
"""

import json
import os
import sys
import re
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class MCPFileServer:
    def __init__(self):
        self.setup_signal_handlers()
        self.run()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        sys.exit(0)

    def log_error(self, message: str):
        """Log error messages to stderr"""
        print(f"MCP Error: {message}", file=sys.stderr)

    def send_response(self, response: Dict[str, Any]):
        """Send JSON-RPC response to stdout"""
        print(json.dumps(response), flush=True)

    def send_error(self, request_id: Optional[str], message: str, code: int = -32603):
        """Send error response"""
        error_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        self.send_response(error_response)

    def handle_initialize(self, request_id: str) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "file-operations-server",
                    "version": "1.0.0"
                }
            }
        }

    def handle_tools_list(self, request_id: str) -> Dict[str, Any]:
        """Handle tools/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "find_files",
                        "description": "Find files matching a pattern in a directory and return their names",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "directory": {
                                    "type": "string",
                                    "description": "The directory path to search in"
                                },
                                "pattern": {
                                    "type": "string",
                                    "description": "File name pattern to search for (supports wildcards like *.txt, config*, etc.)"
                                },
                                "recursive": {
                                    "type": "boolean",
                                    "description": "Whether to search subdirectories recursively (default: false)",
                                    "default": False
                                },
                                "case_sensitive": {
                                    "type": "boolean",
                                    "description": "Whether pattern matching should be case sensitive (default: false)",
                                    "default": False
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of files to return (default: 100)",
                                    "default": 100
                                }
                            },
                            "required": ["directory", "pattern"]
                        }
                    },
                    {
                        "name": "search_in_file",
                        "description": "Search for a string pattern in a file and return matching lines with line numbers",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "filepath": {
                                    "type": "string",
                                    "description": "The absolute or relative path to the file to search in"
                                },
                                "search_string": {
                                    "type": "string",
                                    "description": "The string to search for in the file"
                                },
                                "case_sensitive": {
                                    "type": "boolean",
                                    "description": "Whether the search should be case sensitive (default: false)",
                                    "default": False
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of matching lines to return (default: 100)",
                                    "default": 100
                                },
                                "context_lines": {
                                    "type": "integer",
                                    "description": "Number of context lines to show around each match (default: 0)",
                                    "default": 0
                                }
                            },
                            "required": ["filepath", "search_string"]
                        }
                    }
                ]
            }
        }

    def handle_tools_call(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        try:
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "find_files":
                result = self.find_files(arguments)
            elif tool_name == "search_in_file":
                result = self.search_in_file(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }

    def matches_pattern(self, filename: str, pattern: str, case_sensitive: bool = False) -> bool:
        """Check if filename matches the given pattern with wildcards"""
        if not case_sensitive:
            filename = filename.lower()
            pattern = pattern.lower()

        # Convert shell-style wildcards to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        # Escape other regex special characters
        regex_pattern = re.escape(regex_pattern).replace(r'\.\*', '.*').replace(r'\.', '.')
        
        return bool(re.match(f'^{regex_pattern}$', filename))

    def search_directory(self, directory: str, pattern: str, recursive: bool, 
                        case_sensitive: bool, max_results: int) -> List[Dict[str, Any]]:
        """Recursively search for files matching pattern"""
        found_files = []
        
        try:
            path_obj = Path(directory)
            if not path_obj.is_dir():
                raise ValueError(f"Path is not a directory: {directory}")

            # Use rglob for recursive search, glob for non-recursive
            if recursive:
                paths = path_obj.rglob('*')
            else:
                paths = path_obj.glob('*')

            for file_path in paths:
                if len(found_files) >= max_results:
                    break

                if file_path.is_file():
                    if self.matches_pattern(file_path.name, pattern, case_sensitive):
                        stat = file_path.stat()
                        found_files.append({
                            "name": file_path.name,
                            "path": str(file_path.absolute()),
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })

        except PermissionError as e:
            self.log_error(f"Permission denied accessing {directory}: {e}")
        except Exception as e:
            raise ValueError(f"Error searching directory: {e}")

        return found_files

    def find_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find files matching pattern in directory"""
        directory = args.get("directory")
        pattern = args.get("pattern")
        recursive = args.get("recursive", False)
        case_sensitive = args.get("case_sensitive", False)
        max_results = args.get("max_results", 100)

        if not directory or not pattern:
            raise ValueError("Directory and pattern are required")

        try:
            found_files = self.search_directory(directory, pattern, recursive, case_sensitive, max_results)
            
            result_text = f"File Pattern Search Results\n"
            result_text += f"Directory: {Path(directory).absolute()}\n"
            result_text += f"Pattern: {pattern}\n"
            result_text += f"Recursive: {recursive}\n"
            result_text += f"Case Sensitive: {case_sensitive}\n"
            result_text += f"Found: {len(found_files)} files"
            
            if len(found_files) >= max_results:
                result_text += f" (showing first {max_results})"
            result_text += "\n\n"

            if not found_files:
                result_text += "No files found matching the pattern."
            else:
                result_text += "Matching files:\n"
                for i, file_info in enumerate(found_files[:max_results], 1):
                    result_text += f"{i}. {file_info['name']}\n"
                    result_text += f"   Path: {file_info['path']}\n"
                    result_text += f"   Size: {file_info['size']} bytes\n"
                    result_text += f"   Modified: {file_info['modified']}\n\n"

            return {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }

        except Exception as e:
            raise ValueError(f"Failed to find files: {e}")

    def search_in_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search for string in file"""
        filepath = args.get("filepath")
        search_string = args.get("search_string")
        case_sensitive = args.get("case_sensitive", False)
        max_results = args.get("max_results", 100)
        context_lines = args.get("context_lines", 0)

        if not filepath or not search_string:
            raise ValueError("Filepath and search_string are required")

        try:
            file_path = Path(filepath)
            if not file_path.exists():
                raise ValueError(f"File does not exist: {filepath}")
            
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {filepath}")

            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                with open(file_path, 'r', encoding='latin1') as f:
                    lines = f.readlines()

            # Search for matches
            search_term = search_string if case_sensitive else search_string.lower()
            matches = []

            for i, line in enumerate(lines):
                if len(matches) >= max_results:
                    break

                search_line = line if case_sensitive else line.lower()
                
                if search_term in search_line:
                    match = {
                        "line_number": i + 1,
                        "content": line.rstrip('\n\r'),
                        "context_before": [],
                        "context_after": []
                    }

                    # Add context lines if requested
                    if context_lines > 0:
                        # Context before
                        start_before = max(0, i - context_lines)
                        for j in range(start_before, i):
                            match["context_before"].append({
                                "line_number": j + 1,
                                "content": lines[j].rstrip('\n\r')
                            })

                        # Context after
                        end_after = min(len(lines), i + context_lines + 1)
                        for j in range(i + 1, end_after):
                            match["context_after"].append({
                                "line_number": j + 1,
                                "content": lines[j].rstrip('\n\r')
                            })

                    matches.append(match)

            # Format results
            result_text = f'Search Results for "{search_string}" in {file_path.absolute()}\n'
            result_text += f"Found {len(matches)} matches"
            if len(matches) >= max_results:
                result_text += f" (limited to {max_results})"
            result_text += f"\nCase sensitive: {case_sensitive}\n\n"

            if not matches:
                result_text += "No matches found."
            else:
                for i, match in enumerate(matches, 1):
                    result_text += f"--- Match {i} ---\n"
                    
                    # Context before
                    for ctx in match["context_before"]:
                        result_text += f"{ctx['line_number']}: {ctx['content']}\n"
                    
                    # The matching line (highlighted)
                    result_text += f"{match['line_number']}: >>> {match['content']} <<<\n"
                    
                    # Context after
                    for ctx in match["context_after"]:
                        result_text += f"{ctx['line_number']}: {ctx['content']}\n"
                    
                    result_text += "\n"

            return {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }

        except Exception as e:
            raise ValueError(f"Failed to search in file: {e}")

    def process_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming MCP request"""
        try:
            method = request.get("method")
            request_id = request.get("id")
            params = request.get("params", {})

            if method == "initialize":
                return self.handle_initialize(request_id)
            elif method == "tools/list":
                return self.handle_tools_list(request_id)
            elif method == "tools/call":
                return self.handle_tools_call(request_id, params)
            elif method == "notifications/initialized":
                # No response needed for notifications
                return None
            else:
                self.send_error(request_id, f"Unknown method: {method}", -32601)
                return None

        except Exception as e:
            self.send_error(request.get("id"), f"Internal error: {e}")
            return None

    def run(self):
        """Main server loop"""
        self.log_error("MCP File Operations Server ready on stdio")
        
        buffer = ""
        
        try:
            while True:
                # Read from stdin
                chunk = sys.stdin.read(1)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Process complete JSON messages (line-separated)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        try:
                            request = json.loads(line)
                            response = self.process_request(request)
                            if response:
                                self.send_response(response)
                        except json.JSONDecodeError as e:
                            self.log_error(f"Invalid JSON: {e}")
                            self.send_error(None, f"Invalid JSON: {e}", -32700)
                        except Exception as e:
                            self.log_error(f"Request processing error: {e}")
                            self.send_error(None, f"Request processing error: {e}")

        except KeyboardInterrupt:
            self.log_error("Server interrupted")
        except Exception as e:
            self.log_error(f"Fatal error: {e}")
        finally:
            sys.exit(0)


if __name__ == "__main__":
    MCPFileServer()