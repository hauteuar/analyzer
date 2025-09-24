#!/usr/bin/env python3
"""
MCP File Operations Server - Remote SSH Version
Single file MCP server with SSH support for remote Linux servers
"""

import json
import os
import sys
import re
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import tempfile


class RemoteMCPFileServer:
    def __init__(self):
        # Load configuration from environment variables or config file
        self.load_config()
        self.setup_signal_handlers()
        self.run()

    def load_config(self):
        """Load SSH configuration from environment or config file"""
        # Try to load from environment variables first
        self.ssh_host = os.getenv('MCP_SSH_HOST')
        self.ssh_user = os.getenv('MCP_SSH_USER')
        self.ssh_password = os.getenv('MCP_SSH_PASSWORD')
        self.ssh_key_file = os.getenv('MCP_SSH_KEY_FILE')
        self.ssh_port = int(os.getenv('MCP_SSH_PORT', '22'))

        # Try to load from config file if env vars not set
        config_file = os.getenv('MCP_CONFIG_FILE', 'mcp_config.json')
        if not self.ssh_host and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.ssh_host = config.get('ssh_host')
                    self.ssh_user = config.get('ssh_user')
                    self.ssh_password = config.get('ssh_password')
                    self.ssh_key_file = config.get('ssh_key_file')
                    self.ssh_port = config.get('ssh_port', 22)
            except Exception as e:
                self.log_error(f"Error loading config file: {e}")

        # Default to local if no SSH config provided
        self.use_ssh = bool(self.ssh_host and self.ssh_user)
        
        if self.use_ssh:
            self.log_error(f"Configured for SSH: {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")
        else:
            self.log_error("Running in local mode (no SSH configuration found)")

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

    def execute_command(self, command: str, input_data: str = None) -> tuple:
        """Execute command locally or via SSH"""
        if not self.use_ssh:
            # Local execution
            try:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    input=input_data,
                    timeout=30
                )
                return result.stdout, result.stderr, result.returncode
            except Exception as e:
                return "", str(e), 1
        else:
            # SSH execution
            return self.execute_ssh_command(command, input_data)

    def execute_ssh_command(self, command: str, input_data: str = None) -> tuple:
        """Execute command via SSH"""
        try:
            # Build SSH command
            ssh_cmd = ['ssh']
            
            # Add port if specified
            if self.ssh_port != 22:
                ssh_cmd.extend(['-p', str(self.ssh_port)])
            
            # Add key file if specified
            if self.ssh_key_file and os.path.exists(self.ssh_key_file):
                ssh_cmd.extend(['-i', self.ssh_key_file])
            
            # SSH options for automation
            ssh_cmd.extend([
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'LogLevel=ERROR'
            ])
            
            # Add user@host
            ssh_cmd.append(f"{self.ssh_user}@{self.ssh_host}")
            
            # Add the command
            ssh_cmd.append(command)
            
            # Handle password authentication with sshpass if available
            if self.ssh_password and not self.ssh_key_file:
                # Try to use sshpass for password authentication
                try:
                    result = subprocess.run(['which', 'sshpass'], capture_output=True)
                    if result.returncode == 0:
                        ssh_cmd = ['sshpass', '-p', self.ssh_password] + ssh_cmd
                    else:
                        self.log_error("Warning: sshpass not available, password authentication may not work")
                except:
                    pass
            
            # Execute SSH command
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                input=input_data,
                timeout=60
            )
            
            return result.stdout, result.stderr, result.returncode
            
        except subprocess.TimeoutExpired:
            return "", "SSH command timed out", 1
        except Exception as e:
            return "", f"SSH execution error: {e}", 1

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
        server_info = {
            "name": "remote-file-operations-server",
            "version": "1.0.0"
        }
        
        if self.use_ssh:
            server_info["description"] = f"Remote file operations via SSH ({self.ssh_user}@{self.ssh_host})"
        else:
            server_info["description"] = "Local file operations"
            
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": server_info
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
                        "description": f"Find files matching a pattern in a directory {'on remote server' if self.use_ssh else 'locally'}",
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
                        "description": f"Search for a string pattern in a file {'on remote server' if self.use_ssh else 'locally'}",
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

    def find_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find files matching pattern using find command"""
        directory = args.get("directory")
        pattern = args.get("pattern")
        recursive = args.get("recursive", False)
        case_sensitive = args.get("case_sensitive", False)
        max_results = args.get("max_results", 100)

        if not directory or not pattern:
            raise ValueError("Directory and pattern are required")

        try:
            # Build find command
            find_cmd = f"find '{directory}'"
            
            # Add recursive flag
            if not recursive:
                find_cmd += " -maxdepth 1"
            
            # Add file type
            find_cmd += " -type f"
            
            # Add name pattern
            if case_sensitive:
                find_cmd += f" -name '{pattern}'"
            else:
                find_cmd += f" -iname '{pattern}'"
            
            # Add stat info and limit results
            find_cmd += f" -printf '%p|%s|%T@\\n' | head -n {max_results}"
            
            stdout, stderr, returncode = self.execute_command(find_cmd)
            
            if returncode != 0:
                raise ValueError(f"Find command failed: {stderr}")
            
            # Parse results
            found_files = []
            for line in stdout.strip().split('\n'):
                if line:
                    try:
                        path, size, mtime = line.split('|')
                        found_files.append({
                            "name": os.path.basename(path),
                            "path": path,
                            "size": int(float(size)),
                            "modified": datetime.fromtimestamp(float(mtime)).isoformat()
                        })
                    except ValueError:
                        continue
            
            # Format response
            location = f"{'Remote' if self.use_ssh else 'Local'} server"
            if self.use_ssh:
                location += f" ({self.ssh_user}@{self.ssh_host})"
                
            result_text = f"File Pattern Search Results - {location}\n"
            result_text += f"Directory: {directory}\n"
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
                for i, file_info in enumerate(found_files, 1):
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
        """Search for string in file using grep"""
        filepath = args.get("filepath")
        search_string = args.get("search_string")
        case_sensitive = args.get("case_sensitive", False)
        max_results = args.get("max_results", 100)
        context_lines = args.get("context_lines", 0)

        if not filepath or not search_string:
            raise ValueError("Filepath and search_string are required")

        try:
            # Build grep command
            grep_cmd = "grep"
            
            # Case sensitivity
            if not case_sensitive:
                grep_cmd += " -i"
            
            # Line numbers
            grep_cmd += " -n"
            
            # Context lines
            if context_lines > 0:
                grep_cmd += f" -C {context_lines}"
            
            # Max results (using head)
            grep_cmd += f" '{search_string}' '{filepath}' | head -n {max_results * (1 + 2 * context_lines)}"
            
            stdout, stderr, returncode = self.execute_command(grep_cmd)
            
            # grep returns 1 if no matches found, which is not an error
            if returncode not in [0, 1]:
                raise ValueError(f"Grep command failed: {stderr}")
            
            # Parse results
            lines = stdout.strip().split('\n') if stdout.strip() else []
            
            location = f"{'Remote' if self.use_ssh else 'Local'} server"
            if self.use_ssh:
                location += f" ({self.ssh_user}@{self.ssh_host})"
                
            result_text = f'Search Results for "{search_string}" - {location}\n'
            result_text += f"File: {filepath}\n"
            result_text += f"Found {len([l for l in lines if l and not l.startswith('--')])} matches"
            result_text += f"\nCase sensitive: {case_sensitive}\n\n"

            if not lines or not any(line.strip() for line in lines):
                result_text += "No matches found."
            else:
                result_text += "Matches:\n"
                for line in lines:
                    if line.strip():
                        if line.startswith('--'):
                            result_text += "\n"
                        else:
                            # Parse line number and content
                            try:
                                if ':' in line:
                                    line_num, content = line.split(':', 1)
                                    if search_string.lower() in content.lower() or case_sensitive and search_string in content:
                                        result_text += f"{line_num}: >>> {content} <<<\n"
                                    else:
                                        result_text += f"{line_num}: {content}\n"
                                else:
                                    result_text += f"{line}\n"
                            except:
                                result_text += f"{line}\n"

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
                return None
            else:
                self.send_error(request_id, f"Unknown method: {method}", -32601)
                return None

        except Exception as e:
            self.send_error(request.get("id"), f"Internal error: {e}")
            return None

    def run(self):
        """Main server loop"""
        self.log_error("Remote MCP File Operations Server ready on stdio")
        
        buffer = ""
        
        try:
            while True:
                chunk = sys.stdin.read(1)
                if not chunk:
                    break
                
                buffer += chunk
                
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
    RemoteMCPFileServer()