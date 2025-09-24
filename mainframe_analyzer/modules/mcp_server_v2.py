#!/usr/bin/env python3
"""
MCP File Operations Server - Paramiko SSH Version
Single file MCP server with paramiko SSH support for remote Linux servers
"""

import json
import os
import sys
import re
import signal
import stat
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import tempfile
import time

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False


class ParamikoMCPFileServer:
    def __init__(self):
        if not PARAMIKO_AVAILABLE:
            print("Error: paramiko library is required for SSH functionality", file=sys.stderr)
            print("Install with: pip install paramiko", file=sys.stderr)
            sys.exit(1)
            
        self.ssh_client = None
        self.sftp_client = None
        self.load_config()
        self.setup_signal_handlers()
        
        # Initialize SSH connection if configured
        if self.use_ssh:
            self.connect_ssh()
            
        self.run()

    def load_config(self):
        """Load SSH configuration from environment or config file"""
        # Try to load from environment variables first
        self.ssh_host = os.getenv('MCP_SSH_HOST')
        self.ssh_user = os.getenv('MCP_SSH_USER')
        self.ssh_password = os.getenv('MCP_SSH_PASSWORD')
        self.ssh_key_file = os.getenv('MCP_SSH_KEY_FILE')
        self.ssh_port = int(os.getenv('MCP_SSH_PORT', '22'))
        self.ssh_timeout = int(os.getenv('MCP_SSH_TIMEOUT', '10'))

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
                    self.ssh_timeout = config.get('ssh_timeout', 10)
            except Exception as e:
                self.log_error(f"Error loading config file: {e}")

        # Default to local if no SSH config provided
        self.use_ssh = bool(self.ssh_host and self.ssh_user)
        
        if self.use_ssh:
            self.log_error(f"Configured for SSH: {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")
        else:
            self.log_error("Running in local mode (no SSH configuration found)")

    def connect_ssh(self):
        """Establish SSH connection using paramiko"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Prepare connection parameters
            connect_kwargs = {
                'hostname': self.ssh_host,
                'port': self.ssh_port,
                'username': self.ssh_user,
                'timeout': self.ssh_timeout,
                'banner_timeout': self.ssh_timeout,
                'auth_timeout': self.ssh_timeout,
            }
            
            # Use key file if provided, otherwise password
            if self.ssh_key_file and os.path.exists(self.ssh_key_file):
                self.log_error(f"Using SSH key: {self.ssh_key_file}")
                connect_kwargs['key_filename'] = self.ssh_key_file
            elif self.ssh_password:
                self.log_error("Using SSH password authentication")
                connect_kwargs['password'] = self.ssh_password
            else:
                # Try default SSH keys
                self.log_error("Trying default SSH keys")
                connect_kwargs['look_for_keys'] = True
            
            # Connect
            self.ssh_client.connect(**connect_kwargs)
            
            # Initialize SFTP client for file operations
            self.sftp_client = self.ssh_client.open_sftp()
            
            self.log_error(f"Successfully connected to {self.ssh_user}@{self.ssh_host}")
            
        except paramiko.AuthenticationException as e:
            self.log_error(f"SSH authentication failed: {e}")
            raise Exception(f"SSH authentication failed: {e}")
        except paramiko.SSHException as e:
            self.log_error(f"SSH connection error: {e}")
            raise Exception(f"SSH connection error: {e}")
        except Exception as e:
            self.log_error(f"Failed to connect via SSH: {e}")
            raise Exception(f"Failed to connect via SSH: {e}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean up SSH connections"""
        try:
            if self.sftp_client:
                self.sftp_client.close()
            if self.ssh_client:
                self.ssh_client.close()
        except:
            pass

    def log_error(self, message: str):
        """Log error messages to stderr"""
        print(f"MCP Error: {message}", file=sys.stderr)

    def execute_command(self, command: str) -> tuple:
        """Execute command locally or via SSH"""
        if not self.use_ssh:
            # Local execution
            try:
                import subprocess
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                return result.stdout, result.stderr, result.returncode
            except Exception as e:
                return "", str(e), 1
        else:
            # SSH execution using paramiko
            return self.execute_ssh_command(command)

    def execute_ssh_command(self, command: str) -> tuple:
        """Execute command via SSH using paramiko"""
        try:
            if not self.ssh_client:
                raise Exception("SSH client not connected")
            
            # Execute command
            stdin, stdout, stderr = self.ssh_client.exec_command(
                command, 
                timeout=30,
                get_pty=False
            )
            
            # Read output
            stdout_data = stdout.read().decode('utf-8', errors='replace')
            stderr_data = stderr.read().decode('utf-8', errors='replace')
            exit_status = stdout.channel.recv_exit_status()
            
            # Close channels
            stdin.close()
            stdout.close()
            stderr.close()
            
            return stdout_data, stderr_data, exit_status
            
        except paramiko.SSHException as e:
            return "", f"SSH execution error: {e}", 1
        except Exception as e:
            return "", f"Command execution error: {e}", 1

    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """Get file information locally or via SFTP"""
        try:
            if not self.use_ssh:
                # Local file info
                path_obj = Path(filepath)
                if not path_obj.exists():
                    return None
                
                stat_info = path_obj.stat()
                return {
                    'name': path_obj.name,
                    'path': str(path_obj.absolute()),
                    'size': stat_info.st_size,
                    'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                    'is_file': path_obj.is_file()
                }
            else:
                # Remote file info via SFTP
                if not self.sftp_client:
                    return None
                
                try:
                    stat_info = self.sftp_client.stat(filepath)
                    return {
                        'name': os.path.basename(filepath),
                        'path': filepath,
                        'size': stat_info.st_size,
                        'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        'is_file': stat.S_ISREG(stat_info.st_mode)
                    }
                except FileNotFoundError:
                    return None
                    
        except Exception as e:
            self.log_error(f"Error getting file info for {filepath}: {e}")
            return None

    def read_file_content(self, filepath: str, max_size: int = 10*1024*1024) -> str:
        """Read file content locally or via SFTP"""
        try:
            if not self.use_ssh:
                # Local file reading
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read(max_size)
            else:
                # Remote file reading via SFTP
                if not self.sftp_client:
                    raise Exception("SFTP client not available")
                
                with self.sftp_client.open(filepath, 'r') as f:
                    content = f.read(max_size)
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    return content
                    
        except Exception as e:
            raise Exception(f"Error reading file {filepath}: {e}")

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
            "name": "paramiko-file-operations-server",
            "version": "1.0.0"
        }
        
        if self.use_ssh:
            server_info["description"] = f"Remote file operations via Paramiko SSH ({self.ssh_user}@{self.ssh_host})"
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
                        "description": f"Find files matching a pattern in a directory {'on remote server via Paramiko SSH' if self.use_ssh else 'locally'}",
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
                        "description": f"Search for a string pattern in a file {'on remote server via Paramiko SSH' if self.use_ssh else 'locally'}",
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
            find_cmd += f" -printf '%p|%s|%T@\\n' 2>/dev/null | head -n {max_results}"
            
            stdout, stderr, returncode = self.execute_command(find_cmd)
            
            # Parse results
            found_files = []
            for line in stdout.strip().split('\n'):
                if line and '|' in line:
                    try:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            path = parts[0]
                            size = int(float(parts[1]))
                            mtime = float(parts[2])
                            
                            found_files.append({
                                "name": os.path.basename(path),
                                "path": path,
                                "size": size,
                                "modified": datetime.fromtimestamp(mtime).isoformat()
                            })
                    except (ValueError, IndexError):
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
        """Search for string in file"""
        filepath = args.get("filepath")
        search_string = args.get("search_string")
        case_sensitive = args.get("case_sensitive", False)
        max_results = args.get("max_results", 100)
        context_lines = args.get("context_lines", 0)

        if not filepath or not search_string:
            raise ValueError("Filepath and search_string are required")

        try:
            # Read file content
            content = self.read_file_content(filepath)
            lines = content.splitlines()
            
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
                        "content": line,
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
                                "content": lines[j]
                            })

                        # Context after
                        end_after = min(len(lines), i + context_lines + 1)
                        for j in range(i + 1, end_after):
                            match["context_after"].append({
                                "line_number": j + 1,
                                "content": lines[j]
                            })

                    matches.append(match)

            # Format results
            location = f"{'Remote' if self.use_ssh else 'Local'} server"
            if self.use_ssh:
                location += f" ({self.ssh_user}@{self.ssh_host})"
                
            result_text = f'Search Results for "{search_string}" - {location}\n'
            result_text += f"File: {filepath}\n"
            result_text += f"Found {len(matches)} matches"
            if len(matches) >= max_results:
                result_text += f" (limited to {max_results})"
            result_text += f"\nCase sensitive: {case_sensitive}\n\n"

            if not matches:
                result_text += "No matches found."
            else:
                result_text += "Matches:\n"
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
                return None
            else:
                self.send_error(request_id, f"Unknown method: {method}", -32601)
                return None

        except Exception as e:
            self.send_error(request.get("id"), f"Internal error: {e}")
            return None

    def run(self):
        """Main server loop"""
        self.log_error("Paramiko MCP File Operations Server ready on stdio")
        
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
            self.cleanup()
            sys.exit(0)


if __name__ == "__main__":
    ParamikoMCPFileServer()