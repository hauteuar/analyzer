"""
Program Flow Query Analyzer Extension
Detects complex program flow analysis requirements and routes appropriately
"""

import re
from typing import Dict, List, Optional
from enum import Enum

class ProgramFlowQueryType(Enum):
    """Program flow query types"""
    PROGRAM_CALL_CHAIN = "program_call_chain"
    DATA_PASSING_ANALYSIS = "data_passing_analysis"
    FILE_UPDATE_FLOW = "file_update_flow"
    BUSINESS_PROCESS_FLOW = "business_process_flow"
    DFHCOMMAREA_ANALYSIS = "dfhcommarea_analysis"
    CROSS_PROGRAM_DATA_FLOW = "cross_program_data_flow"

class ProgramFlowQueryAnalyzer:
    """Analyzer for program flow queries"""
    
    def __init__(self):
        # Patterns indicating program flow analysis needs
        self.program_flow_patterns = {
            'call_chain': [
                r'calls?\s+.*program', r'calling\s+.*program', r'program\s+flow',
                r'XCTL.*program', r'LINK.*program', r'calls?\s+via', r'routes?\s+to'
            ],
            'data_passing': [
                r'DFHCOMMAREA', r'data.*passed', r'values?\s+passed', r'parameters?',
                r'passed.*to.*program', r'shared.*data', r'communication.*area'
            ],
            'file_operations': [
                r'updating?\s+.*file', r'writes?\s+to.*file', r'file.*operations?',
                r'PRO\s+file', r'updating?\s+PRO', r'file.*updates?'
            ],
            'business_flow': [
                r'business\s+flow', r'process\s+flow', r'end.*to.*end',
                r'what.*does.*program.*do', r'business\s+process', r'workflow'
            ],
            'cross_program': [
                r'between\s+programs?', r'across\s+programs?', r'multiple\s+programs?',
                r'program.*interaction', r'program.*communication'
            ]
        }
        
        # Keywords that indicate complex analysis
        self.complexity_indicators = [
            'via', 'through', 'calling', 'passing', 'updating', 'flow', 'process',
            'interaction', 'communication', 'chain', 'sequence'
        ]
    
    def analyze_program_flow_query(self, message: str, entities: List[str]) -> Dict:
        """Analyze if query requires program flow analysis"""
        message_lower = message.lower()
        
        # Check for program flow indicators
        flow_indicators = ['program', 'call', 'xctl', 'link', 'flow', 'process']
        is_flow_query = any(indicator in message_lower for indicator in flow_indicators)
        
        if not is_flow_query:
            return {'is_program_flow_query': False}
        
        # Determine specific flow analysis types
        flow_types = []
        
        for analysis_type, patterns in self.program_flow_patterns.items():
            if any(re.search(pattern, message_lower) for pattern in patterns):
                flow_types.append(analysis_type)
        
        # Check complexity level
        complexity_score = len(flow_types)
        complexity_score += sum(1 for indicator in self.complexity_indicators 
                              if indicator in message_lower)
        
        # Check for specific program mentions
        program_entities = [e for e in entities if self._looks_like_program_name(e)]
        
        return {
            'is_program_flow_query': complexity_score > 0,
            'flow_analysis_types': flow_types,
            'complexity_score': complexity_score,
            'program_entities': program_entities,
            'requires_cross_program_analysis': 'cross_program' in flow_types,
            'requires_data_flow_analysis': 'data_passing' in flow_types,
            'requires_file_analysis': 'file_operations' in flow_types
        }
    
    def _looks_like_program_name(self, entity: str) -> bool:
        """Check if entity looks like a program name"""
        # COBOL program naming patterns
        return (len(entity) >= 4 and 
                entity.isalnum() and 
                entity.isupper() and
                not entity.isdigit())
    
    def generate_program_flow_prompt(self, flow_analysis: Dict, original_query: str) -> str:
        """Generate specialized prompt for program flow analysis"""
        
        program_name = flow_analysis.get('program_name', 'UNKNOWN')
        
        prompt_parts = [
            f"You are analyzing the COBOL program flow starting from '{program_name}' in a wealth management system.",
            f"User question: {original_query}",
            "",
            "Based on the comprehensive program flow analysis below, provide a detailed explanation covering:",
        ]
        
        # Add specific instructions based on flow types detected
        if flow_analysis.get('program_calls'):
            prompt_parts.extend([
                "1. PROGRAM CALL CHAIN: Explain the sequence of program calls and their purpose",
                "2. CONTROL FLOW: Describe how control flows between programs (XCTL, LINK, etc.)",
            ])
        
        if flow_analysis.get('data_passing_analysis'):
            prompt_parts.extend([
                "3. DATA PASSING: Explain how data is passed between programs",
                "4. DFHCOMMAREA USAGE: Describe the structure and content of shared data areas",
            ])
        
        if flow_analysis.get('file_operations'):
            prompt_parts.extend([
                "5. FILE OPERATIONS: Explain file reads, writes, and updates in the flow",
                "6. DATA PERSISTENCE: Describe what data is saved and where",
            ])
        
        if flow_analysis.get('business_flow'):
            prompt_parts.extend([
                "7. BUSINESS PROCESS: Explain the overall business purpose and outcome",
                "8. END-TO-END FLOW: Describe the complete business process from start to finish",
            ])
        
        prompt_parts.extend([
            "",
            "Focus on the business purpose and data flow. Use specific examples from the analysis.",
            "Explain technical concepts in business terms when possible.",
            "",
            "=== PROGRAM FLOW ANALYSIS DATA ==="
        ])
        
        # Add program call chain
        if flow_analysis.get('program_calls'):
            prompt_parts.extend([
                "",
                "PROGRAM CALL CHAIN:",
            ])
            
            for i, call in enumerate(flow_analysis['program_calls'], 1):
                prompt_parts.extend([
                    f"{i}. {call.source_program} → {call.target_program}",
                    f"   Call Type: {call.call_type}",
                    f"   Data Area: {call.data_area}",
                    f"   Business Context: {call.business_context}",
                    f"   Line: {call.line_number}",
                    ""
                ])
        
        # Add data passing analysis
        if flow_analysis.get('data_passing_analysis'):
            data_passing = flow_analysis['data_passing_analysis']
            
            if data_passing.get('dfhcommarea_usage'):
                prompt_parts.extend([
                    "",
                    "DFHCOMMAREA USAGE:",
                ])
                
                for usage in data_passing['dfhcommarea_usage']:
                    prompt_parts.extend([
                        f"Program: {usage['program']} (Line {usage['line_number']})",
                        f"Usage: {usage['usage_type']}",
                        f"Business Context: {usage['business_context']}",
                        "",
                        "Code Context:",
                    ])
                    prompt_parts.extend(usage['context_lines'])
                    prompt_parts.append("")
        
        # Add file operations
        if flow_analysis.get('file_operations'):
            prompt_parts.extend([
                "",
                "FILE OPERATIONS:",
            ])
            
            for file_op in flow_analysis['file_operations']:
                prompt_parts.extend([
                    f"Program: {file_op.program}",
                    f"Operation: {file_op.operation_type} on {file_op.file_name}",
                    f"Record Layout: {file_op.record_layout}",
                    f"Data Flow: {file_op.data_flow_direction}",
                    f"Business Purpose: {file_op.business_purpose}",
                    f"Related Calls: {', '.join(file_op.related_program_calls)}",
                    ""
                ])
        
        # Add business flow summary
        if flow_analysis.get('business_flow'):
            business_flow = flow_analysis['business_flow']
            prompt_parts.extend([
                "",
                "BUSINESS FLOW SUMMARY:",
                f"Flow Name: {business_flow.flow_name}",
                f"Entry Point: {business_flow.entry_point}",
                f"Business Purpose: {business_flow.business_purpose}",
                f"End Result: {business_flow.end_result}",
                ""
            ])
        
        # Add cross-program dependencies
        if flow_analysis.get('cross_program_dependencies'):
            prompt_parts.extend([
                "",
                "CROSS-PROGRAM DEPENDENCIES:",
            ])
            
            for dep in flow_analysis['cross_program_dependencies']:
                prompt_parts.append(f"• {dep.get('description', 'Dependency relationship')}")
        
        prompt_parts.append("")
        prompt_parts.append(flow_analysis.get('flow_summary', ''))
        
        return "\n".join(prompt_parts)