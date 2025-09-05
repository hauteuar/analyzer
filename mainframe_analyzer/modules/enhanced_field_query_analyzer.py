"""
Enhanced Query Analyzer for Field-Specific Queries
Detects complex field analysis requirements and routes to appropriate handlers
"""

import re
from typing import Dict, List, Optional
from enum import Enum

class FieldQueryType(Enum):
    """Specific field query types"""
    GROUP_STRUCTURE_ANALYSIS = "group_structure_analysis"
    CONDITIONAL_ASSIGNMENT_ANALYSIS = "conditional_assignment_analysis"
    TRANSACTION_FLOW_ANALYSIS = "transaction_flow_analysis"
    CROSS_PROGRAM_USAGE = "cross_program_usage"
    VALUE_ASSIGNMENT_PATTERNS = "value_assignment_patterns"
    BUSINESS_LOGIC_FLOW = "business_logic_flow"

class EnhancedFieldQueryAnalyzer:
    """Enhanced analyzer for complex field queries"""
    
    def __init__(self):
        # Patterns that indicate complex field analysis needs
        self.complex_field_patterns = {
            'group_analysis': [
                r'group\s+variable', r'group\s+field', r'structure\s+of', 
                r'child\s+fields', r'sub\s*fields', r'contains\s+what'
            ],
            'conditional_analysis': [
                r'when\s+is.*moved', r'different\s+values', r'conditions?', 
                r'what\s+values', r'how\s+.*\s+populated', r'assigned\s+when'
            ],
            'transaction_analysis': [
                r'transaction', r'XCTL', r'calls?\s+program', r'program\s+flow',
                r'business\s+flow', r'routing', r'control\s+flow'
            ],
            'cross_program': [
                r'across\s+programs', r'other\s+programs', r'multiple\s+programs',
                r'used\s+in.*program', r'called\s+from'
            ],
            'assignment_patterns': [
                r'moved\s+to', r'assigned\s+.*values?', r'populated\s+with',
                r'receives?\s+.*values?', r'different\s+.*assignments?'
            ]
        }
        
        # Business context keywords
        self.business_keywords = [
            'transaction', 'business', 'process', 'flow', 'logic', 'rule',
            'condition', 'when', 'how', 'why', 'purpose', 'used'
        ]
    
    def analyze_field_query(self, message: str, entities: List[str]) -> Dict:
        """Analyze if query requires complex field analysis"""
        message_lower = message.lower()
        
        # Check if this is a field-focused query
        field_indicators = ['field', 'variable', 'data', 'structure']
        is_field_query = any(indicator in message_lower for indicator in field_indicators)
        
        if not is_field_query and not entities:
            return {'is_complex_field_query': False}
        
        # Determine specific analysis types needed
        analysis_types = []
        
        for analysis_type, patterns in self.complex_field_patterns.items():
            if any(re.search(pattern, message_lower) for pattern in patterns):
                analysis_types.append(analysis_type)
        
        # Check for business context requirements
        business_focus = any(keyword in message_lower for keyword in self.business_keywords)
        
        # Determine complexity level
        complexity_score = len(analysis_types) + (0.5 if business_focus else 0)
        
        return {
            'is_complex_field_query': complexity_score > 0,
            'analysis_types': analysis_types,
            'business_focus': business_focus,
            'complexity_score': complexity_score,
            'requires_semantic_search': True,
            'requires_cross_reference': 'cross_program' in analysis_types,
            'requires_flow_analysis': 'transaction_analysis' in analysis_types,
            'field_entities': entities
        }
    
    def generate_field_analysis_prompt(self, field_analysis: Dict, original_query: str) -> str:
        """Generate specialized prompt for field analysis"""
        
        field_name = field_analysis.get('field_name', 'UNKNOWN')
        
        prompt_parts = [
            f"You are analyzing the COBOL field '{field_name}' in a wealth management system.",
            f"User question: {original_query}",
            "",
            "Based on the comprehensive field analysis below, provide a detailed explanation that covers:",
        ]
        
        # Add specific analysis instructions based on field type
        if field_analysis.get('field_type') == 'group':
            prompt_parts.extend([
                "1. GROUP STRUCTURE: Explain the group field structure and its child fields",
                "2. CHILD FIELD PURPOSES: Describe what each child field contains and its business purpose",
            ])
        
        if field_analysis.get('conditional_assignments'):
            prompt_parts.extend([
                "3. VALUE ASSIGNMENTS: Explain when and how different values are assigned to this field",
                "4. BUSINESS CONDITIONS: Describe the business conditions that trigger each assignment",
            ])
        
        if field_analysis.get('control_flow_patterns'):
            prompt_parts.extend([
                "5. PROGRAM FLOW: Explain how this field is used in program control flow (XCTL, CALL, etc.)",
                "6. TRANSACTION ROUTING: Describe how different values route to different programs",
            ])
        
        if field_analysis.get('transaction_patterns'):
            prompt_parts.extend([
                "7. TRANSACTION PATTERNS: Explain the transaction codes and their business meanings",
            ])
        
        prompt_parts.extend([
            "",
            "Make your explanation business-focused and explain the logic flow clearly.",
            "Use specific examples from the code analysis.",
            "",
            "=== FIELD ANALYSIS DATA ==="
        ])
        
        # Add the analysis data
        if field_analysis.get('group_analysis'):
            prompt_parts.extend([
                "",
                f"GROUP STRUCTURE for {field_name}:",
                f"Level: {field_analysis['group_analysis'].level}",
                f"Purpose: {field_analysis['group_analysis'].business_purpose}",
                "",
                "Child Fields:"
            ])
            
            for child in field_analysis['group_analysis'].child_fields:
                child_desc = f"  {child['level']:02d} {child['name']}"
                if child['picture']:
                    child_desc += f" PIC {child['picture']}"
                if child['value']:
                    child_desc += f" VALUE '{child['value']}'"
                prompt_parts.append(child_desc)
        
        if field_analysis.get('conditional_assignments'):
            prompt_parts.extend([
                "",
                "CONDITIONAL ASSIGNMENTS:"
            ])
            
            for assignment in field_analysis['conditional_assignments']:
                prompt_parts.extend([
                    f"",
                    f"Program: {assignment['program']} (Line {assignment['line_number']})",
                    f"Value: {assignment['source_value']}",
                    f"Condition: {assignment['condition']}",
                    f"Business Context: {assignment['business_context']}",
                    f"Transaction ID: {assignment.get('transaction_identifier', 'N/A')}",
                    "",
                    "Code Context:",
                ])
                prompt_parts.extend(assignment['context_lines'])
        
        if field_analysis.get('control_flow_patterns'):
            prompt_parts.extend([
                "",
                "PROGRAM CONTROL FLOW:"
            ])
            
            for flow in field_analysis['control_flow_patterns']:
                prompt_parts.extend([
                    f"",
                    f"Control: {flow['control_type']} from {flow['source_program']} to {flow['target_program']}",
                    f"Usage: {flow['usage_pattern']}",
                    f"Business Flow: {flow['business_flow']}",
                    "",
                    "Context:"
                ])
                prompt_parts.extend(flow['context_lines'])
        
        prompt_parts.append("")
        prompt_parts.append(field_analysis.get('business_logic_summary', ''))
        
        return "\n".join(prompt_parts)