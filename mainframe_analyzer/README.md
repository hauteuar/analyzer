# Mainframe Code Analyzer - Project Structure & Setup

## 📁 Directory Structure
```
mainframe_analyzer/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── modules/                        # Backend modules
│   ├── __init__.py                # Module init file
│   ├── token_manager.py           # Token management & chunking
│   ├── llm_client.py              # VLLM client integration
│   ├── database_manager.py        # SQLite database operations
│   ├── cobol_parser.py            # COBOL parsing & friendly names
│   ├── field_analyzer.py          # Field mapping analysis
│   ├── component_extractor.py     # Component extraction
│   └── chat_manager.py            # Context-aware chat
├── templates/                      # HTML templates
│   └── index.html                 # Main frontend interface
├── static/                         # Static assets (optional)
│   ├── css/
│   ├── js/
│   └── images/
└── data/                          # Data storage
    └── mainframe_analyzer.db      # SQLite database (auto-created)
```

## 🚀 Setup Instructions

### 1. Create Project Directory
```bash
mkdir mainframe_analyzer
cd mainframe_analyzer
```

### 2. Create Module Structure
```bash
mkdir modules templates static data
touch modules/__init__.py
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure LLM Endpoint
Edit `app.py` and update the LLM endpoint URL:
```python
# Change this line to match your VLLM server
analyzer = MainframeAnalyzer(llm_endpoint="http://your-server-ip:8100/generate")
```

### 5. Create Module Files
Create all the module files in the `modules/` directory:
- `token_manager.py`
- `llm_client.py` 
- `database_manager.py`
- `cobol_parser.py`
- `field_analyzer.py`
- `component_extractor.py`
- `chat_manager.py`

### 6. Create Templates
Place `index.html` in the `templates/` directory.

### 7. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## ⚙️ Configuration Options

### LLM Client Configuration
In `modules/llm_client.py`, you can adjust:
- Timeout settings
- Retry attempts
- Rate limiting delays
- Request headers

### Token Management
In `modules/token_manager.py`:
- `MAX_TOKENS_PER_CALL`: Maximum tokens per LLM call (default: 6000)
- `EFFECTIVE_CONTENT_LIMIT`: Content limit reserving space for prompts (default: 5500)
- `CHUNK_OVERLAP_TOKENS`: Overlap between chunks (default: 200)

### Database Configuration
The SQLite database is automatically created. For production, consider:
- Moving to PostgreSQL or MySQL
- Adding connection pooling
- Implementing database backups

## 🎯 Key Features

### 1. **Modular Architecture**
- Separated concerns across different modules
- Easy to maintain and extend
- Clean interfaces between components

### 2. **Token-Aware Processing**
- Automatic chunking for large files
- Context preservation across chunks
- Rate limiting and retry logic

### 3. **Intelligent Friendly Names**
- Converts COBOL naming conventions to readable names
- Context-aware name generation
- Consistent throughout the application

### 4. **Complete SQL Storage**
- All analysis results persisted
- Queryable field relationships
- Chat context and history stored

### 5. **Smart Chat Context**
- Automatic context detection from queries
- Field and record layout awareness
- Token budget management for responses

### 6. **Field Matrix with Expand/Collapse**
- Hierarchical view of record layouts
- Collapsible field groups
- Usage type color coding

### 7. **Comprehensive Field Mapping**
- COBOL to Oracle data type conversion
- Business logic classification
- Multi-program consolidation

## 🔧 Customization

### Adding New File Types
1. Add parser logic in `component_extractor.py`
2. Update file type detection in frontend
3. Add specific analysis patterns

### Extending Business Logic Patterns
Update `business_logic_patterns` in `field_analyzer.py`:
```python
self.business_logic_patterns['NEW_PATTERN'] = [
    r'pattern_regex_here',
    r'another_pattern'
]
```

### Custom LLM Prompts
Modify prompts in `token_manager.py` method `_get_base_prompt()`:
```python
def _get_base_prompt(self, analysis_type: str) -> str:
    prompts = {
        'your_analysis_type': """
        Your custom prompt here...
        """
    }
```

## 🐛 Troubleshooting

### Common Issues

1. **LLM Connection Failed**
   - Verify VLLM server is running
   - Check endpoint URL and port
   - Ensure network connectivity

2. **Database Errors**
   - Check file permissions in data directory
   - Verify SQLite installation
   - Check disk space

3. **Token Limit Exceeded**
   - Files will be automatically chunked
   - Check token limits in configuration
   - Monitor token usage in UI

4. **Frontend Not Loading**
   - Verify templates directory exists
   - Check Flask template configuration
   - Ensure all static files are present

### Performance Optimization

1. **Database Indexing**
   - Indexes are automatically created
   - Monitor query performance
   - Consider additional indexes for custom queries

2. **Memory Usage**
   - Large files are processed in chunks
   - Conversation history is limited
   - Regular cleanup routines run automatically

3. **Concurrent Processing**
   - Consider using Celery for background processing
   - Implement job queues for large analyses
   - Add progress tracking for long operations

## 📊 Monitoring & Logging

### Log Files
Application logs are written to console. For production:
```python
import logging
logging.basicConfig(
    filename='mainframe_analyzer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Metrics Tracking
- Token usage per session
- Processing time per file
- LLM call success rates
- User interaction patterns

## 🔒 Security Considerations

### Air-Gapped Environment
- No external API calls except to your VLLM server
- All data stays on your infrastructure
- SQLite database for local storage

### Data Privacy
- Code content is processed locally
- Chat conversations stored locally
- No data transmission outside your network

### Access Control
For production deployment, consider:
- User authentication
- Session management
- Role-based access control
- Audit logging

## 🚀 Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
export LLM_ENDPOINT=http://your-vllm-server:8100/generate
export DATABASE_PATH=/data/mainframe_analyzer.db
```

## 📋 Usage Workflow

### 1. Initial Setup
1. Start the application
2. Create a new session (automatic)
3. Upload COBOL files using drag-and-drop interface

### 2. Component Analysis
1. Files are automatically parsed and analyzed
2. Components extracted with friendly names
3. Record layouts identified and stored
4. Field relationships mapped

### 3. Field Matrix Exploration
1. Select record layout or program
2. View hierarchical field structure
3. Expand/collapse field groups with +/- icons
4. See field usage types with color coding

### 4. Field Mapping Analysis
1. Enter target file name
2. System finds all programs that write to file
3. Analyzes business logic patterns
4. Generates Oracle conversion mappings

### 5. Interactive Chat
1. Ask about specific fields or layouts
2. System automatically includes relevant context
3. Get business logic explanations
4. Request conversion guidance

## 🎨 UI Features

### Visual Indicators
- **Usage Types**: Color-coded field backgrounds
  - Blue: INPUT fields
  - Green: OUTPUT fields  
  - Orange: DERIVED fields
  - Pink: REFERENCE fields
  - Purple: STATIC fields
  - Red: UNUSED fields

### Interactive Elements
- **Expand/Collapse**: Click on record layout headers
- **Hover Effects**: Table rows highlight on hover
- **Loading States**: Progress indicators during processing
- **Token Counters**: Real-time token usage display

### Responsive Design
- Collapsible side panels
- Mobile-friendly interface
- Adaptive layouts for different screen sizes

## 🔄 Data Flow

### File Upload Process
```
1. File Upload → 2. COBOL Parser → 3. Component Extractor → 4. Database Storage
                     ↓
5. LLM Analysis ← 4. Token Management ← 3. Content Chunking
                     ↓
6. Results Consolidation → 7. Database Update → 8. UI Refresh
```

### Chat Query Process
```
1. User Query → 2. Query Analysis → 3. Context Building → 4. LLM Call
                     ↓
8. UI Update ← 7. Response Storage ← 6. Response Processing ← 5. LLM Response
```

### Field Mapping Process
```
1. Target File Input → 2. Program Discovery → 3. Field Analysis → 4. LLM Processing
                           ↓
8. Results Display ← 7. Database Storage ← 6. Mapping Generation ← 5. Logic Classification
```

## 🧪 Testing

### Unit Tests
Create `tests/` directory with:
```python
# test_token_manager.py
import unittest
from modules.token_manager import TokenManager

class TestTokenManager(unittest.TestCase):
    def setUp(self):
        self.token_manager = TokenManager()
    
    def test_token_estimation(self):
        text = "MOVE WS-FIELD TO OUTPUT-FIELD"
        tokens = self.token_manager.estimate_tokens(text)
        self.assertGreater(tokens, 0)
```

### Integration Tests
```python
# test_field_analyzer.py
def test_field_mapping_analysis():
    # Test complete field mapping workflow
    pass

def test_chat_context_building():
    # Test chat context functionality
    pass
```

### Load Testing
Use tools like `locust` to test:
- Concurrent file uploads
- Multiple chat sessions
- Large file processing

## 📈 Scaling Considerations

### Horizontal Scaling
- Multiple Flask app instances behind load balancer
- Shared database across instances
- Session affinity for chat conversations

### Vertical Scaling
- Increase server resources for LLM calls
- More memory for large file processing
- SSD storage for database performance

### Database Scaling
- Move from SQLite to PostgreSQL
- Implement read replicas
- Add database connection pooling

## 🎯 Future Enhancements

### Planned Features
1. **Batch Processing**: Queue multiple files for analysis
2. **Export Formats**: Excel, PDF, Word reports
3. **Visual Dependency Maps**: Interactive component relationships
4. **Advanced Analytics**: Code complexity metrics
5. **Collaboration**: Multi-user sessions and sharing
6. **Version Control**: Track analysis changes over time

### API Extensions
```python
# Additional endpoints to consider
@app.route('/api/export/ddl/<session_id>')
def generate_oracle_ddl(session_id):
    # Generate Oracle DDL from COBOL layouts
    pass

@app.route('/api/complexity/<session_id>')
def analyze_complexity(session_id):
    # Calculate code complexity metrics
    pass

@app.route('/api/migrate-plan/<session_id>')
def generate_migration_plan(session_id):
    # Create detailed migration strategy
    pass
```

## 🔍 Advanced Configuration

### Custom Field Usage Patterns
```python
# In field_analyzer.py, add custom patterns
CUSTOM_PATTERNS = {
    'VALIDATION': [
        r'IF\s+.*\s+NOT\s+VALID',
        r'PERFORM\s+VALIDATE-.*'
    ],
    'FORMATTING': [
        r'STRING\s+.*\s+DELIMITED',
        r'UNSTRING\s+.*\s+INTO'
    ]
}
```

### Enhanced Chat Prompts
```python
# Create domain-specific prompts
DOMAIN_PROMPTS = {
    'banking': "Focus on financial calculations and regulatory compliance...",
    'insurance': "Emphasize policy processing and claims handling...",
    'finance': "Focus on  capital markets calcualtions..."
}
```

### Performance Tuning
```python
# Database optimization
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  # 64MB cache
PRAGMA journal_mode = WAL;
PRAGMA temp_store = MEMORY;
```

# Set build flags and install
CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=all-major" pip install llama-cpp-python --verbose

# If no CUDA, use CPU version
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --verbose