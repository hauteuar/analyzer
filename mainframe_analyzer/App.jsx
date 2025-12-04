import React, { useState, useEffect } from 'react';
import { Plus, Calendar, BarChart3, Trash2, Upload, Settings, Link2, ExternalLink, MessageSquare, RefreshCw, CheckCheck, ChevronRight, ChevronDown, Download, TrendingUp, PieChart, Wifi, WifiOff, Edit2, Filter } from 'lucide-react';

// Inline CSS styles for Node 16 compatibility
const styles = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  
  .container { min-height: 100vh; background-color: #f9fafb; padding-bottom: 80px; }
  .header { background-color: white; border-bottom: 1px solid #e5e7eb; padding: 16px; position: sticky; top: 0; z-index: 40; }
  .header-content { max-width: 1280px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  .app-title { font-size: 24px; font-weight: bold; color: #1f2937; display: flex; align-items: center; gap: 8px; }
  
  .status-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; padding: 2px 8px; border-radius: 4px; }
  .status-badge.online { background-color: #dcfce7; color: #166534; }
  .status-badge.offline { background-color: #fee2e2; color: #991b1b; }
  
  .btn { padding: 8px 16px; border-radius: 8px; border: none; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; font-size: 14px; transition: all 0.2s; }
  .btn:hover { opacity: 0.9; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn.btn-primary { background-color: #2563eb; color: white; }
  .btn.btn-primary:hover { background-color: #1d4ed8; }
  .btn.btn-success { background-color: #16a34a; color: white; }
  .btn.btn-purple { background-color: #9333ea; color: white; }
  .btn.btn-secondary { background-color: #6b7280; color: white; }
  .btn.btn-danger { background-color: #dc2626; color: white; }
  .btn.btn-outline { background-color: white; border: 1px solid #d1d5db; color: #374151; }
  
  .main-content { max-width: 1280px; margin: 0 auto; padding: 24px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap; }
  .tab { display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 8px; font-size: 14px; cursor: pointer; border: none; transition: all 0.2s; }
  .tab.active { background-color: #2563eb; color: white; }
  .tab.inactive { background-color: white; color: #374151; }
  .tab.inactive:hover { background-color: #f3f4f6; }
  
  .card { background-color: white; border-radius: 8px; border: 1px solid #e5e7eb; padding: 24px; margin-bottom: 24px; }
  .grid { display: grid; gap: 16px; }
  .grid-4 { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
  
  .stat-card { padding: 24px; border-radius: 8px; border: 1px solid; }
  .stat-card.blue { background-color: #dbeafe; border-color: #93c5fd; }
  .stat-card.green { background-color: #dcfce7; border-color: #86efac; }
  .stat-card.yellow { background-color: #fef3c7; border-color: #fde68a; }
  .stat-card.red { background-color: #fee2e2; border-color: #fca5a5; }
  
  .modal-overlay { position: fixed; inset: 0; background-color: rgba(0, 0, 0, 0.5); display: flex; align-items: center; justify-content: center; z-index: 50; padding: 16px; }
  .modal { background-color: white; border-radius: 8px; padding: 24px; width: 100%; max-width: 500px; max-height: 90vh; overflow-y: auto; }
  .modal-large { max-width: 900px; }
  
  .form-group { margin-bottom: 16px; }
  .label { display: block; font-size: 14px; font-weight: 600; margin-bottom: 4px; color: #374151; }
  .input, .select, .textarea { width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 14px; }
  .input:focus, .select:focus, .textarea:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1); }
  
  .hierarchy-item { display: flex; align-items: center; gap: 8px; padding: 8px; border: 1px solid; border-radius: 4px; margin-bottom: 4px; cursor: pointer; transition: all 0.2s; }
  .hierarchy-item:hover { background-color: #f9fafb; }
  .hierarchy-item.overdue { background-color: #fee2e2; border-color: #f87171; }
  .hierarchy-item.epic { background-color: #e9d5ff; color: #6b21a8; border-color: #c084fc; }
  .hierarchy-item.story { background-color: #dbeafe; color: #1e40af; border-color: #93c5fd; }
  .hierarchy-item.task { background-color: #dcfce7; color: #166534; border-color: #86efac; }
  .hierarchy-item.subtask { background-color: #f3f4f6; color: #374151; border-color: #d1d5db; }
  
  .status-select { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; border: none; cursor: pointer; }
  .status-select.review { background-color: #dcfce7; color: #166534; }
  .status-select.in-progress { background-color: #dbeafe; color: #1e40af; }
  .status-select.pending { background-color: #f3f4f6; color: #374151; }
  
  .icon-btn { padding: 4px; border: none; background: none; cursor: pointer; transition: all 0.2s; }
  .icon-btn:hover { background-color: #f3f4f6; border-radius: 4px; }
  .icon-btn.purple { color: #9333ea; }
  .icon-btn.blue { color: #2563eb; }
  .icon-btn.red { color: #dc2626; }
  
  .checkbox-wrapper { display: flex; align-items: center; gap: 8px; margin: 12px 0; }
  .checkbox { width: 16px; height: 16px; cursor: pointer; }
  
  .epic-selector { max-height: 300px; overflow-y: auto; border: 1px solid #d1d5db; border-radius: 4px; padding: 8px; }
  .epic-item { display: flex; align-items: center; gap: 8px; padding: 8px; margin: 4px 0; border: 1px solid #e5e7eb; border-radius: 4px; transition: all 0.2s; cursor: pointer; }
  .epic-item:hover { background-color: #f9fafb; }
  .epic-item.selected { background-color: #dbeafe; border-color: #2563eb; }
  
  .gantt-container { position: relative; overflow-x: auto; }
  .gantt-row { position: relative; height: 40px; border-bottom: 1px solid #e5e7eb; }
  .gantt-bar { position: absolute; height: 30px; top: 5px; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: 600; padding: 0 8px; white-space: nowrap; overflow: hidden; }
  .gantt-bar.overdue { background-color: #dc2626; }
  .gantt-bar.review { background-color: #16a34a; }
  .gantt-bar.in-progress { background-color: #2563eb; }
  .gantt-bar.pending { background-color: #6b7280; }
  
  .today-marker { position: absolute; top: 0; bottom: 0; width: 2px; background-color: #dc2626; z-index: 10; }
  .today-dot { position: absolute; top: -4px; left: 50%; transform: translateX(-50%); width: 8px; height: 8px; background-color: #dc2626; border-radius: 50%; }
  
  .chart-container { padding: 20px; background: white; border-radius: 8px; }
  .chart-label { font-size: 12px; color: #6b7280; margin-bottom: 4px; }
  .chart-value { font-size: 20px; font-weight: bold; color: #1f2937; }
  
  .filter-panel { background: white; padding: 16px; border-radius: 8px; margin-bottom: 16px; }
  .filter-option { display: flex; align-items: center; gap: 8px; padding: 8px; }
  
  .custom-field { padding: 12px; background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 4px; margin-bottom: 8px; }
  .custom-field-label { font-size: 12px; color: #6b7280; margin-bottom: 4px; }
`;

const ProjectManager = () => {
  // State Management
  const [backendConnected, setBackendConnected] = useState(false);
  const [useBackend, setUseBackend] = useState(false);
  const [backendUrl, setBackendUrl] = useState('http://localhost:3001/api');
  const [showBackendSettings, setShowBackendSettings] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState(null);
  
  const [projects, setProjects] = useState([]);
  const [activeView, setActiveView] = useState('dashboard');
  const [selectedProject, setSelectedProject] = useState(null);
  const [expandedItems, setExpandedItems] = useState(new Set());
  
  // Modals
  const [showProjectModal, setShowProjectModal] = useState(false);
  const [showItemModal, setShowItemModal] = useState(false);
  const [showJiraSettingsModal, setShowJiraSettingsModal] = useState(false);
  const [showItemDetailsModal, setShowItemDetailsModal] = useState(false);
  const [showEditItemModal, setShowEditItemModal] = useState(false);
  const [showEpicSelectorModal, setShowEpicSelectorModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  
  // Selection States
  const [selectedItem, setSelectedItem] = useState(null);
  const [editingItem, setEditingItem] = useState(null);
  const [availableEpics, setAvailableEpics] = useState([]);
  const [selectedEpics, setSelectedEpics] = useState([]);
  
  // Filters
  const [timelineFilters, setTimelineFilters] = useState({
    showEpics: true,
    showStories: true,
    showTasks: true,
    showSubtasks: false
  });
  
  const [selectedChartType, setSelectedChartType] = useState('gantt');
  
  // Jira Configuration with custom fields
  const [jiraConfig, setJiraConfig] = useState({
    url: '',
    email: '',
    apiToken: '',
    defaultProject: '',
    connected: false,
    customFields: {
      storyPoints: '',
      epicLink: '',
      sprint: '',
      labels: ''
    }
  });
  
  const [jiraProjects, setJiraProjects] = useState([]);
  const [selectedJiraUrl, setSelectedJiraUrl] = useState('');
  const [customJiraUrl, setCustomJiraUrl] = useState('');
  const [testingConnection, setTestingConnection] = useState(false);
  
  const predefinedJiraUrls = [
    { value: 'https://jira3.horizon.bankofamerica.com', label: 'BofA Jira 3 (Horizon)' },
    { value: 'https://jira2.horizon.bankofamerica.com', label: 'BofA Jira 2 (Horizon)' },
    { value: 'custom', label: 'Custom URL' }
  ];
  
  // Form States
  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    startDate: '',
    endDate: '',
    status: 'planning'
  });
  
  const [newItem, setNewItem] = useState({
    name: '',
    type: 'task',
    parentId: null,
    status: 'pending',
    priority: 'medium',
    startDate: '',
    endDate: '',
    assignee: '',
    estimatedHours: 0,
    createInJira: false,
    jiraEpicName: '',
    jiraStoryPoints: '',
    jiraLabels: ''
  });
  
  const [dateValidationError, setDateValidationError] = useState('');
  const [newComment, setNewComment] = useState('');
  const [postToJira, setPostToJira] = useState(false);

  // Initialize
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = styles;
    document.head.appendChild(style);
    
    loadInitialData();
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);
  
  useEffect(() => {
    if (!useBackend) {
      localStorage.setItem('projectManagerData', JSON.stringify(projects));
    }
  }, [projects, useBackend]);

  // Load initial data
  const loadInitialData = () => {
    try {
      const saved = localStorage.getItem('projectManagerData');
      const savedJira = localStorage.getItem('jiraConfig');
      
      if (saved) {
        setProjects(JSON.parse(saved));
      } else {
        setProjects(getDefaultProjects());
      }
      
      if (savedJira) {
        setJiraConfig(JSON.parse(savedJira));
      }
      
      const savedBackend = localStorage.getItem('useBackend') === 'true';
      const savedUrl = localStorage.getItem('backendUrl');
      
      if (savedUrl) setBackendUrl(savedUrl);
      if (savedBackend) {
        setUseBackend(true);
        checkBackendConnection();
      }
    } catch (error) {
      console.error('Error loading data:', error);
      setProjects(getDefaultProjects());
    }
  };

  const getDefaultProjects = () => [{
    id: 1,
    name: 'Website Redesign',
    description: 'Complete overhaul of company website',
    startDate: '2024-10-01',
    endDate: '2024-12-31',
    status: 'in-progress',
    items: []
  }];
  
  // Backend functions
  const checkBackendConnection = async () => {
    try {
      const response = await fetch(`${backendUrl}/health`);
      if (response.ok) {
        setBackendConnected(true);
        syncFromBackend();
        return true;
      }
    } catch (error) {
      setBackendConnected(false);
    }
    return false;
  };
  
  const enableBackend = async () => {
    const connected = await checkBackendConnection();
    if (connected) {
      setUseBackend(true);
      localStorage.setItem('backendUrl', backendUrl);
      localStorage.setItem('useBackend', 'true');
      setShowBackendSettings(false);
      alert('✅ Connected to backend server!');
    } else {
      alert('❌ Cannot connect to backend server');
    }
  };
  
  const syncFromBackend = async () => {
    if (!useBackend || !backendConnected) return;
    
    try {
      const response = await fetch(`${backendUrl}/projects`);
      if (response.ok) {
        const data = await response.json();
        setProjects(data);
        setLastSyncTime(new Date().toISOString());
      }
    } catch (error) {
      console.error('Sync error:', error);
    }
  };
  
  const saveProjectToBackend = async (project) => {
    if (!useBackend) return;
    
    try {
      await fetch(`${backendUrl}/projects/${project.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(project)
      });
    } catch (error) {
      console.error('Error saving to backend:', error);
    }
  };
  
  // Jira Functions
  const testJiraConnection = async () => {
    const url = selectedJiraUrl === 'custom' ? customJiraUrl : selectedJiraUrl;
    if (!url || !jiraConfig.email || !jiraConfig.apiToken) {
      alert('Please fill all fields');
      return;
    }
    
    setTestingConnection(true);
    
    try {
      const response = await fetch(`${backendUrl}/jira/test-connection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: url,
          email: jiraConfig.email,
          apiToken: jiraConfig.apiToken
        })
      });
      
      const data = await response.json();
      
      if (data.connected) {
        setJiraProjects(data.projects);
        setJiraConfig({ ...jiraConfig, url: url });
        alert('✅ Connection successful! Please select a project.');
      } else {
        alert(`❌ Connection failed: ${data.error}`);
      }
    } catch (error) {
      alert('Error testing connection');
    } finally {
      setTestingConnection(false);
    }
  };
  
  const connectToJira = () => {
    const url = selectedJiraUrl === 'custom' ? customJiraUrl : selectedJiraUrl;
    
    if (url && jiraConfig.email && jiraConfig.apiToken && jiraConfig.defaultProject) {
      const updatedConfig = { 
        ...jiraConfig, 
        url: url,
        connected: true 
      };
      setJiraConfig(updatedConfig);
      localStorage.setItem('jiraConfig', JSON.stringify(updatedConfig));
      alert('Successfully connected to Jira!');
      setShowJiraSettingsModal(false);
    } else {
      alert('Please fill all fields and select a project');
    }
  };
  
  const loadEpicsFromJira = async () => {
    if (!jiraConfig.connected) {
      alert('Please connect to Jira first');
      return;
    }
    
    try {
      const response = await fetch(`${backendUrl}/jira/get-epics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jiraConfig })
      });
      
      if (response.ok) {
        const data = await response.json();
        setAvailableEpics(data.epics);
        setSelectedEpics([]);
        setShowEpicSelectorModal(true);
      }
    } catch (error) {
      alert('Error loading epics from Jira');
    }
  };
  
  const importSelectedEpics = async () => {
    if (selectedEpics.length === 0) {
      alert('Please select at least one epic');
      return;
    }
    
    try {
      const response = await fetch(`${backendUrl}/jira/import-epics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          jiraConfig,
          epicKeys: selectedEpics 
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        
        const updatedProjects = projects.map(p => {
          if (p.id === selectedProject.id) {
            return {
              ...p,
              items: [...p.items, ...data.items]
            };
          }
          return p;
        });
        
        setProjects(updatedProjects);
        const updated = updatedProjects.find(p => p.id === selectedProject.id);
        if (updated) {
          setSelectedProject(updated);
          if (useBackend) {
            await saveProjectToBackend(updated);
          }
        }
        
        setShowEpicSelectorModal(false);
        alert(`✅ Imported ${selectedEpics.length} epic(s) with their children`);
      }
    } catch (error) {
      alert('Error importing epics');
    }
  };
  
  const createItemInJira = async (item) => {
    if (!jiraConfig.connected) return null;
    
    try {
      const response = await fetch(`${backendUrl}/jira/create-issue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jiraConfig,
          item: {
            ...item,
            storyPoints: item.jiraStoryPoints,
            epicName: item.jiraEpicName,
            labels: item.jiraLabels ? item.jiraLabels.split(',').map(l => l.trim()) : []
          }
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        return {
          issueKey: data.key,
          issueId: data.id,
          issueUrl: `${jiraConfig.url}/browse/${data.key}`,
          lastSynced: new Date().toISOString()
        };
      }
    } catch (error) {
      console.error('Error creating Jira issue:', error);
    }
    return null;
  };
  
  const updateJiraStatus = async (item, newStatus) => {
    if (!item.jira || !jiraConfig.connected) return;
    
    try {
      const response = await fetch(`${backendUrl}/jira/update-status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jiraConfig,
          issueKey: item.jira.issueKey,
          newStatus
        })
      });
      
      const data = await response.json();
      if (data.success) {
        console.log('✅ Jira status updated');
      }
    } catch (error) {
      console.error('Error updating Jira status:', error);
    }
  };
  
  const syncFromJira = async (item) => {
    if (!item.jira || !jiraConfig.connected) return;
    
    try {
      const response = await fetch(`${backendUrl}/jira/sync-issue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jiraConfig,
          issueKey: item.jira.issueKey
        })
      });
      
      if (response.ok) {
        const syncedData = await response.json();
        
        const updatedProjects = projects.map(p => {
          if (p.id === selectedProject.id) {
            return {
              ...p,
              items: p.items.map(i => 
                i.id === item.id 
                  ? { ...i, ...syncedData, jira: { ...i.jira, lastSynced: syncedData.lastSynced } }
                  : i
              )
            };
          }
          return p;
        });
        
        setProjects(updatedProjects);
        const updated = updatedProjects.find(p => p.id === selectedProject.id);
        if (updated) setSelectedProject(updated);
        
        alert('✅ Synced from Jira');
      }
    } catch (error) {
      alert('Error syncing from Jira');
    }
  };
  
  const syncToJira = async (item) => {
    if (!item.jira || !jiraConfig.connected) return;
    
    try {
      const response = await fetch(`${backendUrl}/jira/update-issue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jiraConfig,
          issueKey: item.jira.issueKey,
          updates: {
            summary: item.name,
            assignee: item.assignee,
            priority: item.priority,
            duedate: item.endDate,
            estimatedHours: item.estimatedHours
          }
        })
      });
      
      if (response.ok) {
        alert('✅ Synced to Jira');
      }
    } catch (error) {
      alert('Error syncing to Jira');
    }
  };
  
  // CRUD Operations
  const addProject = async () => {
    if (!newProject.name || !newProject.startDate || !newProject.endDate) {
      alert('Please fill all required fields');
      return;
    }
    
    const project = {
      ...newProject,
      id: Date.now(),
      items: []
    };
    
    const updatedProjects = [...projects, project];
    setProjects(updatedProjects);
    
    if (useBackend) {
      await saveProjectToBackend(project);
    }
    
    setNewProject({ name: '', description: '', startDate: '', endDate: '', status: 'planning' });
    setShowProjectModal(false);
  };
  
  const addItem = async () => {
    if (!newItem.name || !newItem.startDate || !newItem.endDate) {
      alert('Please fill all required fields');
      return;
    }
    
    const item = {
      id: Date.now(),
      name: newItem.name,
      type: newItem.type,
      level: newItem.type === 'epic' ? 1 : newItem.type === 'story' ? 2 : newItem.type === 'task' ? 3 : 4,
      parentId: newItem.parentId,
      children: [],
      status: newItem.status,
      priority: newItem.priority,
      startDate: newItem.startDate,
      endDate: newItem.endDate,
      assignee: newItem.assignee,
      estimatedHours: parseInt(newItem.estimatedHours) || 0,
      actualHours: 0,
      comments: [],
      jira: null
    };
    
    // Create in Jira if requested
    if (newItem.createInJira && jiraConfig.connected) {
      const jiraData = await createItemInJira({
        ...item,
        jiraEpicName: newItem.jiraEpicName,
        jiraStoryPoints: newItem.jiraStoryPoints,
        jiraLabels: newItem.jiraLabels
      });
      if (jiraData) {
        item.jira = jiraData;
      }
    }
    
    const updatedProjects = projects.map(p => {
      if (p.id === selectedProject.id) {
        const updatedItems = [...p.items, item];
        
        // Update parent's children array
        if (item.parentId) {
          updatedItems.forEach(i => {
            if (i.id === item.parentId && !i.children.includes(item.id)) {
              i.children.push(item.id);
            }
          });
        }
        
        return { ...p, items: updatedItems };
      }
      return p;
    });
    
    setProjects(updatedProjects);
    const updated = updatedProjects.find(p => p.id === selectedProject.id);
    if (updated) {
      setSelectedProject(updated);
      if (useBackend) {
        await saveProjectToBackend(updated);
      }
    }
    
    setNewItem({
      name: '',
      type: 'task',
      parentId: null,
      status: 'pending',
      priority: 'medium',
      startDate: '',
      endDate: '',
      assignee: '',
      estimatedHours: 0,
      createInJira: false,
      jiraEpicName: '',
      jiraStoryPoints: '',
      jiraLabels: ''
    });
    setShowItemModal(false);
  };
  
  const updateItemStatus = async (projectId, itemId, newStatus) => {
    const project = projects.find(p => p.id === projectId);
    const item = project?.items.find(i => i.id === itemId);
    
    const updatedProjects = projects.map(p => {
      if (p.id === projectId) {
        return {
          ...p,
          items: p.items.map(i => 
            i.id === itemId ? { ...i, status: newStatus } : i
          )
        };
      }
      return p;
    });
    
    setProjects(updatedProjects);
    
    if (item?.jira && jiraConfig.connected) {
      await updateJiraStatus(item, newStatus);
    }
    
    if (useBackend) {
      const updated = updatedProjects.find(p => p.id === projectId);
      await saveProjectToBackend(updated);
    }
  };
  
  const deleteItem = async (projectId, itemId) => {
    if (!confirm('Are you sure you want to delete this item?')) return;
    
    const updatedProjects = projects.map(p => {
      if (p.id === projectId) {
        return {
          ...p,
          items: p.items.filter(i => i.id !== itemId && i.parentId !== itemId)
        };
      }
      return p;
    });
    
    setProjects(updatedProjects);
    
    if (useBackend) {
      const updated = updatedProjects.find(p => p.id === projectId);
      await saveProjectToBackend(updated);
    }
  };
  
  const addComment = async () => {
    if (!newComment.trim() || !selectedItem) return;
    
    const comment = {
      id: Date.now(),
      text: newComment,
      author: 'User',
      timestamp: new Date().toISOString(),
      fromJira: false
    };
    
    const updatedProjects = projects.map(p => {
      if (p.id === selectedProject.id) {
        return {
          ...p,
          items: p.items.map(i => 
            i.id === selectedItem.id 
              ? { ...i, comments: [...i.comments, comment] }
              : i
          )
        };
      }
      return p;
    });
    
    setProjects(updatedProjects);
    
    // Post to Jira if requested
    if (postToJira && selectedItem.jira && jiraConfig.connected) {
      try {
        await fetch(`${backendUrl}/jira/add-comment`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jiraConfig,
            issueKey: selectedItem.jira.issueKey,
            comment: newComment
          })
        });
      } catch (error) {
        console.error('Error posting comment to Jira:', error);
      }
    }
    
    const updated = updatedProjects.find(p => p.id === selectedProject.id);
    if (updated) {
      setSelectedProject(updated);
      const updatedItem = updated.items.find(i => i.id === selectedItem.id);
      setSelectedItem(updatedItem);
      if (useBackend) {
        await saveProjectToBackend(updated);
      }
    }
    
    setNewComment('');
    setPostToJira(false);
  };
  
  // Import/Export Functions
  const exportToExcel = () => {
    if (!selectedProject) {
      alert('Please select a project first');
      return;
    }
    
    const XLSX = window.XLSX;
    const worksheet = XLSX.utils.json_to_sheet(
      selectedProject.items.map(item => ({
        Name: item.name,
        Type: item.type,
        Status: item.status,
        Priority: item.priority,
        Assignee: item.assignee,
        'Start Date': item.startDate,
        'End Date': item.endDate,
        'Estimated Hours': item.estimatedHours,
        'Actual Hours': item.actualHours,
        'Jira Key': item.jira?.issueKey || ''
      }))
    );
    
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Tasks');
    XLSX.writeFile(workbook, `${selectedProject.name}_export.xlsx`);
  };
  
  const importFromExcel = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const XLSX = window.XLSX;
      const data = new Uint8Array(e.target.result);
      const workbook = XLSX.read(data, { type: 'array' });
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      const jsonData = XLSX.utils.sheet_to_json(worksheet);
      
      const importedItems = jsonData.map((row, index) => ({
        id: Date.now() + index,
        name: row.Name || 'Unnamed',
        type: row.Type || 'task',
        level: row.Type === 'epic' ? 1 : row.Type === 'story' ? 2 : 3,
        parentId: null,
        children: [],
        status: row.Status || 'pending',
        priority: row.Priority || 'medium',
        startDate: row['Start Date'] || new Date().toISOString().split('T')[0],
        endDate: row['End Date'] || new Date().toISOString().split('T')[0],
        assignee: row.Assignee || '',
        estimatedHours: parseInt(row['Estimated Hours']) || 0,
        actualHours: parseInt(row['Actual Hours']) || 0,
        comments: [],
        jira: null
      }));
      
      const updatedProjects = projects.map(p => {
        if (p.id === selectedProject.id) {
          return { ...p, items: [...p.items, ...importedItems] };
        }
        return p;
      });
      
      setProjects(updatedProjects);
      alert(`✅ Imported ${importedItems.length} items`);
    };
    reader.readAsArrayBuffer(file);
  };
  
  // Utility Functions
  const getItemIcon = (type) => {
    const icons = { epic: '📦', story: '📖', task: '✓', subtask: '○' };
    return icons[type] || '•';
  };
  
  const isOverdue = (item) => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const endDate = new Date(item.endDate);
    endDate.setHours(0, 0, 0, 0);
    return endDate < today && item.status !== 'review';
  };
  
  const getDaysOverdue = (item) => {
    if (!isOverdue(item)) return 0;
    const today = new Date();
    const endDate = new Date(item.endDate);
    return Math.ceil((today - endDate) / (1000 * 60 * 60 * 24));
  };
  
  const getFilteredItems = () => {
    if (!selectedProject) return [];
    return selectedProject.items.filter(item => {
      if (item.type === 'epic' && !timelineFilters.showEpics) return false;
      if (item.type === 'story' && !timelineFilters.showStories) return false;
      if (item.type === 'task' && !timelineFilters.showTasks) return false;
      if (item.type === 'subtask' && !timelineFilters.showSubtasks) return false;
      return true;
    });
  };
  
  const getStatusCounts = () => {
    const allItems = projects.flatMap(p => p.items);
    return {
      total: projects.length,
      review: allItems.filter(i => i.status === 'review').length,
      inProgress: allItems.filter(i => i.status === 'in-progress').length,
      overdue: allItems.filter(i => isOverdue(i)).length
    };
  };
  
  // Render Functions
  const renderHierarchyTree = (items, parentId = null, indent = 0) => {
    const children = items.filter(item => item.parentId === parentId);
    
    return children.map(item => {
      const hasChildren = items.some(i => i.parentId === item.id);
      const isExpanded = expandedItems.has(item.id);
      
      return (
        <div key={item.id} style={{ marginLeft: `${indent * 20}px` }}>
          <div className={`hierarchy-item ${isOverdue(item) ? 'overdue' : item.type}`}>
            {hasChildren && (
              <button 
                onClick={() => {
                  const newExpanded = new Set(expandedItems);
                  if (isExpanded) {
                    newExpanded.delete(item.id);
                  } else {
                    newExpanded.add(item.id);
                  }
                  setExpandedItems(newExpanded);
                }}
                className="icon-btn"
              >
                {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </button>
            )}
            
            <span style={{ fontSize: '18px' }}>{getItemIcon(item.type)}</span>
            
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 'bold' }}>
                {item.name}
                {isOverdue(item) && (
                  <span style={{ marginLeft: '8px', fontSize: '12px', color: '#dc2626' }}>
                    ({getDaysOverdue(item)} days overdue)
                  </span>
                )}
                {item.jira && (
                  <a 
                    href={item.jira.issueUrl} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{ marginLeft: '8px', color: '#9333ea', fontSize: '12px' }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    {item.jira.issueKey}
                  </a>
                )}
              </div>
              <div style={{ fontSize: '12px', color: '#6b7280' }}>
                {item.assignee} • {item.estimatedHours}h estimated • {item.startDate} to {item.endDate}
              </div>
            </div>
            
            <select
              value={item.status}
              onChange={(e) => updateItemStatus(selectedProject.id, item.id, e.target.value)}
              className={`status-select ${item.status}`}
              onClick={(e) => e.stopPropagation()}
            >
              <option value="pending">Pending</option>
              <option value="in-progress">In Progress</option>
              <option value="review">Review</option>
            </select>
            
            <div style={{ display: 'flex', gap: '4px' }}>
              {item.jira && (
                <>
                  <button 
                    onClick={() => syncFromJira(item)}
                    className="icon-btn purple"
                    title="Sync from Jira"
                  >
                    <RefreshCw size={16} />
                  </button>
                  <button 
                    onClick={() => syncToJira(item)}
                    className="icon-btn purple"
                    title="Sync to Jira"
                  >
                    <CheckCheck size={16} />
                  </button>
                </>
              )}
              <button 
                onClick={() => {
                  setEditingItem(item);
                  setShowEditItemModal(true);
                }}
                className="icon-btn blue"
                title="Edit"
              >
                <Edit2 size={16} />
              </button>
              <button 
                onClick={() => {
                  setSelectedItem(item);
                  setShowItemDetailsModal(true);
                }}
                className="icon-btn"
                title="Details"
              >
                <MessageSquare size={16} />
              </button>
              <button 
                onClick={() => deleteItem(selectedProject.id, item.id)}
                className="icon-btn red"
                title="Delete"
              >
                <Trash2 size={16} />
              </button>
            </div>
          </div>
          
          {isExpanded && hasChildren && renderHierarchyTree(items, item.id, indent + 1)}
        </div>
      );
    });
  };
  
  const renderGanttChart = () => {
    if (!selectedProject) return null;
    
    const items = getFilteredItems();
    if (items.length === 0) return <div style={{ textAlign: 'center', padding: '48px', color: '#6b7280' }}>No items to display</div>;
    
    const startDate = new Date(Math.min(...items.map(i => new Date(i.startDate))));
    const endDate = new Date(Math.max(...items.map(i => new Date(i.endDate))));
    const totalDays = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24)) + 1;
    
    const today = new Date();
    const todayPosition = ((today - startDate) / (1000 * 60 * 60 * 24)) / totalDays * 100;
    
    return (
      <div className="gantt-container">
        <div style={{ minWidth: '800px', position: 'relative' }}>
          {items.map(item => {
            const itemStart = new Date(item.startDate);
            const itemEnd = new Date(item.endDate);
            const startPos = ((itemStart - startDate) / (1000 * 60 * 60 * 24)) / totalDays * 100;
            const width = ((itemEnd - itemStart) / (1000 * 60 * 60 * 24) + 1) / totalDays * 100;
            
            return (
              <div key={item.id} className="gantt-row">
                <div style={{ position: 'absolute', left: 0, top: 10, fontSize: '12px', fontWeight: 'bold' }}>
                  {getItemIcon(item.type)} {item.name}
                </div>
                <div 
                  className={`gantt-bar ${isOverdue(item) ? 'overdue' : item.status}`}
                  style={{ left: `${startPos}%`, width: `${width}%`, marginLeft: '150px' }}
                  title={`${item.name} (${item.startDate} to ${item.endDate})`}
                >
                  {item.status.toUpperCase()}
                </div>
              </div>
            );
          })}
          {todayPosition >= 0 && todayPosition <= 100 && (
            <div className="today-marker" style={{ left: `calc(150px + ${todayPosition}%)` }}>
              <div className="today-dot" />
            </div>
          )}
        </div>
      </div>
    );
  };
  
  const renderBurndownChart = () => {
    if (!selectedProject) return null;
    
    const items = selectedProject.items;
    const totalHours = items.reduce((sum, item) => sum + item.estimatedHours, 0);
    const completedHours = items.filter(i => i.status === 'review').reduce((sum, item) => sum + item.estimatedHours, 0);
    const inProgressHours = items.filter(i => i.status === 'in-progress').reduce((sum, item) => sum + item.estimatedHours, 0);
    const pendingHours = items.filter(i => i.status === 'pending').reduce((sum, item) => sum + item.estimatedHours, 0);
    
    return (
      <div className="chart-container">
        <h3 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '16px' }}>Work Distribution</h3>
        <div style={{ display: 'flex', gap: '24px' }}>
          <div style={{ flex: 1 }}>
            <div className="chart-label">Total Hours</div>
            <div className="chart-value" style={{ color: '#6b7280' }}>{totalHours}h</div>
          </div>
          <div style={{ flex: 1 }}>
            <div className="chart-label">Completed</div>
            <div className="chart-value" style={{ color: '#16a34a' }}>{completedHours}h</div>
            <div style={{ fontSize: '12px', color: '#6b7280' }}>
              {totalHours > 0 ? Math.round((completedHours / totalHours) * 100) : 0}%
            </div>
          </div>
          <div style={{ flex: 1 }}>
            <div className="chart-label">In Progress</div>
            <div className="chart-value" style={{ color: '#2563eb' }}>{inProgressHours}h</div>
            <div style={{ fontSize: '12px', color: '#6b7280' }}>
              {totalHours > 0 ? Math.round((inProgressHours / totalHours) * 100) : 0}%
            </div>
          </div>
          <div style={{ flex: 1 }}>
            <div className="chart-label">Pending</div>
            <div className="chart-value" style={{ color: '#6b7280' }}>{pendingHours}h</div>
            <div style={{ fontSize: '12px', color: '#6b7280' }}>
              {totalHours > 0 ? Math.round((pendingHours / totalHours) * 100) : 0}%
            </div>
          </div>
        </div>
        <div style={{ marginTop: '24px', height: '20px', display: 'flex', borderRadius: '4px', overflow: 'hidden' }}>
          {completedHours > 0 && (
            <div style={{ width: `${(completedHours / totalHours) * 100}%`, backgroundColor: '#16a34a' }} />
          )}
          {inProgressHours > 0 && (
            <div style={{ width: `${(inProgressHours / totalHours) * 100}%`, backgroundColor: '#2563eb' }} />
          )}
          {pendingHours > 0 && (
            <div style={{ width: `${(pendingHours / totalHours) * 100}%`, backgroundColor: '#6b7280' }} />
          )}
        </div>
      </div>
    );
  };
  
  const renderDashboard = () => {
    const stats = getStatusCounts();
    
    return (
      <div>
        <div className="grid grid-4" style={{ marginBottom: '24px' }}>
          <div className="stat-card blue">
            <div style={{ fontSize: '12px', fontWeight: '600', marginBottom: '8px', color: '#2563eb' }}>
              TOTAL PROJECTS
            </div>
            <div style={{ fontSize: '30px', fontWeight: 'bold', color: '#1e40af' }}>
              {stats.total}
            </div>
          </div>
          
          <div className="stat-card green">
            <div style={{ fontSize: '12px', fontWeight: '600', marginBottom: '8px', color: '#16a34a' }}>
              IN REVIEW
            </div>
            <div style={{ fontSize: '30px', fontWeight: 'bold', color: '#166534' }}>
              {stats.review}
            </div>
          </div>
          
          <div className="stat-card yellow">
            <div style={{ fontSize: '12px', fontWeight: '600', marginBottom: '8px', color: '#ca8a04' }}>
              IN PROGRESS
            </div>
            <div style={{ fontSize: '30px', fontWeight: 'bold', color: '#a16207' }}>
              {stats.inProgress}
            </div>
          </div>
          
          <div className="stat-card red">
            <div style={{ fontSize: '12px', fontWeight: '600', marginBottom: '8px', color: '#dc2626' }}>
              OVERDUE
            </div>
            <div style={{ fontSize: '30px', fontWeight: 'bold', color: '#991b1b' }}>
              {stats.overdue}
            </div>
          </div>
        </div>
        
        <div className="card">
          <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>Active Projects</h2>
          {projects.map(project => (
            <div 
              key={project.id}
              className="card"
              style={{ marginBottom: '16px', cursor: 'pointer' }}
              onClick={() => {
                setSelectedProject(project);
                setActiveView('hierarchy');
              }}
            >
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '8px' }}>
                {project.name}
              </h3>
              <p style={{ color: '#6b7280', marginBottom: '12px' }}>{project.description}</p>
              <div style={{ display: 'flex', gap: '16px', fontSize: '14px', color: '#6b7280' }}>
                <span>{project.items.length} items</span>
                <span>Start: {new Date(project.startDate).toLocaleDateString()}</span>
                <span>End: {new Date(project.endDate).toLocaleDateString()}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  const renderHierarchy = () => {
    if (!selectedProject) return <div className="card">Select a project to view items</div>;
    
    return (
      <div>
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '8px' }}>
            <div>
              <h2 style={{ fontSize: '24px', fontWeight: 'bold' }}>{selectedProject.name}</h2>
              <p style={{ color: '#6b7280' }}>{selectedProject.description}</p>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {jiraConfig.connected && (
                <button onClick={loadEpicsFromJira} className="btn btn-purple">
                  <Download size={20} />
                  Import from Jira
                </button>
              )}
              <button onClick={exportToExcel} className="btn btn-success">
                <Download size={20} />
                Export
              </button>
              <label className="btn btn-secondary" style={{ cursor: 'pointer' }}>
                <Upload size={20} />
                Import
                <input 
                  type="file" 
                  accept=".xlsx,.xls" 
                  onChange={importFromExcel}
                  style={{ display: 'none' }}
                />
              </label>
              <button onClick={() => setShowItemModal(true)} className="btn btn-primary">
                <Plus size={20} />
                Add Item
              </button>
            </div>
          </div>
          
          {selectedProject.items.length > 0 ? (
            renderHierarchyTree(selectedProject.items, null, 0)
          ) : (
            <div style={{ textAlign: 'center', padding: '48px', color: '#6b7280' }}>
              No items yet. Click "Add Item" to create your first item.
            </div>
          )}
        </div>
      </div>
    );
  };
  
  const renderTimeline = () => {
    if (!selectedProject) return <div className="card">Select a project to view timeline</div>;
    
    return (
      <div>
        <div className="filter-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 'bold' }}>Timeline View</h3>
            <div style={{ display: 'flex', gap: '16px' }}>
              <label className="checkbox-wrapper">
                <input
                  type="checkbox"
                  className="checkbox"
                  checked={timelineFilters.showEpics}
                  onChange={(e) => setTimelineFilters({ ...timelineFilters, showEpics: e.target.checked })}
                />
                <span>Epics</span>
              </label>
              <label className="checkbox-wrapper">
                <input
                  type="checkbox"
                  className="checkbox"
                  checked={timelineFilters.showStories}
                  onChange={(e) => setTimelineFilters({ ...timelineFilters, showStories: e.target.checked })}
                />
                <span>Stories</span>
              </label>
              <label className="checkbox-wrapper">
                <input
                  type="checkbox"
                  className="checkbox"
                  checked={timelineFilters.showTasks}
                  onChange={(e) => setTimelineFilters({ ...timelineFilters, showTasks: e.target.checked })}
                />
                <span>Tasks</span>
              </label>
              <label className="checkbox-wrapper">
                <input
                  type="checkbox"
                  className="checkbox"
                  checked={timelineFilters.showSubtasks}
                  onChange={(e) => setTimelineFilters({ ...timelineFilters, showSubtasks: e.target.checked })}
                />
                <span>Subtasks</span>
              </label>
            </div>
          </div>
          
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={() => setSelectedChartType('gantt')}
              className={`btn ${selectedChartType === 'gantt' ? 'btn-primary' : 'btn-outline'}`}
            >
              <Calendar size={16} />
              Gantt Chart
            </button>
            <button 
              onClick={() => setSelectedChartType('burndown')}
              className={`btn ${selectedChartType === 'burndown' ? 'btn-primary' : 'btn-outline'}`}
            >
              <TrendingUp size={16} />
              Burndown
            </button>
          </div>
        </div>
        
        <div className="card">
          {selectedChartType === 'gantt' ? renderGanttChart() : renderBurndownChart()}
        </div>
      </div>
    );
  };
  
  return (
    <div className="container">
      <div className="header">
        <div className="header-content">
          <div>
            <h1 className="app-title">
              Project Manager Pro
              {useBackend && backendConnected && (
                <span className="status-badge online">
                  <Wifi size={12} /> Team Sync
                </span>
              )}
              {useBackend && !backendConnected && (
                <span className="status-badge offline">
                  <WifiOff size={12} /> Offline
                </span>
              )}
            </h1>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button onClick={() => setShowBackendSettings(true)} className={`btn ${useBackend && backendConnected ? 'btn-success' : 'btn-secondary'}`}>
              {useBackend && backendConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
              Backend
            </button>
            {jiraConfig.connected ? (
              <button onClick={() => setShowJiraSettingsModal(true)} className="btn btn-purple">
                <Settings size={16} />
                Jira Connected
              </button>
            ) : (
              <button onClick={() => setShowJiraSettingsModal(true)} className="btn btn-purple">
                <Link2 size={16} />
                Connect Jira
              </button>
            )}
            <button onClick={() => setShowProjectModal(true)} className="btn btn-primary">
              <Plus size={16} />
              New Project
            </button>
          </div>
        </div>
      </div>
      
      <div className="main-content">
        <div className="tabs">
          <button
            onClick={() => setActiveView('dashboard')}
            className={`tab ${activeView === 'dashboard' ? 'active' : 'inactive'}`}
          >
            <BarChart3 size={18} />
            Dashboard
          </button>
          <button
            onClick={() => setActiveView('hierarchy')}
            className={`tab ${activeView === 'hierarchy' ? 'active' : 'inactive'}`}
          >
            <ChevronRight size={18} />
            Hierarchy
          </button>
          <button
            onClick={() => setActiveView('timeline')}
            className={`tab ${activeView === 'timeline' ? 'active' : 'inactive'}`}
            disabled={!selectedProject}
          >
            <Calendar size={18} />
            Timeline
          </button>
        </div>
        
        {activeView === 'dashboard' && renderDashboard()}
        {activeView === 'hierarchy' && renderHierarchy()}
        {activeView === 'timeline' && renderTimeline()}
      </div>
      
      {/* Add Project Modal */}
      {showProjectModal && (
        <div className="modal-overlay" onClick={() => setShowProjectModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>Create New Project</h2>
            <div className="form-group">
              <label className="label">Project Name *</label>
              <input
                type="text"
                value={newProject.name}
                onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                className="input"
                placeholder="Enter project name"
              />
            </div>
            <div className="form-group">
              <label className="label">Description</label>
              <textarea
                value={newProject.description}
                onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                className="textarea"
                placeholder="Enter description"
                rows="3"
              />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Start Date *</label>
                <input
                  type="date"
                  value={newProject.startDate}
                  onChange={(e) => setNewProject({ ...newProject, startDate: e.target.value })}
                  className="input"
                />
              </div>
              <div className="form-group">
                <label className="label">End Date *</label>
                <input
                  type="date"
                  value={newProject.endDate}
                  onChange={(e) => setNewProject({ ...newProject, endDate: e.target.value })}
                  className="input"
                />
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px', marginTop: '24px' }}>
              <button onClick={addProject} className="btn btn-primary" style={{ flex: 1 }}>
                Create Project
              </button>
              <button onClick={() => setShowProjectModal(false)} className="btn btn-secondary" style={{ flex: 1 }}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Add Item Modal */}
      {showItemModal && (
        <div className="modal-overlay" onClick={() => setShowItemModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>Add New Item</h2>
            
            <div className="form-group">
              <label className="label">Item Name *</label>
              <input
                type="text"
                value={newItem.name}
                onChange={(e) => setNewItem({ ...newItem, name: e.target.value })}
                className="input"
                placeholder="Enter item name"
              />
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Type</label>
                <select
                  value={newItem.type}
                  onChange={(e) => setNewItem({ ...newItem, type: e.target.value })}
                  className="select"
                >
                  <option value="epic">Epic</option>
                  <option value="story">Story</option>
                  <option value="task">Task</option>
                  <option value="subtask">Subtask</option>
                </select>
              </div>
              
              <div className="form-group">
                <label className="label">Parent</label>
                <select
                  value={newItem.parentId || ''}
                  onChange={(e) => setNewItem({ ...newItem, parentId: e.target.value ? parseInt(e.target.value) : null })}
                  className="select"
                >
                  <option value="">None (Top Level)</option>
                  {selectedProject?.items.map(item => (
                    <option key={item.id} value={item.id}>
                      {getItemIcon(item.type)} {item.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Status</label>
                <select
                  value={newItem.status}
                  onChange={(e) => setNewItem({ ...newItem, status: e.target.value })}
                  className="select"
                >
                  <option value="pending">Pending</option>
                  <option value="in-progress">In Progress</option>
                  <option value="review">Review</option>
                </select>
              </div>
              
              <div className="form-group">
                <label className="label">Priority</label>
                <select
                  value={newItem.priority}
                  onChange={(e) => setNewItem({ ...newItem, priority: e.target.value })}
                  className="select"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Start Date *</label>
                <input
                  type="date"
                  value={newItem.startDate}
                  onChange={(e) => setNewItem({ ...newItem, startDate: e.target.value })}
                  className="input"
                />
              </div>
              
              <div className="form-group">
                <label className="label">End Date *</label>
                <input
                  type="date"
                  value={newItem.endDate}
                  onChange={(e) => setNewItem({ ...newItem, endDate: e.target.value })}
                  className="input"
                />
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Assignee</label>
                <input
                  type="text"
                  value={newItem.assignee}
                  onChange={(e) => setNewItem({ ...newItem, assignee: e.target.value })}
                  className="input"
                  placeholder="Enter assignee name"
                />
              </div>
              
              <div className="form-group">
                <label className="label">Estimated Hours</label>
                <input
                  type="number"
                  value={newItem.estimatedHours}
                  onChange={(e) => setNewItem({ ...newItem, estimatedHours: e.target.value })}
                  className="input"
                  placeholder="0"
                />
              </div>
            </div>
            
            {jiraConfig.connected && (
              <div style={{ border: '1px solid #e5e7eb', borderRadius: '8px', padding: '16px', marginTop: '16px', backgroundColor: '#f9fafb' }}>
                <label className="checkbox-wrapper">
                  <input
                    type="checkbox"
                    className="checkbox"
                    checked={newItem.createInJira}
                    onChange={(e) => setNewItem({ ...newItem, createInJira: e.target.checked })}
                  />
                  <span style={{ fontWeight: 'bold' }}>Create in Jira</span>
                </label>
                
                {newItem.createInJira && (
                  <div style={{ marginTop: '12px' }}>
                    <div className="custom-field">
                      <div className="custom-field-label">Epic Name (if creating new epic)</div>
                      <input
                        type="text"
                        value={newItem.jiraEpicName}
                        onChange={(e) => setNewItem({ ...newItem, jiraEpicName: e.target.value })}
                        className="input"
                        placeholder="Leave empty to use existing epic"
                      />
                    </div>
                    
                    <div className="custom-field">
                      <div className="custom-field-label">Story Points</div>
                      <input
                        type="number"
                        value={newItem.jiraStoryPoints}
                        onChange={(e) => setNewItem({ ...newItem, jiraStoryPoints: e.target.value })}
                        className="input"
                        placeholder="Enter story points"
                      />
                    </div>
                    
                    <div className="custom-field">
                      <div className="custom-field-label">Labels (comma-separated)</div>
                      <input
                        type="text"
                        value={newItem.jiraLabels}
                        onChange={(e) => setNewItem({ ...newItem, jiraLabels: e.target.value })}
                        className="input"
                        placeholder="label1, label2, label3"
                      />
                    </div>
                    
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px' }}>
                      Note: Additional custom fields may be required by your Jira workflow when transitioning to "In Development"
                    </div>
                  </div>
                )}
              </div>
            )}
            
            <div style={{ display: 'flex', gap: '8px', marginTop: '24px' }}>
              <button onClick={addItem} className="btn btn-primary" style={{ flex: 1 }}>
                Add Item
              </button>
              <button onClick={() => setShowItemModal(false)} className="btn btn-secondary" style={{ flex: 1 }}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Item Details Modal */}
      {showItemDetailsModal && selectedItem && (
        <div className="modal-overlay" onClick={() => setShowItemDetailsModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>
              {getItemIcon(selectedItem.type)} {selectedItem.name}
            </h2>
            
            <div style={{ marginBottom: '16px', padding: '12px', backgroundColor: '#f9fafb', borderRadius: '4px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '14px' }}>
                <div><strong>Type:</strong> {selectedItem.type}</div>
                <div><strong>Status:</strong> {selectedItem.status}</div>
                <div><strong>Priority:</strong> {selectedItem.priority}</div>
                <div><strong>Assignee:</strong> {selectedItem.assignee || 'Unassigned'}</div>
                <div><strong>Estimated:</strong> {selectedItem.estimatedHours}h</div>
                <div><strong>Actual:</strong> {selectedItem.actualHours}h</div>
                <div><strong>Start:</strong> {selectedItem.startDate}</div>
                <div><strong>End:</strong> {selectedItem.endDate}</div>
              </div>
              
              {selectedItem.jira && (
                <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid #e5e7eb' }}>
                  <strong>Jira:</strong>{' '}
                  <a 
                    href={selectedItem.jira.issueUrl} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{ color: '#9333ea' }}
                  >
                    {selectedItem.jira.issueKey} <ExternalLink size={12} style={{ display: 'inline' }} />
                  </a>
                  <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
                    Last synced: {new Date(selectedItem.jira.lastSynced).toLocaleString()}
                  </div>
                </div>
              )}
            </div>
            
            <div style={{ marginBottom: '16px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '8px' }}>Comments</h3>
              <div style={{ maxHeight: '200px', overflowY: 'auto', marginBottom: '12px' }}>
                {selectedItem.comments.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '24px', color: '#6b7280', fontSize: '14px' }}>
                    No comments yet
                  </div>
                ) : (
                  selectedItem.comments.map(comment => (
                    <div 
                      key={comment.id} 
                      style={{ 
                        padding: '8px', 
                        marginBottom: '8px', 
                        backgroundColor: comment.fromJira ? '#f3e8ff' : '#f9fafb',
                        borderRadius: '4px',
                        border: '1px solid #e5e7eb'
                      }}
                    >
                      <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>
                        <strong>{comment.author}</strong> • {new Date(comment.timestamp).toLocaleString()}
                        {comment.fromJira && <span style={{ marginLeft: '4px', color: '#9333ea' }}>(from Jira)</span>}
                      </div>
                      <div style={{ fontSize: '14px' }}>{comment.text}</div>
                    </div>
                  ))
                )}
              </div>
              
              <textarea
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                className="textarea"
                placeholder="Add a comment..."
                rows="3"
              />
              
              {selectedItem.jira && jiraConfig.connected && (
                <label className="checkbox-wrapper" style={{ marginTop: '8px' }}>
                  <input
                    type="checkbox"
                    className="checkbox"
                    checked={postToJira}
                    onChange={(e) => setPostToJira(e.target.checked)}
                  />
                  <span>Post comment to Jira</span>
                </label>
              )}
              
              <button 
                onClick={addComment}
                disabled={!newComment.trim()}
                className="btn btn-primary" 
                style={{ width: '100%', marginTop: '8px' }}
              >
                Add Comment
              </button>
            </div>
            
            <button 
              onClick={() => setShowItemDetailsModal(false)} 
              className="btn btn-secondary" 
              style={{ width: '100%' }}
            >
              Close
            </button>
          </div>
        </div>
      )}
      
      {/* Backend Settings Modal */}
      {showBackendSettings && (
        <div className="modal-overlay" onClick={() => setShowBackendSettings(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>Backend Settings</h2>
            {useBackend && backendConnected ? (
              <div style={{ padding: '16px', backgroundColor: '#dcfce7', borderRadius: '8px', border: '1px solid #86efac' }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#166534' }}>✅ Connected to Backend</div>
                <div style={{ fontSize: '14px', color: '#166534' }}>
                  Server: {backendUrl}<br/>
                  Last sync: {lastSyncTime ? new Date(lastSyncTime).toLocaleString() : 'Never'}
                </div>
                <button 
                  onClick={() => {
                    setUseBackend(false);
                    setBackendConnected(false);
                    localStorage.setItem('useBackend', 'false');
                    setShowBackendSettings(false);
                  }}
                  className="btn btn-danger" 
                  style={{ marginTop: '12px', width: '100%' }}
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <div>
                <div className="form-group">
                  <label className="label">Backend Server URL</label>
                  <input
                    type="text"
                    value={backendUrl}
                    onChange={(e) => setBackendUrl(e.target.value)}
                    className="input"
                    placeholder="http://localhost:3001/api"
                  />
                </div>
                <button onClick={enableBackend} className="btn btn-success" style={{ width: '100%' }}>
                  <Wifi size={16} />
                  Connect to Backend
                </button>
              </div>
            )}
            <button 
              onClick={() => setShowBackendSettings(false)} 
              className="btn btn-secondary" 
              style={{ width: '100%', marginTop: '12px' }}
            >
              Close
            </button>
          </div>
        </div>
      )}
      
      {/* Jira Settings Modal */}
      {showJiraSettingsModal && (
        <div className="modal-overlay" onClick={() => setShowJiraSettingsModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>Jira Integration</h2>
            {jiraConfig.connected ? (
              <div style={{ padding: '16px', backgroundColor: '#dcfce7', borderRadius: '8px', border: '1px solid #86efac' }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#166534' }}>✅ Connected to Jira</div>
                <div style={{ fontSize: '14px', color: '#166534' }}>
                  URL: {jiraConfig.url}<br/>
                  Project: {jiraConfig.defaultProject}
                </div>
                <button 
                  onClick={() => {
                    setJiraConfig({ 
                      url: '', 
                      email: '', 
                      apiToken: '', 
                      defaultProject: '', 
                      connected: false,
                      customFields: {
                        storyPoints: '',
                        epicLink: '',
                        sprint: '',
                        labels: ''
                      }
                    });
                    localStorage.removeItem('jiraConfig');
                    setShowJiraSettingsModal(false);
                  }}
                  className="btn btn-danger" 
                  style={{ marginTop: '12px', width: '100%' }}
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <div>
                <div className="form-group">
                  <label className="label">Jira URL</label>
                  <select
                    value={selectedJiraUrl}
                    onChange={(e) => setSelectedJiraUrl(e.target.value)}
                    className="select"
                  >
                    <option value="">Select Jira Instance</option>
                    {predefinedJiraUrls.map(url => (
                      <option key={url.value} value={url.value}>
                        {url.label}
                      </option>
                    ))}
                  </select>
                  
                  {selectedJiraUrl === 'custom' && (
                    <input
                      type="text"
                      value={customJiraUrl}
                      onChange={(e) => setCustomJiraUrl(e.target.value)}
                      className="input"
                      placeholder="https://your-company.atlassian.net"
                      style={{ marginTop: '8px' }}
                    />
                  )}
                </div>
                
                <div className="form-group">
                  <label className="label">Email</label>
                  <input
                    type="email"
                    value={jiraConfig.email}
                    onChange={(e) => setJiraConfig({ ...jiraConfig, email: e.target.value })}
                    className="input"
                    placeholder="your.email@company.com"
                  />
                </div>
                
                <div className="form-group">
                  <label className="label">API Token (Bearer Token)</label>
                  <input
                    type="password"
                    value={jiraConfig.apiToken}
                    onChange={(e) => setJiraConfig({ ...jiraConfig, apiToken: e.target.value })}
                    className="input"
                    placeholder="Enter your Bearer token"
                  />
                </div>
                
                <button 
                  onClick={testJiraConnection}
                  disabled={testingConnection}
                  className="btn btn-primary"
                  style={{ width: '100%', marginBottom: '12px' }}
                >
                  {testingConnection ? 'Testing...' : 'Test Connection & Get Projects'}
                </button>
                
                {jiraProjects.length > 0 && (
                  <div className="form-group">
                    <label className="label">Select Project</label>
                    <select
                      value={jiraConfig.defaultProject}
                      onChange={(e) => setJiraConfig({ ...jiraConfig, defaultProject: e.target.value })}
                      className="select"
                    >
                      <option value="">Choose a project...</option>
                      {jiraProjects.map(project => (
                        <option key={project.key} value={project.key}>
                          {project.key} - {project.name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                
                <button 
                  onClick={connectToJira}
                  disabled={!jiraConfig.defaultProject}
                  className="btn btn-purple"
                  style={{ width: '100%' }}
                >
                  Connect to Jira
                </button>
              </div>
            )}
            <button 
              onClick={() => setShowJiraSettingsModal(false)} 
              className="btn btn-secondary" 
              style={{ width: '100%', marginTop: '12px' }}
            >
              Close
            </button>
          </div>
        </div>
      )}
      
      {/* Epic Selector Modal */}
      {showEpicSelectorModal && (
        <div className="modal-overlay" onClick={() => setShowEpicSelectorModal(false)}>
          <div className="modal modal-large" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>
              Select Epics to Import
            </h2>
            <p style={{ marginBottom: '16px', color: '#6b7280' }}>
              Select the epics you want to import. Their stories and tasks will be imported automatically.
            </p>
            <div className="epic-selector">
              {availableEpics.map(epic => (
                <div 
                  key={epic.key}
                  className={`epic-item ${selectedEpics.includes(epic.key) ? 'selected' : ''}`}
                  onClick={() => {
                    if (selectedEpics.includes(epic.key)) {
                      setSelectedEpics(selectedEpics.filter(k => k !== epic.key));
                    } else {
                      setSelectedEpics([...selectedEpics, epic.key]);
                    }
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selectedEpics.includes(epic.key)}
                    onChange={() => {}}
                    className="checkbox"
                  />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 'bold' }}>{epic.key}: {epic.name}</div>
                    <div style={{ fontSize: '12px', color: '#6b7280' }}>
                      Status: {epic.status} • Assignee: {epic.assignee}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: '8px', marginTop: '24px' }}>
              <button 
                onClick={importSelectedEpics}
                disabled={selectedEpics.length === 0}
                className="btn btn-primary" 
                style={{ flex: 1 }}
              >
                Import {selectedEpics.length} Epic(s)
              </button>
              <button 
                onClick={() => setShowEpicSelectorModal(false)} 
                className="btn btn-secondary" 
                style={{ flex: 1 }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProjectManager;