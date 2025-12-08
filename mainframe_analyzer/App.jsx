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
  const [showStoryImportModal, setShowStoryImportModal] = useState(false);
  const [storyImportEpicFilter, setStoryImportEpicFilter] = useState('');
  const [availableStories, setAvailableStories] = useState([]);
  const [selectedStories, setSelectedStories] = useState([]);
  const [epicSearchQuery, setEpicSearchQuery] = useState('');
  const [showExportModal, setShowExportModal] = useState(false);
  
  // Selection States
  const [selectedItem, setSelectedItem] = useState(null);
  const [editingItem, setEditingItem] = useState(null);
  const [availableEpics, setAvailableEpics] = useState([]);
  const [selectedEpics, setSelectedEpics] = useState([]);
  const [projectEpics, setProjectEpics] = useState([]); // Epics from current project
  const [selectedJiraEpic, setSelectedJiraEpic] = useState(''); // Selected epic for linking
  const [epicDropdownSearch, setEpicDropdownSearch] = useState('');
  const [showEpicDropdown, setShowEpicDropdown] = useState(false);
  
  // Calendar & Timeline
  const [calendarView, setCalendarView] = useState(false);
  const [selectedMonth, setSelectedMonth] = useState(new Date());
  const [calendarEpicFilter, setCalendarEpicFilter] = useState(''); // Filter calendar by epic
  const [hierarchyAssigneeFilter, setHierarchyAssigneeFilter] = useState(''); // Filter hierarchy by assignee
  const [holidays, setHolidays] = useState([]);
  
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
    },
    autoSyncInterval: 0 // 0 = disabled, otherwise minutes
  });
  
  const [autoSyncEnabled, setAutoSyncEnabled] = useState(false);
  
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
      // Save to localStorage when backend is disabled
      localStorage.setItem('projectManagerData', JSON.stringify(projects));
    } else if (useBackend && backendConnected && projects.length > 0) {
      // Save to backend when backend is enabled
      saveAllProjectsToBackend();
    }
  }, [projects, useBackend, backendConnected]);
  
  // Auto-sync from Jira at interval
  useEffect(() => {
    if (!autoSyncEnabled || !jiraConfig.connected || !selectedProject || jiraConfig.autoSyncInterval === 0) {
      return;
    }
    
    const intervalMs = jiraConfig.autoSyncInterval * 60 * 1000; // Convert minutes to ms
    
    console.log(`Auto-sync from Jira enabled: every ${jiraConfig.autoSyncInterval} minutes`);
    
    const syncInterval = setInterval(() => {
      console.log('Auto-syncing from Jira...');
      syncAllFromJira();
    }, intervalMs);
    
    return () => {
      console.log('Auto-sync from Jira disabled');
      clearInterval(syncInterval);
    };
  }, [autoSyncEnabled, jiraConfig.connected, jiraConfig.autoSyncInterval, selectedProject]);
  
  // Save all projects to backend
  const saveAllProjectsToBackend = async () => {
    if (!useBackend || !backendConnected) return;
    
    try {
      for (const project of projects) {
        await saveProjectToBackend(project);
      }
      setLastSyncTime(new Date().toISOString());
    } catch (error) {
      console.error('Error saving all projects:', error);
    }
  };

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
      
      // Load data from backend immediately
      await syncFromBackend();
      
      setShowBackendSettings(false);
      alert('✅ Connected to backend server! Data loaded.');
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
  
  
  const loadProjectEpics = async () => {
    if (!selectedProject) return;
    
    // Get epics from current project
    const localEpics = selectedProject.items
      .filter(item => item.type === 'epic')
      .map(epic => ({
        id: epic.id,
        key: epic.jira?.issueKey || `LOCAL-${epic.id}`,
        name: epic.name,
        isLocal: !epic.jira
      }));
    
    // If Jira connected, also get Jira epics for the project
    if (jiraConfig.connected && useBackend && backendConnected) {
      try {
        console.log('Fetching Jira epics from:', `${backendUrl}/jira/get-epics`);
        const response = await fetch(`${backendUrl}/jira/get-epics`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ jiraConfig })
        });
        
        console.log('Response status:', response.status);
        
        if (response.ok) {
          const data = await response.json();
          console.log('Received Jira epics:', data.epics?.length);
          const jiraEpics = data.epics.map(epic => ({
            id: epic.id,
            key: epic.key,
            name: epic.name,
            isLocal: false
          }));
          
          // Combine local and Jira epics, removing duplicates
          const allEpics = [...localEpics];
          jiraEpics.forEach(jiraEpic => {
            if (!allEpics.some(e => e.key === jiraEpic.key)) {
              allEpics.push(jiraEpic);
            }
          });
          
          setProjectEpics(allEpics);
        } else {
          console.error('Failed to fetch Jira epics:', response.status, response.statusText);
          const errorText = await response.text();
          console.error('Error details:', errorText);
          setProjectEpics(localEpics);
        }
      } catch (error) {
        console.error('Error loading Jira epics:', error);
        setProjectEpics(localEpics);
      }
    } else {
      setProjectEpics(localEpics);
    }
  };
  
  const loadEpicsFromJira = async (searchQuery = '') => {
    if (!jiraConfig.connected) {
      alert('Please connect to Jira first');
      return;
    }
    
    try {
      // Show loading state for initial load
      if (!searchQuery) {
        setAvailableEpics([]);
        setShowEpicSelectorModal(true);
      }
      
      const response = await fetch(`${backendUrl}/jira/get-epics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jiraConfig, searchQuery })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to load epics');
      }
      
      const data = await response.json();
      setAvailableEpics(data.epics);
      
      if (!searchQuery) {
        // Only reset selection when doing initial load
        setSelectedEpics([]);
      }
    } catch (error) {
      console.error('Error loading epics from Jira:', error);
      alert(`❌ Error loading epics: ${error.message}`);
      setShowEpicSelectorModal(false);
    }
  };
  
  const loadStoriesFromJira = async (epicKey = '') => {
    if (!jiraConfig.connected) {
      alert('Please connect to Jira first');
      return;
    }
    
    if (!epicKey) {
      alert('Please select an epic first');
      return;
    }
    
    try {
      const response = await fetch(`${backendUrl}/jira/get-stories-by-epic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jiraConfig, epicKey })
      });
      
      if (response.ok) {
        const data = await response.json();
        setAvailableStories(data.stories);
        setSelectedStories([]);
      } else {
        alert('Error loading stories from Jira');
      }
    } catch (error) {
      console.error('Error loading stories:', error);
      alert('Error loading stories from Jira');
    }
  };
  
  const openStoryImportModal = async () => {
    if (!jiraConfig.connected) {
      alert('Please connect to Jira first');
      return;
    }
    
    // Load epics for the filter dropdown
    try {
      const response = await fetch(`${backendUrl}/jira/get-epics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jiraConfig, searchQuery: '' })
      });
      
      if (response.ok) {
        const data = await response.json();
        setAvailableEpics(data.epics);
        setShowStoryImportModal(true);
        setAvailableStories([]);
        setStoryImportEpicFilter('');
      }
    } catch (error) {
      alert('Error loading epics from Jira');
    }
  };
  
  const importSelectedStories = async () => {
    if (selectedStories.length === 0) {
      alert('Please select at least one story');
      return;
    }
    
    if (!storyImportEpicFilter) {
      alert('Please select an epic first');
      return;
    }
    
    try {
      // Find the epic in the current project to link stories to it
      const epicInProject = selectedProject.items.find(
        item => item.jira && item.jira.issueKey === storyImportEpicFilter
      );
      
      if (!epicInProject) {
        alert(`Epic ${storyImportEpicFilter} not found in project. Please import the epic first.`);
        return;
      }
      
      const response = await fetch(`${backendUrl}/jira/import-epics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jiraConfig,
          epicKeys: selectedStories.map(s => s.key)
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to import stories');
      }
      
      const data = await response.json();
      
      // Process imported items and link to epic
      const processedItems = [];
      data.items.forEach(epic => {
        // For stories, add them under the epic
        if (epic.type === 'story' || epic.type === 'task') {
          const newItem = {
            ...epic,
            id: Date.now() + Math.random(),
            parentId: epicInProject.id,
            level: 2,
            children: epic.children || []
          };
          processedItems.push(newItem);
          
          // Process children (tasks/subtasks)
          if (epic.children && epic.children.length > 0) {
            epic.children.forEach(child => {
              processedItems.push({
                ...child,
                id: Date.now() + Math.random(),
                parentId: newItem.id,
                level: 3
              });
            });
          }
        }
      });
      
      const updatedProjects = projects.map(p => {
        if (p.id === selectedProject.id) {
          return {
            ...p,
            items: [...p.items, ...processedItems]
          };
        }
        return p;
      });
      
      setProjects(updatedProjects);
      setSelectedProject({
        ...selectedProject,
        items: [...selectedProject.items, ...processedItems]
      });
      
      setShowStoryImportModal(false);
      alert(`✅ Imported ${processedItems.length} item(s) under ${epicInProject.name}`);
    } catch (error) {
      console.error('Error importing stories:', error);
      alert('Error importing stories from Jira');
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
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to import epics');
      }
      
      const data = await response.json();
      
      if (!data.items || data.items.length === 0) {
        alert('⚠️ No items were imported. The epics might be empty.');
        setShowEpicSelectorModal(false);
        return;
      }
      
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
      alert(`✅ Imported ${data.items.length} item(s) from ${selectedEpics.length} epic(s)`);
    } catch (error) {
      console.error('Error importing epics:', error);
      alert(`❌ Error importing epics: ${error.message}`);
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
            epicLink: selectedJiraEpic || null, // Add the selected epic
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
  
  
  
  // Sync ALL Jira-linked items in a project from Jira
  const syncAllFromJira = async () => {
    if (!selectedProject || !jiraConfig.connected) {
      alert('Please select a project and ensure Jira is connected');
      return;
    }
    
    const jiraLinkedItems = selectedProject.items.filter(item => item.jira);
    
    if (jiraLinkedItems.length === 0) {
      alert('No Jira-linked items to sync');
      return;
    }
    
    let successCount = 0;
    let errorCount = 0;
    
    for (const item of jiraLinkedItems) {
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
          
          // Update the item with synced data
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
          
          successCount++;
        } else {
          errorCount++;
        }
      } catch (error) {
        console.error(`Error syncing ${item.jira.issueKey}:`, error);
        errorCount++;
      }
    }
    
    alert(`✅ Synced ${successCount} items from Jira${errorCount > 0 ? ` (${errorCount} failed)` : ''}`);
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
        
        console.log('✅ Synced from Jira:', item.jira.issueKey);
      }
    } catch (error) {
      console.error('Error syncing from Jira:', error);
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
    setSelectedJiraEpic(''); // Reset epic selection
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
    
    try {
      if (!window.XLSX) {
        alert('❌ Excel library not loaded. Please refresh the page.');
        return;
      }
      
      const XLSX = window.XLSX;
      
      if (selectedProject.items.length === 0) {
        alert('⚠️ No items to export');
        return;
      }
      
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
          'Jira Key': item.jira?.issueKey || '',
          'Parent Type': item.parentId ? selectedProject.items.find(i => i.id === item.parentId)?.type : ''
        }))
      );
      
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, 'Tasks');
      XLSX.writeFile(workbook, `${selectedProject.name.replace(/[^a-z0-9]/gi, '_')}_export.xlsx`);
      
      console.log('✅ Export successful');
    } catch (error) {
      console.error('Export error:', error);
      alert(`❌ Export failed: ${error.message}`);
    }
  };
  
  const downloadTemplate = () => {
    try {
      if (!window.XLSX) {
        alert('❌ Excel library not loaded. Please refresh the page.');
        return;
      }
      
      const XLSX = window.XLSX;
      
      // Create template with sample data
      const templateData = [
        {
          Name: 'Example Epic',
          Type: 'epic',
          Status: 'pending',
          Priority: 'high',
          Assignee: 'John Doe',
          'Start Date': '2024-01-01',
          'End Date': '2024-03-31',
          'Estimated Hours': 160,
          'Actual Hours': 0,
          'Jira Key': '',
          'Parent Type': ''
        },
        {
          Name: 'Example Story',
          Type: 'story',
          Status: 'in-progress',
          Priority: 'medium',
          Assignee: 'Jane Smith',
          'Start Date': '2024-01-01',
          'End Date': '2024-01-31',
          'Estimated Hours': 40,
          'Actual Hours': 10,
          'Jira Key': '',
          'Parent Type': 'epic'
        },
        {
          Name: 'Example Task',
          Type: 'task',
          Status: 'review',
          Priority: 'low',
          Assignee: 'Bob Jones',
          'Start Date': '2024-01-01',
          'End Date': '2024-01-15',
          'Estimated Hours': 8,
          'Actual Hours': 8,
          'Jira Key': '',
          'Parent Type': 'story'
        }
      ];
      
      const worksheet = XLSX.utils.json_to_sheet(templateData);
      
      // Add column widths
      worksheet['!cols'] = [
        { wch: 30 }, // Name
        { wch: 10 }, // Type
        { wch: 12 }, // Status
        { wch: 10 }, // Priority
        { wch: 15 }, // Assignee
        { wch: 12 }, // Start Date
        { wch: 12 }, // End Date
        { wch: 15 }, // Estimated Hours
        { wch: 15 }, // Actual Hours
        { wch: 12 }, // Jira Key
        { wch: 12 }  // Parent Type
      ];
      
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, 'Template');
      
      // Add instructions sheet
      const instructions = [
        { Instruction: 'HOW TO USE THIS TEMPLATE' },
        { Instruction: '' },
        { Instruction: '1. Fill in your tasks in the Template sheet' },
        { Instruction: '2. Required columns: Name, Type' },
        { Instruction: '3. Type must be: epic, story, task, or subtask' },
        { Instruction: '4. Status must be: pending, in-progress, or review' },
        { Instruction: '5. Priority must be: low, medium, or high' },
        { Instruction: '6. Dates format: YYYY-MM-DD (e.g., 2024-01-15)' },
        { Instruction: '7. Parent Type: Leave blank for top-level, or specify epic/story' },
        { Instruction: '8. Save and import into Project Manager Pro' },
        { Instruction: '' },
        { Instruction: 'COLUMN DESCRIPTIONS:' },
        { Instruction: 'Name: Task name (required)' },
        { Instruction: 'Type: epic/story/task/subtask (required)' },
        { Instruction: 'Status: pending/in-progress/review' },
        { Instruction: 'Priority: low/medium/high' },
        { Instruction: 'Assignee: Person assigned to task' },
        { Instruction: 'Start Date: Task start date' },
        { Instruction: 'End Date: Task end date' },
        { Instruction: 'Estimated Hours: Time estimate' },
        { Instruction: 'Actual Hours: Time spent' },
        { Instruction: 'Jira Key: Leave blank (filled after Jira sync)' },
        { Instruction: 'Parent Type: epic/story (for hierarchy)' }
      ];
      
      const instructionsSheet = XLSX.utils.json_to_sheet(instructions);
      instructionsSheet['!cols'] = [{ wch: 80 }];
      XLSX.utils.book_append_sheet(workbook, instructionsSheet, 'Instructions');
      
      XLSX.writeFile(workbook, 'ProjectManager_Import_Template.xlsx');
      
      console.log('✅ Template downloaded');
    } catch (error) {
      console.error('Template download error:', error);
      alert(`❌ Template download failed: ${error.message}`);
    }
  };
  
  const importFromExcel = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
      if (!window.XLSX) {
        alert('❌ Excel library not loaded. Please refresh the page.');
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const XLSX = window.XLSX;
          const data = new Uint8Array(e.target.result);
          const workbook = XLSX.read(data, { type: 'array' });
          
          if (!workbook.SheetNames || workbook.SheetNames.length === 0) {
            alert('❌ No sheets found in Excel file');
            return;
          }
          
          const worksheet = workbook.Sheets[workbook.SheetNames[0]];
          const jsonData = XLSX.utils.sheet_to_json(worksheet);
          
          if (jsonData.length === 0) {
            alert('❌ No data found in Excel file');
            return;
          }
          
          // Validate required columns
          const firstRow = jsonData[0];
          if (!firstRow.Name || !firstRow.Type) {
            alert('❌ Missing required columns: Name and Type');
            return;
          }
          
          const importedItems = jsonData.map((row, index) => {
            // Validate type
            const validTypes = ['epic', 'story', 'task', 'subtask'];
            const type = row.Type?.toLowerCase();
            if (!validTypes.includes(type)) {
              console.warn(`Invalid type "${row.Type}" for row ${index + 1}, defaulting to "task"`);
            }
            
            // Validate status
            const validStatuses = ['pending', 'in-progress', 'review'];
            const status = row.Status?.toLowerCase();
            if (status && !validStatuses.includes(status)) {
              console.warn(`Invalid status "${row.Status}" for row ${index + 1}, defaulting to "pending"`);
            }
            
            // Validate priority
            const validPriorities = ['low', 'medium', 'high'];
            const priority = row.Priority?.toLowerCase();
            if (priority && !validPriorities.includes(priority)) {
              console.warn(`Invalid priority "${row.Priority}" for row ${index + 1}, defaulting to "medium"`);
            }
            
            return {
              id: Date.now() + index,
              name: row.Name || 'Unnamed',
              type: validTypes.includes(type) ? type : 'task',
              level: type === 'epic' ? 1 : type === 'story' ? 2 : type === 'task' ? 3 : 4,
              parentId: null,
              children: [],
              status: validStatuses.includes(status) ? status : 'pending',
              priority: validPriorities.includes(priority) ? priority : 'medium',
              startDate: row['Start Date'] || new Date().toISOString().split('T')[0],
              endDate: row['End Date'] || new Date().toISOString().split('T')[0],
              assignee: row.Assignee || '',
              estimatedHours: parseInt(row['Estimated Hours']) || 0,
              actualHours: parseInt(row['Actual Hours']) || 0,
              comments: [],
              jira: null
            };
          });
          
          const updatedProjects = projects.map(p => {
            if (p.id === selectedProject.id) {
              return { ...p, items: [...p.items, ...importedItems] };
            }
            return p;
          });
          
          setProjects(updatedProjects);
          setSelectedProject({ ...selectedProject, items: [...selectedProject.items, ...importedItems] });
          
          alert(`✅ Successfully imported ${importedItems.length} items!`);
          
          // Reset file input
          event.target.value = '';
        } catch (error) {
          console.error('Import processing error:', error);
          alert(`❌ Failed to process file: ${error.message}`);
        }
      };
      
      reader.onerror = () => {
        alert('❌ Failed to read file');
      };
      
      reader.readAsArrayBuffer(file);
    } catch (error) {
      console.error('Import error:', error);
      alert(`❌ Import failed: ${error.message}`);
    }
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
              <div style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                <span>{item.name}</span>
                {isOverdue(item) && (
                  <span style={{ 
                    padding: '2px 6px', 
                    fontSize: '11px', 
                    backgroundColor: '#fee2e2', 
                    color: '#dc2626',
                    borderRadius: '4px',
                    fontWeight: 'bold'
                  }}>
                    {getDaysOverdue(item)} days overdue
                  </span>
                )}
                {item.jira ? (
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '2px 8px',
                    backgroundColor: '#f3e8ff',
                    color: '#7c3aed',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    border: '1px solid #c084fc'
                  }}>
                    <ExternalLink size={12} />
                    <a 
                      href={item.jira.issueUrl} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{ color: '#7c3aed', textDecoration: 'none' }}
                      onClick={(e) => e.stopPropagation()}
                      title="Open in Jira"
                    >
                      {item.jira.issueKey}
                    </a>
                  </span>
                ) : (
                  <span style={{
                    padding: '2px 8px',
                    backgroundColor: '#f3f4f6',
                    color: '#6b7280',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    border: '1px solid #d1d5db'
                  }}>
                    Not in Jira
                  </span>
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
  
  // Progress Chart - Overall completion percentage
  const renderProgressChart = () => {
    if (!selectedProject) return null;
    
    const items = selectedProject.items;
    const total = items.length;
    const completed = items.filter(i => i.status === 'review').length;
    const inProgress = items.filter(i => i.status === 'in-progress').length;
    const pending = items.filter(i => i.status === 'pending').length;
    const completionPercent = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    return (
      <div className="chart-container">
        <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px', textAlign: 'center' }}>Project Completion Progress</h3>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '32px' }}>
          <div style={{ position: 'relative', width: '200px', height: '200px' }}>
            <svg width="200" height="200" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="100" cy="100" r="80" fill="none" stroke="#e5e7eb" strokeWidth="20" />
              <circle cx="100" cy="100" r="80" fill="none" stroke={completionPercent > 50 ? '#2563eb' : '#eab308'} strokeWidth="20"
                strokeDasharray={`${(completionPercent / 100) * 502.4} 502.4`} strokeLinecap="round" />
            </svg>
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
              <div style={{ fontSize: '48px', fontWeight: 'bold' }}>{completionPercent}%</div>
              <div style={{ fontSize: '14px', color: '#6b7280' }}>Complete</div>
            </div>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
          <div style={{ textAlign: 'center', padding: '16px', backgroundColor: '#f3f4f6', borderRadius: '8px' }}>
            <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>TOTAL</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{total}</div>
          </div>
          <div style={{ textAlign: 'center', padding: '16px', backgroundColor: '#dcfce7', borderRadius: '8px' }}>
            <div style={{ fontSize: '12px', color: '#166534', marginBottom: '4px' }}>COMPLETED</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#16a34a' }}>{completed}</div>
          </div>
          <div style={{ textAlign: 'center', padding: '16px', backgroundColor: '#dbeafe', borderRadius: '8px' }}>
            <div style={{ fontSize: '12px', color: '#1e40af', marginBottom: '4px' }}>IN PROGRESS</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2563eb' }}>{inProgress}</div>
          </div>
          <div style={{ textAlign: 'center', padding: '16px', backgroundColor: '#fef3c7', borderRadius: '8px' }}>
            <div style={{ fontSize: '12px', color: '#92400e', marginBottom: '4px' }}>PENDING</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ca8a04' }}>{pending}</div>
          </div>
        </div>
      </div>
    );
  };
  
  
  const renderWorkloadChart = () => {
    if (!selectedProject) return null;
    const items = selectedProject.items;
    const wMap = {};
    items.forEach(item => {
      const a = item.assignee || 'Unassigned';
      if (!wMap[a]) wMap[a] = { total: 0, pending: 0, inProgress: 0, review: 0, hours: 0 };
      wMap[a].total++; wMap[a].hours += item.estimatedHours || 0;
      if (item.status === 'pending') wMap[a].pending++;
      if (item.status === 'in-progress') wMap[a].inProgress++;
      if (item.status === 'review') wMap[a].review++;
    });
    const wData = Object.entries(wMap).sort((a, b) => b[1].total - a[1].total);
    
    return (
      <div className="chart-container">
        <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px' }}>Team Workload</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {wData.map(([a, d]) => (
            <div key={a} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ width: '150px', fontSize: '14px', fontWeight: '600' }}>{a}</div>
              <div style={{ flex: 1 }}><div style={{ display: 'flex', height: '32px', borderRadius: '4px', overflow: 'hidden', backgroundColor: '#f3f4f6' }}>
                {d.review > 0 && <div style={{ width: `${(d.review/d.total)*100}%`, backgroundColor: '#16a34a', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '12px', fontWeight: 'bold' }}>{d.review}</div>}
                {d.inProgress > 0 && <div style={{ width: `${(d.inProgress/d.total)*100}%`, backgroundColor: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '12px', fontWeight: 'bold' }}>{d.inProgress}</div>}
                {d.pending > 0 && <div style={{ width: `${(d.pending/d.total)*100}%`, backgroundColor: '#ca8a04', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '12px', fontWeight: 'bold' }}>{d.pending}</div>}
              </div></div>
              <div style={{ width: '120px', textAlign: 'right' }}>
                <div style={{ fontSize: '16px', fontWeight: 'bold' }}>{d.total} items</div>
                <div style={{ fontSize: '12px', color: '#6b7280' }}>{d.hours}h</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  const renderEpicProgressChart = () => {
    if (!selectedProject) return null;
    const items = selectedProject.items;
    const epics = items.filter(i => i.type === 'epic');
    if (epics.length === 0) return (<div className="chart-container" style={{ textAlign: 'center', padding: '48px' }}>
      <div style={{ fontSize: '48px' }}>🟣</div><h3>No Epics</h3><p style={{ color: '#6b7280' }}>Create epics to see progress</p></div>);
    
    return (<div className="chart-container"><h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px' }}>Epic Progress</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {epics.map(e => {
          const c = items.filter(i => i.parentId === e.id || (items.find(x => x.id === i.parentId)?.parentId === e.id));
          const t = c.length; const co = c.filter(i => i.status === 'review').length;
          const ip = c.filter(i => i.status === 'in-progress').length; const p = c.filter(i => i.status === 'pending').length;
          const pc = t > 0 ? Math.round((co/t)*100) : 0;
          return (<div key={e.id} style={{ padding: '20px', backgroundColor: '#f9fafb', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div><div style={{ fontSize: '16px', fontWeight: 'bold' }}>{e.jira && <span style={{ color: '#7c3aed', marginRight: '8px' }}>{e.jira.issueKey}</span>}{e.name}</div>
              <div style={{ fontSize: '12px', color: '#6b7280' }}>{t} items • {e.assignee}</div></div>
              <div style={{ textAlign: 'right' }}><div style={{ fontSize: '32px', fontWeight: 'bold', color: pc === 100 ? '#16a34a' : '#1f2937' }}>{pc}%</div></div>
            </div>
            <div style={{ height: '24px', display: 'flex', borderRadius: '6px', overflow: 'hidden', backgroundColor: '#e5e7eb' }}>
              {co > 0 && <div style={{ width: `${(co/t)*100}%`, backgroundColor: '#16a34a' }} />}
              {ip > 0 && <div style={{ width: `${(ip/t)*100}%`, backgroundColor: '#2563eb' }} />}
              {p > 0 && <div style={{ width: `${(p/t)*100}%`, backgroundColor: '#ca8a04' }} />}
            </div>
            <div style={{ display: 'flex', gap: '16px', marginTop: '12px', fontSize: '12px' }}>
              <span style={{ color: '#16a34a' }}>✓ {co}</span><span style={{ color: '#2563eb' }}>⟳ {ip}</span><span style={{ color: '#ca8a04' }}>○ {p}</span>
            </div>
          </div>);
        })}
      </div>
    </div>);
  };
  
  const renderDashboard = () => {
    const stats = getStatusCounts();
    
    // Helper function to get project stats
    const getProjectStats = (project) => {
      const items = project.items;
      return {
        total: items.length,
        pending: items.filter(i => i.status === 'pending').length,
        inProgress: items.filter(i => i.status === 'in-progress').length,
        review: items.filter(i => i.status === 'review').length,
        overdue: items.filter(i => isOverdue(i)).length
      };
    };
    
    return (
      <div>
        {/* Overall Stats - Smaller */}
        <div className="grid grid-4" style={{ marginBottom: '24px' }}>
          <div className="stat-card blue" style={{ padding: '12px' }}>
            <div style={{ fontSize: '10px', fontWeight: '600', marginBottom: '4px', color: '#2563eb' }}>
              TOTAL ITEMS
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#1e40af' }}>
              {stats.total}
            </div>
          </div>
          
          <div className="stat-card green" style={{ padding: '12px' }}>
            <div style={{ fontSize: '10px', fontWeight: '600', marginBottom: '4px', color: '#16a34a' }}>
              IN REVIEW
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#166534' }}>
              {stats.review}
            </div>
          </div>
          
          <div className="stat-card yellow" style={{ padding: '12px' }}>
            <div style={{ fontSize: '10px', fontWeight: '600', marginBottom: '4px', color: '#ca8a04' }}>
              IN PROGRESS
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#a16207' }}>
              {stats.inProgress}
            </div>
          </div>
          
          <div className="stat-card red" style={{ padding: '12px' }}>
            <div style={{ fontSize: '10px', fontWeight: '600', marginBottom: '4px', color: '#dc2626' }}>
              OVERDUE
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#991b1b' }}>
              {stats.overdue}
            </div>
          </div>
        </div>
        
        {/* Project List with Stats */}
        <div className="card">
          <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>Active Projects</h2>
          {projects.map(project => {
            const projectStats = getProjectStats(project);
            return (
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
                
                {/* Project-Level Stats */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px', marginBottom: '12px' }}>
                  <div style={{ 
                    padding: '8px', 
                    backgroundColor: '#dbeafe', 
                    borderRadius: '6px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '9px', fontWeight: '600', color: '#1e40af', marginBottom: '2px' }}>
                      TOTAL
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#1e40af' }}>
                      {projectStats.total}
                    </div>
                  </div>
                  
                  <div style={{ 
                    padding: '8px', 
                    backgroundColor: '#fef3c7', 
                    borderRadius: '6px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '9px', fontWeight: '600', color: '#92400e', marginBottom: '2px' }}>
                      PENDING
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#92400e' }}>
                      {projectStats.pending}
                    </div>
                  </div>
                  
                  <div style={{ 
                    padding: '8px', 
                    backgroundColor: '#fed7aa', 
                    borderRadius: '6px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '9px', fontWeight: '600', color: '#9a3412', marginBottom: '2px' }}>
                      PROGRESS
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#9a3412' }}>
                      {projectStats.inProgress}
                    </div>
                  </div>
                  
                  <div style={{ 
                    padding: '8px', 
                    backgroundColor: '#d1fae5', 
                    borderRadius: '6px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '9px', fontWeight: '600', color: '#065f46', marginBottom: '2px' }}>
                      REVIEW
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#065f46' }}>
                      {projectStats.review}
                    </div>
                  </div>
                  
                  <div style={{ 
                    padding: '8px', 
                    backgroundColor: '#fecaca', 
                    borderRadius: '6px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '9px', fontWeight: '600', color: '#991b1b', marginBottom: '2px' }}>
                      OVERDUE
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#991b1b' }}>
                      {projectStats.overdue}
                    </div>
                  </div>
                </div>
                
                {/* Additional Info */}
                <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: '#6b7280', flexWrap: 'wrap' }}>
                  <span>
                    {project.items.filter(i => i.jira).length > 0 ? (
                      <span style={{ color: '#7c3aed', fontWeight: 'bold' }}>
                        🔗 {project.items.filter(i => i.jira).length} in Jira
                      </span>
                    ) : (
                      <span style={{ color: '#9ca3af' }}>No Jira items</span>
                    )}
                  </span>
                  <span>Start: {new Date(project.startDate).toLocaleDateString()}</span>
                  <span>End: {new Date(project.endDate).toLocaleDateString()}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };
  
  const renderHierarchy = () => {
    if (!selectedProject) return <div className="card">Select a project to view items</div>;
    
    // Get unique assignees
    const uniqueAssignees = [...new Set(selectedProject.items.map(item => item.assignee))].filter(Boolean).sort();
    
    // Filter items by assignee
    const filteredItems = hierarchyAssigneeFilter 
      ? selectedProject.items.filter(item => item.assignee === hierarchyAssigneeFilter)
      : selectedProject.items;
    
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
                <>
                  <button onClick={loadEpicsFromJira} className="btn btn-purple">
                    <Download size={20} />
                    Import Epics
                  </button>
                  <button onClick={openStoryImportModal} className="btn btn-purple" title="Import stories from existing epics in Jira">
                    <Download size={20} />
                    Import Stories
                  </button>
                  <button 
                    onClick={syncAllFromJira} 
                    className="btn btn-purple"
                    title="Sync all Jira-linked items from Jira"
                  >
                    <RefreshCw size={20} />
                    Sync All from Jira
                  </button>
                </>
              )}
              <button onClick={exportToExcel} className="btn btn-success">
                <Download size={20} />
                Export
              </button>
              <button onClick={downloadTemplate} className="btn btn-outline" title="Download Excel template for importing tasks">
                <Download size={20} />
                Template
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
              <button onClick={() => {
                loadProjectEpics();
                setShowItemModal(true);
              }} className="btn btn-primary">
                <Plus size={20} />
                Add Item
              </button>
            </div>
          </div>
          
          {/* Assignee Filter */}
          {uniqueAssignees.length > 0 && (
            <div style={{ marginBottom: '16px', display: 'flex', gap: '8px', alignItems: 'center' }}>
              <label style={{ fontSize: '14px', fontWeight: '600', color: '#6b7280' }}>
                Filter by Assignee:
              </label>
              <select
                value={hierarchyAssigneeFilter}
                onChange={(e) => setHierarchyAssigneeFilter(e.target.value)}
                className="select"
                style={{ minWidth: '200px' }}
              >
                <option value="">All Assignees ({selectedProject.items.length} items)</option>
                {uniqueAssignees.map(assignee => (
                  <option key={assignee} value={assignee}>
                    {assignee} ({selectedProject.items.filter(i => i.assignee === assignee).length} items)
                  </option>
                ))}
              </select>
              {hierarchyAssigneeFilter && (
                <button
                  onClick={() => setHierarchyAssigneeFilter('')}
                  className="icon-btn"
                  title="Clear filter"
                >
                  ✕
                </button>
              )}
              {hierarchyAssigneeFilter && (
                <span style={{ fontSize: '12px', color: '#6b7280' }}>
                  Showing {filteredItems.length} item(s) for {hierarchyAssigneeFilter}
                </span>
              )}
            </div>
          )}
          
          {filteredItems.length > 0 ? (
            renderHierarchyTree(filteredItems, null, 0)
          ) : (
            <div style={{ textAlign: 'center', padding: '48px', color: '#6b7280' }}>
              {hierarchyAssigneeFilter ? `No items assigned to ${hierarchyAssigneeFilter}` : 'No items yet. Click "Add Item" to create your first item.'}
            </div>
          )}
        </div>
      </div>
    );
  };
  
  const renderVelocityChart = () => {
    if (!selectedProject) return null;
    
    const items = selectedProject.items;
    const completedItems = items.filter(i => i.status === 'review');
    const inProgressItems = items.filter(i => i.status === 'in-progress');
    
    // Group by type
    const epicStats = {
      completed: completedItems.filter(i => i.type === 'epic').length,
      inProgress: inProgressItems.filter(i => i.type === 'epic').length,
      total: items.filter(i => i.type === 'epic').length
    };
    
    const storyStats = {
      completed: completedItems.filter(i => i.type === 'story').length,
      inProgress: inProgressItems.filter(i => i.type === 'story').length,
      total: items.filter(i => i.type === 'story').length
    };
    
    const taskStats = {
      completed: completedItems.filter(i => i.type === 'task').length,
      inProgress: inProgressItems.filter(i => i.type === 'task').length,
      total: items.filter(i => i.type === 'task').length
    };
    
    return (
      <div className="chart-container">
        <h3 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '16px' }}>Team Velocity</h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '24px' }}>
          <div>
            <div className="chart-label">Epics</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '8px' }}>
              <div className="chart-value" style={{ color: '#9333ea' }}>{epicStats.completed}</div>
              <div style={{ fontSize: '14px', color: '#6b7280' }}>/ {epicStats.total} completed</div>
            </div>
            <div style={{ height: '150px', backgroundColor: '#f3f4f6', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ 
                position: 'absolute', 
                bottom: 0, 
                width: '100%', 
                height: `${epicStats.total > 0 ? (epicStats.completed / epicStats.total) * 100 : 0}%`, 
                backgroundColor: '#9333ea',
                transition: 'height 0.3s'
              }} />
            </div>
            <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px', textAlign: 'center' }}>
              {epicStats.inProgress} in progress
            </div>
          </div>
          
          <div>
            <div className="chart-label">Stories</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '8px' }}>
              <div className="chart-value" style={{ color: '#2563eb' }}>{storyStats.completed}</div>
              <div style={{ fontSize: '14px', color: '#6b7280' }}>/ {storyStats.total} completed</div>
            </div>
            <div style={{ height: '150px', backgroundColor: '#f3f4f6', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ 
                position: 'absolute', 
                bottom: 0, 
                width: '100%', 
                height: `${storyStats.total > 0 ? (storyStats.completed / storyStats.total) * 100 : 0}%`, 
                backgroundColor: '#2563eb',
                transition: 'height 0.3s'
              }} />
            </div>
            <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px', textAlign: 'center' }}>
              {storyStats.inProgress} in progress
            </div>
          </div>
          
          <div>
            <div className="chart-label">Tasks</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '8px' }}>
              <div className="chart-value" style={{ color: '#16a34a' }}>{taskStats.completed}</div>
              <div style={{ fontSize: '14px', color: '#6b7280' }}>/ {taskStats.total} completed</div>
            </div>
            <div style={{ height: '150px', backgroundColor: '#f3f4f6', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ 
                position: 'absolute', 
                bottom: 0, 
                width: '100%', 
                height: `${taskStats.total > 0 ? (taskStats.completed / taskStats.total) * 100 : 0}%`, 
                backgroundColor: '#16a34a',
                transition: 'height 0.3s'
              }} />
            </div>
            <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px', textAlign: 'center' }}>
              {taskStats.inProgress} in progress
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  const renderStatusChart = () => {
    if (!selectedProject) return null;
    
    const items = selectedProject.items;
    const pending = items.filter(i => i.status === 'pending').length;
    const inProgress = items.filter(i => i.status === 'in-progress').length;
    const review = items.filter(i => i.status === 'review').length;
    const total = items.length;
    
    const pendingPercent = total > 0 ? (pending / total) * 100 : 0;
    const inProgressPercent = total > 0 ? (inProgress / total) * 100 : 0;
    const reviewPercent = total > 0 ? (review / total) * 100 : 0;
    
    return (
      <div className="chart-container">
        <h3 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '16px' }}>Status Distribution</h3>
        
        <div style={{ display: 'flex', gap: '48px', alignItems: 'center' }}>
          <div style={{ position: 'relative', width: '200px', height: '200px' }}>
            <svg viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="50" cy="50" r="40" fill="none" stroke="#f3f4f6" strokeWidth="20" />
              <circle 
                cx="50" 
                cy="50" 
                r="40" 
                fill="none" 
                stroke="#6b7280" 
                strokeWidth="20"
                strokeDasharray={`${pendingPercent * 2.51} ${251 - pendingPercent * 2.51}`}
                strokeDashoffset="0"
              />
              <circle 
                cx="50" 
                cy="50" 
                r="40" 
                fill="none" 
                stroke="#2563eb" 
                strokeWidth="20"
                strokeDasharray={`${inProgressPercent * 2.51} ${251 - inProgressPercent * 2.51}`}
                strokeDashoffset={`-${pendingPercent * 2.51}`}
              />
              <circle 
                cx="50" 
                cy="50" 
                r="40" 
                fill="none" 
                stroke="#16a34a" 
                strokeWidth="20"
                strokeDasharray={`${reviewPercent * 2.51} ${251 - reviewPercent * 2.51}`}
                strokeDashoffset={`-${(pendingPercent + inProgressPercent) * 2.51}`}
              />
            </svg>
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{total}</div>
              <div style={{ fontSize: '12px', color: '#6b7280' }}>Items</div>
            </div>
          </div>
          
          <div style={{ flex: 1 }}>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '16px', height: '16px', backgroundColor: '#6b7280', borderRadius: '2px' }} />
                  <span style={{ fontSize: '14px' }}>Pending</span>
                </div>
                <span style={{ fontSize: '14px', fontWeight: 'bold' }}>{pending} ({pendingPercent.toFixed(0)}%)</span>
              </div>
              <div style={{ height: '8px', backgroundColor: '#f3f4f6', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${pendingPercent}%`, height: '100%', backgroundColor: '#6b7280' }} />
              </div>
            </div>
            
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '16px', height: '16px', backgroundColor: '#2563eb', borderRadius: '2px' }} />
                  <span style={{ fontSize: '14px' }}>In Progress</span>
                </div>
                <span style={{ fontSize: '14px', fontWeight: 'bold' }}>{inProgress} ({inProgressPercent.toFixed(0)}%)</span>
              </div>
              <div style={{ height: '8px', backgroundColor: '#f3f4f6', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${inProgressPercent}%`, height: '100%', backgroundColor: '#2563eb' }} />
              </div>
            </div>
            
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '16px', height: '16px', backgroundColor: '#16a34a', borderRadius: '2px' }} />
                  <span style={{ fontSize: '14px' }}>Review/Done</span>
                </div>
                <span style={{ fontSize: '14px', fontWeight: 'bold' }}>{review} ({reviewPercent.toFixed(0)}%)</span>
              </div>
              <div style={{ height: '8px', backgroundColor: '#f3f4f6', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${reviewPercent}%`, height: '100%', backgroundColor: '#16a34a' }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  const renderCalendar = () => {
    if (!selectedProject) return <div className="card">Select a project to view calendar</div>;
    
    const currentDate = selectedMonth;
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    
    // Get first and last day of month
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();
    
    // Get epics for filter dropdown
    const projectEpics = selectedProject.items.filter(item => item.type === 'epic');
    
    // Create calendar grid
    const calendarDays = [];
    
    // Add empty cells for days before month starts
    for (let i = 0; i < startingDayOfWeek; i++) {
      calendarDays.push(null);
    }
    
    // Add days of month
    for (let day = 1; day <= daysInMonth; day++) {
      calendarDays.push(new Date(year, month, day));
    }
    
    const isToday = (date) => {
      if (!date) return false;
      const today = new Date();
      return date.getDate() === today.getDate() &&
             date.getMonth() === today.getMonth() &&
             date.getFullYear() === today.getFullYear();
    };
    
    const isWeekend = (date) => {
      if (!date) return false;
      const day = date.getDay();
      return day === 0 || day === 6;
    };
    
    // Helper to check if item has parent epic (must be before getItemsForDate)
    const hasParentEpic = (item, epicId) => {
      if (!item.parentId) return false;
      const parent = selectedProject.items.find(i => i.id === item.parentId);
      if (!parent) return false;
      
      // Handle string vs number comparison
      const epicIdNum = typeof epicId === 'string' ? parseInt(epicId) : epicId;
      const parentIdNum = typeof parent.id === 'string' ? parseInt(parent.id) : parent.id;
      
      if (parent.id === epicId || parentIdNum === epicIdNum) return true;
      return hasParentEpic(parent, epicId);
    };
    
    const getItemsForDate = (date) => {
      if (!date) return [];
      const dateStr = date.toISOString().split('T')[0];
      let items = selectedProject.items.filter(item => {
        const itemStart = item.startDate;
        const itemEnd = item.endDate;
        return dateStr >= itemStart && dateStr <= itemEnd;
      });
      
      // Filter by epic if selected
      if (calendarEpicFilter) {
        // Convert filter to number if it's a string number
        const filterIdNum = typeof calendarEpicFilter === 'string' ? 
          parseInt(calendarEpicFilter) : calendarEpicFilter;
        
        items = items.filter(item => {
          // Show the epic itself
          if (item.id === filterIdNum || item.id === calendarEpicFilter) return true;
          // Show children of the epic
          if (item.parentId === filterIdNum || item.parentId === calendarEpicFilter) return true;
          // Check parent hierarchy
          return hasParentEpic(item, filterIdNum) || hasParentEpic(item, calendarEpicFilter);
        });
      }
      
      return items;
    };
    
    const changeMonth = (delta) => {
      setSelectedMonth(new Date(year, month + delta, 1));
    };
    
    return (
      <div>
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button onClick={() => changeMonth(-1)} className="btn btn-outline">
                &larr; Previous
              </button>
              <h2 style={{ fontSize: '20px', fontWeight: 'bold' }}>
                {currentDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
              </h2>
              <button onClick={() => changeMonth(1)} className="btn btn-outline">
                Next &rarr;
              </button>
            </div>
            
            {/* Epic Filter */}
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <label style={{ fontSize: '12px', fontWeight: '600', color: '#6b7280' }}>
                Filter by Epic:
              </label>
              <select
                value={calendarEpicFilter}
                onChange={(e) => setCalendarEpicFilter(e.target.value)}
                className="select"
                style={{ minWidth: '200px' }}
              >
                <option value="">All Items</option>
                {projectEpics.map(epic => (
                  <option key={epic.id} value={epic.id}>
                    {epic.jira ? `${epic.jira.issueKey} - ` : ''}{epic.name}
                  </option>
                ))}
              </select>
              {calendarEpicFilter && (
                <button
                  onClick={() => setCalendarEpicFilter('')}
                  className="icon-btn"
                  title="Clear filter"
                >
                  ✕
                </button>
              )}
            </div>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '2px' }}>
            {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
              <div key={day} style={{ 
                padding: '6px', 
                textAlign: 'center', 
                fontWeight: 'bold', 
                fontSize: '11px',
                color: '#6b7280',
                backgroundColor: '#f9fafb'
              }}>
                {day}
              </div>
            ))}
            
            {calendarDays.map((date, index) => {
              if (!date) {
                return <div key={`empty-${index}`} style={{ minHeight: '80px', backgroundColor: '#f9fafb' }} />;
              }
              
              const items = getItemsForDate(date);
              const dayClass = isToday(date) ? 'today' : isWeekend(date) ? 'weekend' : 'normal';
              
              return (
                <div key={index} className={`calendar-day ${dayClass}`} style={{ minHeight: '80px' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '2px', fontSize: '11px' }}>
                    {date.getDate()}
                  </div>
                  <div style={{ fontSize: '9px' }}>
                    {items.slice(0, 3).map(item => (
                      <div 
                        key={item.id}
                        style={{
                          padding: '1px 3px',
                          marginBottom: '1px',
                          borderRadius: '2px',
                          backgroundColor: item.status === 'review' ? '#dcfce7' :
                                         item.status === 'in-progress' ? '#dbeafe' : '#f3f4f6',
                          color: item.status === 'review' ? '#166534' :
                                item.status === 'in-progress' ? '#1e40af' : '#374151',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          cursor: 'pointer'
                        }}
                        title={`${item.name} (${item.assignee})`}
                        onClick={() => {
                          setSelectedItem(item);
                          setShowItemDetailsModal(true);
                        }}
                      >
                        {getItemIcon(item.type)} {item.name.length > 15 ? item.name.substring(0, 15) + '...' : item.name}
                      </div>
                    ))}
                    {items.length > 3 && (
                      <div style={{ fontSize: '9px', color: '#6b7280', marginTop: '1px' }}>
                        +{items.length - 3} more
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
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
          
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
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
            <button 
              onClick={() => setSelectedChartType('progress')}
              className={`btn ${selectedChartType === 'progress' ? 'btn-primary' : 'btn-outline'}`}
            >
              📊 Progress
            </button>
            <button 
              onClick={() => setSelectedChartType('status')}
              className={`btn ${selectedChartType === 'status' ? 'btn-primary' : 'btn-outline'}`}
            >
              🥧 Status
            </button>
            <button 
              onClick={() => setSelectedChartType('workload')}
              className={`btn ${selectedChartType === 'workload' ? 'btn-primary' : 'btn-outline'}`}
            >
              👥 Workload
            </button>
            <button 
              onClick={() => setSelectedChartType('epic')}
              className={`btn ${selectedChartType === 'epic' ? 'btn-primary' : 'btn-outline'}`}
            >
              🟣 Epic Progress
            </button>
          </div>
        </div>
        
        <div className="card">
          {selectedChartType === 'gantt' && renderGanttChart()}
          {selectedChartType === 'burndown' && renderBurndownChart()}
          {selectedChartType === 'progress' && renderProgressChart()}
          {selectedChartType === 'status' && renderStatusChart()}
          {selectedChartType === 'workload' && renderWorkloadChart()}
          {selectedChartType === 'epic' && renderEpicProgressChart()}
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
            {useBackend && backendConnected && (
              <button onClick={async () => {
                await syncFromBackend();
                alert('✅ Synced from backend!');
              }} className="btn btn-outline" title="Sync from backend">
                <RefreshCw size={16} />
                Sync
              </button>
            )}
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
          <button
            onClick={() => setActiveView('calendar')}
            className={`tab ${activeView === 'calendar' ? 'active' : 'inactive'}`}
            disabled={!selectedProject}
          >
            <Calendar size={18} />
            Calendar
          </button>
        </div>
        
        {activeView === 'dashboard' && renderDashboard()}
        {activeView === 'hierarchy' && renderHierarchy()}
        {activeView === 'timeline' && renderTimeline()}
        {activeView === 'calendar' && renderCalendar()}
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
      
      {/* Edit Item Modal */}
      {showEditItemModal && editingItem && (
        <div className="modal-overlay" onClick={() => setShowEditItemModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>Edit Item</h2>
            
            <div className="form-group">
              <label className="label">Item Name *</label>
              <input
                type="text"
                value={editingItem.name}
                onChange={(e) => setEditingItem({ ...editingItem, name: e.target.value })}
                className="input"
                placeholder="Enter item name"
              />
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Type</label>
                <select
                  value={editingItem.type}
                  onChange={(e) => setEditingItem({ ...editingItem, type: e.target.value })}
                  className="select"
                >
                  <option value="epic">Epic</option>
                  <option value="story">Story</option>
                  <option value="task">Task</option>
                  <option value="subtask">Subtask</option>
                </select>
              </div>
              
              <div className="form-group">
                <label className="label">Status</label>
                <select
                  value={editingItem.status}
                  onChange={(e) => setEditingItem({ ...editingItem, status: e.target.value })}
                  className="select"
                >
                  <option value="pending">Pending</option>
                  <option value="in-progress">In Progress</option>
                  <option value="review">Review</option>
                </select>
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Priority</label>
                <select
                  value={editingItem.priority}
                  onChange={(e) => setEditingItem({ ...editingItem, priority: e.target.value })}
                  className="select"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
              
              <div className="form-group">
                <label className="label">Assignee</label>
                <input
                  type="text"
                  value={editingItem.assignee}
                  onChange={(e) => setEditingItem({ ...editingItem, assignee: e.target.value })}
                  className="input"
                  placeholder="Enter assignee name"
                />
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Start Date *</label>
                <input
                  type="date"
                  value={editingItem.startDate}
                  onChange={(e) => setEditingItem({ ...editingItem, startDate: e.target.value })}
                  className="input"
                />
              </div>
              
              <div className="form-group">
                <label className="label">End Date *</label>
                <input
                  type="date"
                  value={editingItem.endDate}
                  onChange={(e) => setEditingItem({ ...editingItem, endDate: e.target.value })}
                  className="input"
                />
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label className="label">Estimated Hours</label>
                <input
                  type="number"
                  value={editingItem.estimatedHours}
                  onChange={(e) => setEditingItem({ ...editingItem, estimatedHours: e.target.value })}
                  className="input"
                  placeholder="0"
                />
              </div>
              
              <div className="form-group">
                <label className="label">Actual Hours</label>
                <input
                  type="number"
                  value={editingItem.actualHours || 0}
                  onChange={(e) => setEditingItem({ ...editingItem, actualHours: e.target.value })}
                  className="input"
                  placeholder="0"
                />
              </div>
            </div>
            
            {editingItem.jira && jiraConfig.connected && (
              <div style={{ border: '1px solid #e5e7eb', borderRadius: '8px', padding: '16px', marginTop: '16px', backgroundColor: '#f3e8ff' }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#6b21a8' }}>
                  🔗 Linked to Jira: {editingItem.jira.issueKey}
                </div>
                <label className="checkbox-wrapper">
                  <input
                    type="checkbox"
                    className="checkbox"
                    checked={true}
                    readOnly
                  />
                  <span>Sync changes to Jira after saving</span>
                </label>
              </div>
            )}
            
            <div style={{ display: 'flex', gap: '8px', marginTop: '24px' }}>
              <button 
                onClick={async () => {
                  const updatedProjects = projects.map(p => {
                    if (p.id === selectedProject.id) {
                      return {
                        ...p,
                        items: p.items.map(i => 
                          i.id === editingItem.id ? editingItem : i
                        )
                      };
                    }
                    return p;
                  });
                  
                  setProjects(updatedProjects);
                  
                  // Sync to Jira if item is linked
                  if (editingItem.jira && jiraConfig.connected) {
                    await syncToJira(editingItem);
                  }
                  
                  const updated = updatedProjects.find(p => p.id === selectedProject.id);
                  if (updated) {
                    setSelectedProject(updated);
                    if (useBackend) {
                      await saveProjectToBackend(updated);
                    }
                  }
                  
                  setShowEditItemModal(false);
                  setEditingItem(null);
                }}
                className="btn btn-primary" 
                style={{ flex: 1 }}
              >
                Save Changes
              </button>
              <button 
                onClick={() => {
                  setShowEditItemModal(false);
                  setEditingItem(null);
                }} 
                className="btn btn-secondary" 
                style={{ flex: 1 }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Add Item Modal */}
      {showItemModal && (
        <div className="modal-overlay" onClick={() => {
          setSelectedJiraEpic('');
          setShowItemModal(false);
        }}>
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
                    {/* Epic Selection - Only show for Story, Task, Subtask */}
                    {newItem.type !== 'epic' && projectEpics.length > 0 && (
                      <div className="custom-field">
                        <div className="custom-field-label">
                          Link to Epic {newItem.type === 'story' ? '(Required for Stories)' : '(Optional)'}
                        </div>
                        
                        {/* Searchable Epic Dropdown */}
                        <div style={{ position: 'relative' }}>
                          <input
                            type="text"
                            value={epicDropdownSearch}
                            onChange={(e) => {
                              setEpicDropdownSearch(e.target.value);
                              setShowEpicDropdown(true);
                            }}
                            onFocus={() => setShowEpicDropdown(true)}
                            placeholder="🔍 Search epics by key or name..."
                            className="input"
                            style={{ width: '100%' }}
                          />
                          
                          {/* Selected Epic Display */}
                          {selectedJiraEpic && !showEpicDropdown && (
                            <div style={{
                              padding: '8px',
                              backgroundColor: '#f3f4f6',
                              borderRadius: '4px',
                              marginTop: '4px',
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center'
                            }}>
                              <span style={{ fontSize: '12px', fontWeight: 'bold' }}>
                                Selected: {projectEpics.find(e => e.key === selectedJiraEpic)?.name} ({selectedJiraEpic})
                              </span>
                              <button
                                onClick={() => {
                                  setSelectedJiraEpic('');
                                  setEpicDropdownSearch('');
                                }}
                                className="icon-btn red"
                                style={{ fontSize: '12px' }}
                              >
                                ✕
                              </button>
                            </div>
                          )}
                          
                          {/* Dropdown Results */}
                          {showEpicDropdown && (
                            <div style={{
                              position: 'absolute',
                              top: '100%',
                              left: 0,
                              right: 0,
                              maxHeight: '300px',
                              overflowY: 'auto',
                              backgroundColor: 'white',
                              border: '1px solid #d1d5db',
                              borderRadius: '4px',
                              marginTop: '4px',
                              zIndex: 1000,
                              boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                            }}>
                              {/* Close button */}
                              <div style={{
                                padding: '8px',
                                borderBottom: '1px solid #e5e7eb',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                backgroundColor: '#f9fafb'
                              }}>
                                <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#6b7280' }}>
                                  Select an Epic
                                </span>
                                <button
                                  onClick={() => {
                                    setShowEpicDropdown(false);
                                    setEpicDropdownSearch('');
                                  }}
                                  style={{
                                    border: 'none',
                                    background: 'none',
                                    cursor: 'pointer',
                                    fontSize: '16px',
                                    color: '#9ca3af'
                                  }}
                                >
                                  ✕
                                </button>
                              </div>
                              
                              {/* Filtered epics */}
                              {(() => {
                                const filteredEpics = projectEpics.filter(epic => {
                                  if (!epicDropdownSearch) return true;
                                  const query = epicDropdownSearch.toLowerCase();
                                  return epic.key.toLowerCase().includes(query) || 
                                         epic.name.toLowerCase().includes(query);
                                });
                                
                                const localEpics = filteredEpics.filter(e => e.isLocal);
                                const jiraEpics = filteredEpics.filter(e => !e.isLocal);
                                
                                return (
                                  <>
                                    {localEpics.length > 0 && (
                                      <>
                                        <div style={{
                                          padding: '8px',
                                          fontSize: '11px',
                                          fontWeight: 'bold',
                                          color: '#6b7280',
                                          backgroundColor: '#f9fafb'
                                        }}>
                                          Project Epics ({localEpics.length})
                                        </div>
                                        {localEpics.map(epic => (
                                          <div
                                            key={epic.id}
                                            onClick={() => {
                                              setSelectedJiraEpic(epic.key);
                                              setShowEpicDropdown(false);
                                              setEpicDropdownSearch('');
                                            }}
                                            style={{
                                              padding: '10px',
                                              cursor: 'pointer',
                                              borderBottom: '1px solid #f3f4f6',
                                              backgroundColor: selectedJiraEpic === epic.key ? '#dbeafe' : 'white'
                                            }}
                                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = selectedJiraEpic === epic.key ? '#dbeafe' : 'white'}
                                          >
                                            <div style={{ fontWeight: 'bold', fontSize: '13px' }}>{epic.key}</div>
                                            <div style={{ fontSize: '12px', color: '#6b7280' }}>{epic.name}</div>
                                          </div>
                                        ))}
                                      </>
                                    )}
                                    
                                    {jiraEpics.length > 0 && (
                                      <>
                                        <div style={{
                                          padding: '8px',
                                          fontSize: '11px',
                                          fontWeight: 'bold',
                                          color: '#6b7280',
                                          backgroundColor: '#f9fafb'
                                        }}>
                                          Jira Epics ({jiraEpics.length})
                                        </div>
                                        {jiraEpics.map(epic => (
                                          <div
                                            key={epic.id}
                                            onClick={() => {
                                              setSelectedJiraEpic(epic.key);
                                              setShowEpicDropdown(false);
                                              setEpicDropdownSearch('');
                                            }}
                                            style={{
                                              padding: '10px',
                                              cursor: 'pointer',
                                              borderBottom: '1px solid #f3f4f6',
                                              backgroundColor: selectedJiraEpic === epic.key ? '#dbeafe' : 'white'
                                            }}
                                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = selectedJiraEpic === epic.key ? '#dbeafe' : 'white'}
                                          >
                                            <div style={{ fontWeight: 'bold', fontSize: '13px' }}>{epic.key}</div>
                                            <div style={{ fontSize: '12px', color: '#6b7280' }}>{epic.name}</div>
                                          </div>
                                        ))}
                                      </>
                                    )}
                                    
                                    {filteredEpics.length === 0 && (
                                      <div style={{ padding: '16px', textAlign: 'center', color: '#9ca3af', fontSize: '12px' }}>
                                        {epicDropdownSearch ? `No epics found matching "${epicDropdownSearch}"` : 'No epics available'}
                                      </div>
                                    )}
                                  </>
                                );
                              })()}
                            </div>
                          )}
                        </div>
                        
                        <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
                          {selectedJiraEpic ? `Will be created under: ${selectedJiraEpic}` : 'Type to search and select an epic'}
                        </div>
                      </div>
                    )}
                    
                    {/* Epic Name - Only show when creating an Epic */}
                    {newItem.type === 'epic' && (
                      <div className="custom-field">
                        <div className="custom-field-label">Epic Name (Required for Epics)</div>
                        <input
                          type="text"
                          value={newItem.jiraEpicName}
                          onChange={(e) => setNewItem({ ...newItem, jiraEpicName: e.target.value })}
                          className="input"
                          placeholder="Enter epic name"
                        />
                      </div>
                    )}
                    
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
              <button onClick={() => {
                setSelectedJiraEpic('');
                setShowItemModal(false);
              }} className="btn btn-secondary" style={{ flex: 1 }}>
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
                <div style={{ fontSize: '14px', color: '#166534', marginBottom: '12px' }}>
                  Server: {backendUrl}<br/>
                  Last sync: {lastSyncTime ? new Date(lastSyncTime).toLocaleString() : 'Never'}<br/>
                  Projects in backend: {projects.length}
                </div>
                
                <button 
                  onClick={async () => {
                    await syncFromBackend();
                    alert('✅ Data synced from backend!');
                  }}
                  className="btn btn-primary" 
                  style={{ marginBottom: '8px', width: '100%' }}
                >
                  <RefreshCw size={16} />
                  Sync from Backend Now
                </button>
                
                <button 
                  onClick={() => {
                    setUseBackend(false);
                    setBackendConnected(false);
                    localStorage.setItem('useBackend', 'false');
                    setShowBackendSettings(false);
                  }}
                  className="btn btn-danger" 
                  style={{ width: '100%' }}
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
                <div style={{ fontSize: '14px', color: '#166534', marginBottom: '12px' }}>
                  URL: {jiraConfig.url}<br/>
                  Project: {jiraConfig.defaultProject}
                </div>
                
                {/* Auto-Sync Settings */}
                <div style={{ padding: '12px', backgroundColor: '#f0fdf4', borderRadius: '4px', marginBottom: '12px', border: '1px solid #bbf7d0' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '8px', fontSize: '14px', color: '#166534' }}>
                    Auto-Sync from Jira
                  </div>
                  
                  <label className="checkbox-wrapper">
                    <input
                      type="checkbox"
                      className="checkbox"
                      checked={autoSyncEnabled}
                      onChange={(e) => setAutoSyncEnabled(e.target.checked)}
                    />
                    <span>Enable automatic sync from Jira</span>
                  </label>
                  
                  {autoSyncEnabled && (
                    <div className="form-group" style={{ marginTop: '8px' }}>
                      <label className="label">Sync Interval (minutes)</label>
                      <select
                        value={jiraConfig.autoSyncInterval}
                        onChange={(e) => setJiraConfig({ ...jiraConfig, autoSyncInterval: parseInt(e.target.value) })}
                        className="select"
                      >
                        <option value="0">Disabled</option>
                        <option value="5">Every 5 minutes</option>
                        <option value="10">Every 10 minutes</option>
                        <option value="15">Every 15 minutes</option>
                        <option value="30">Every 30 minutes</option>
                        <option value="60">Every hour</option>
                      </select>
                      <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
                        {jiraConfig.autoSyncInterval > 0 
                          ? `Will sync all Jira-linked items every ${jiraConfig.autoSyncInterval} minutes`
                          : 'Select an interval to enable auto-sync'}
                      </div>
                    </div>
                  )}
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
                      },
                      autoSyncInterval: 0
                    });
                    setAutoSyncEnabled(false);
                    localStorage.removeItem('jiraConfig');
                    setShowJiraSettingsModal(false);
                  }}
                  className="btn btn-danger" 
                  style={{ width: '100%' }}
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
        <div className="modal-overlay" onClick={() => {
          setShowEpicSelectorModal(false);
          setEpicSearchQuery('');
        }}>
          <div className="modal modal-large" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>
              Select Epics to Import
            </h2>
            <p style={{ marginBottom: '16px', color: '#6b7280' }}>
              Select the epics you want to import. Their stories and tasks will be imported automatically.
            </p>
            
            {/* Search Input */}
            <div style={{ marginBottom: '16px' }}>
              <input
                type="text"
                placeholder="🔍 Search epics by key or name (searches all epics in Jira)..."
                value={epicSearchQuery}
                onChange={(e) => {
                  setEpicSearchQuery(e.target.value);
                  // Debounce server search
                  clearTimeout(window.epicSearchTimeout);
                  window.epicSearchTimeout = setTimeout(() => {
                    loadEpicsFromJira(e.target.value);
                  }, 500);
                }}
                className="input"
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
                Showing {availableEpics.length} epic(s)
              </div>
            </div>
            
            {/* Epic List */}
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
              {availableEpics.length === 0 && (
                <div style={{ textAlign: 'center', padding: '24px', color: '#9ca3af' }}>
                  {epicSearchQuery ? `No epics found matching "${epicSearchQuery}"` : 'No epics available'}
                </div>
              )}
            </div>
            
            <div style={{ display: 'flex', gap: '8px', marginTop: '24px' }}>
              <button 
                onClick={importSelectedEpics}
                disabled={selectedEpics.length === 0}
                className="btn btn-primary" 
                style={{ flex: 1 }}
              >
                Import {selectedEpics.length} Epic(s) + Stories
              </button>
              <button 
                onClick={() => {
                  setShowEpicSelectorModal(false);
                  setEpicSearchQuery('');
                }} 
                className="btn btn-secondary" 
                style={{ flex: 1 }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Story Import Modal */}
      {showStoryImportModal && (
        <div className="modal-overlay" onClick={() => setShowStoryImportModal(false)}>
          <div className="modal modal-large" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>
              Import Stories from Jira
            </h2>
            <p style={{ marginBottom: '16px', color: '#6b7280' }}>
              Select an epic to see its stories, then choose which stories to import.
            </p>
            
            {/* Epic Filter */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{ fontSize: '14px', fontWeight: '600', marginBottom: '8px', display: 'block' }}>
                Select Epic
              </label>
              <select
                value={storyImportEpicFilter}
                onChange={(e) => {
                  setStoryImportEpicFilter(e.target.value);
                  if (e.target.value) {
                    loadStoriesFromJira(e.target.value);
                  } else {
                    setAvailableStories([]);
                    setSelectedStories([]);
                  }
                }}
                className="select"
                style={{ width: '100%' }}
              >
                <option value="">-- Select an Epic --</option>
                {availableEpics.map(epic => (
                  <option key={epic.key} value={epic.key}>
                    {epic.key} - {epic.name}
                  </option>
                ))}
              </select>
            </div>
            
            {/* Story List */}
            {storyImportEpicFilter && (
              <>
                <div style={{ marginBottom: '8px', fontSize: '14px', fontWeight: '600' }}>
                  Stories in {storyImportEpicFilter} ({availableStories.length})
                </div>
                
                <div className="epic-selector" style={{ maxHeight: '400px' }}>
                  {availableStories.map(story => (
                    <div 
                      key={story.key}
                      className={`epic-item ${selectedStories.some(s => s.key === story.key) ? 'selected' : ''}`}
                      onClick={() => {
                        if (selectedStories.some(s => s.key === story.key)) {
                          setSelectedStories(selectedStories.filter(s => s.key !== story.key));
                        } else {
                          setSelectedStories([...selectedStories, story]);
                        }
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={selectedStories.some(s => s.key === story.key)}
                        onChange={() => {}}
                        className="checkbox"
                      />
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 'bold' }}>
                          {story.type === 'story' ? '📘' : '✓'} {story.key}: {story.name}
                        </div>
                        <div style={{ fontSize: '12px', color: '#6b7280' }}>
                          Type: {story.type} • Status: {story.status} • Assignee: {story.assignee}
                        </div>
                        <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '2px' }}>
                          Created: {new Date(story.created).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  ))}
                  {availableStories.length === 0 && (
                    <div style={{ textAlign: 'center', padding: '24px', color: '#9ca3af' }}>
                      {storyImportEpicFilter ? 'No stories found in this epic' : 'Select an epic to see stories'}
                    </div>
                  )}
                </div>
                
                {/* Select All / None */}
                {availableStories.length > 0 && (
                  <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
                    <button 
                      onClick={() => setSelectedStories(availableStories)}
                      className="btn btn-outline"
                      style={{ fontSize: '12px', padding: '4px 12px' }}
                    >
                      Select All
                    </button>
                    <button 
                      onClick={() => setSelectedStories([])}
                      className="btn btn-outline"
                      style={{ fontSize: '12px', padding: '4px 12px' }}
                    >
                      Select None
                    </button>
                  </div>
                )}
              </>
            )}
            
            <div style={{ display: 'flex', gap: '8px', marginTop: '24px' }}>
              <button 
                onClick={importSelectedStories}
                disabled={selectedStories.length === 0 || !storyImportEpicFilter}
                className="btn btn-primary" 
                style={{ flex: 1 }}
              >
                Import {selectedStories.length} Story(ies)
              </button>
              <button 
                onClick={() => {
                  setShowStoryImportModal(false);
                  setStoryImportEpicFilter('');
                  setAvailableStories([]);
                  setSelectedStories([]);
                }} 
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