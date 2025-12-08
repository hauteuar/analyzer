const express = require('express');
const cors = require('cors');
const axios = require('axios');
const Holidays = require('date-holidays');
const path = require('path');
const fs = require('fs');

// Simple file-based database (synchronous for simplicity)
const DB_FILE = 'db.json';

// Initialize database
let db = { projects: [], syncLog: [] };
if (fs.existsSync(DB_FILE)) {
  try {
    db = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
  } catch (error) {
    console.error('Error reading database:', error);
  }
}

// Helper functions for database operations
const saveDB = () => {
  fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));
};

const getProjects = () => db.projects || [];
const setProjects = (projects) => {
  db.projects = projects;
  saveDB();
};

const addSyncLog = (log) => {
  if (!db.syncLog) db.syncLog = [];
  db.syncLog.push(log);
  // Keep only last 100 logs
  if (db.syncLog.length > 100) {
    db.syncLog = db.syncLog.slice(-100);
  }
  saveDB();
};

const getSyncLog = () => db.syncLog || [];

const app = express();

// Initialize holidays
const hd = new Holidays('US');

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    nodeVersion: process.version
  });
});

// Get holidays endpoint
app.get('/api/holidays/:year', (req, res) => {
  const year = parseInt(req.params.year);
  const holidays = hd.getHolidays(year);
  
  const formattedHolidays = holidays.map(holiday => ({
    date: holiday.date.split(' ')[0],
    name: holiday.name,
    type: holiday.type
  }));
  
  res.json(formattedHolidays);
});

// Projects CRUD endpoints
app.get('/api/projects', (req, res) => {
  try {
    const projects = getProjects();
    res.json(projects || []);
  } catch (error) {
    console.error('Error getting projects:', error);
    res.status(500).json({ error: 'Failed to get projects' });
  }
});

app.put('/api/projects/:id', (req, res) => {
  try {
    const projectId = parseInt(req.params.id);
    const project = req.body;
    
    const projects = getProjects();
    const existingIndex = projects.findIndex(p => p.id === projectId);
    
    if (existingIndex !== -1) {
      projects[existingIndex] = project;
    } else {
      projects.push(project);
    }
    
    setProjects(projects);
    res.json(project);
  } catch (error) {
    console.error('Error saving project:', error);
    res.status(500).json({ error: 'Failed to save project' });
  }
});

app.delete('/api/projects/:id', (req, res) => {
  try {
    const projectId = parseInt(req.params.id);
    const projects = getProjects();
    const filteredProjects = projects.filter(p => p.id !== projectId);
    setProjects(filteredProjects);
    res.json({ deleted: true, id: projectId });
  } catch (error) {
    console.error('Error deleting project:', error);
    res.status(500).json({ error: 'Failed to delete project' });
  }
});

// Jira Integration Endpoints

// Test connection and get projects
app.post('/api/jira/test-connection', async (req, res) => {
  const { url, email, apiToken } = req.body;
  
  try {
    const response = await axios.get(
      `${url}/rest/api/2/project`,
      {
        headers: {
          'Authorization': `Bearer ${apiToken}`,
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      }
    );
    
    const projects = response.data.map(project => ({
      key: project.key,
      name: project.name,
      id: project.id
    }));
    
    res.json({ 
      connected: true, 
      projects: projects 
    });
  } catch (error) {
    console.error('Jira connection error:', error.message);
    res.status(400).json({ 
      connected: false, 
      error: error.response?.data?.message || error.message 
    });
  }
});

// Create Jira issue
app.post('/api/jira/create-issue', async (req, res) => {
  const { jiraConfig, item } = req.body;
  
  try {
    // Map our types to Jira issue types (with proper capitalization)
    const issueTypeMap = {
      'epic': 'Epic',
      'story': 'Story',
      'task': 'Task',
      'subtask': 'Sub-task',
      'bug': 'Bug'
    };
    
    const jiraIssueType = issueTypeMap[item.type.toLowerCase()] || 'Task';
    
    // Map our priorities to Jira priorities
    const priorityMap = {
      'critical': 'Highest',
      'high': 'High',
      'medium': 'Medium',
      'low': 'Low'
    };
    
    const jiraPriority = priorityMap[item.priority.toLowerCase()] || 'Medium';
    
    // Base issue data (required fields only)
    const issueData = {
      fields: {
        project: {
          key: jiraConfig.defaultProject
        },
        summary: item.name,
        description: `Created from Project Manager Pro\n\nPriority: ${item.priority}\nEstimated Hours: ${item.estimatedHours || 0}\nAssignee: ${item.assignee || 'Unassigned'}`,
        issuetype: {
          name: jiraIssueType
        },
        priority: {
          name: jiraPriority
        }
      }
    };

    // Add due date if provided
    if (item.endDate) {
      issueData.fields.duedate = item.endDate;
    }
    
    // Add labels if provided
    if (item.labels && Array.isArray(item.labels) && item.labels.length > 0) {
      issueData.fields.labels = item.labels;
    }
    
    // Parent Link - for Sub-tasks (this is usually allowed)
    if (item.parentKey && jiraIssueType === 'Sub-task') {
      issueData.fields.parent = { key: item.parentKey };
    }

    // Try to create issue with optional custom fields
    let response;
    try {
      // Add optional custom fields
      const issueDataWithCustomFields = { ...issueData };
      
      // Epic Name for Epics
      if (item.epicName && jiraIssueType === 'Epic') {
        issueDataWithCustomFields.fields.customfield_10011 = item.epicName;
      }
      
      // Epic Link for Stories/Tasks
      if (item.epicLink && jiraIssueType !== 'Epic') {
        issueDataWithCustomFields.fields.customfield_10014 = item.epicLink;
      }
      
      // Story Points
      if (item.storyPoints) {
        issueDataWithCustomFields.fields.customfield_10016 = parseFloat(item.storyPoints);
      }

      response = await axios.post(
        `${jiraConfig.url}/rest/api/2/issue`,
        issueDataWithCustomFields,
        {
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          }
        }
      );
    } catch (customFieldError) {
      // If custom fields fail, try without them
      console.log('Custom fields not supported on create screen, creating without them...');
      
      response = await axios.post(
        `${jiraConfig.url}/rest/api/2/issue`,
        issueData,
        {
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          }
        }
      );
      
      // Try to update with custom fields after creation
      if (response.data.key) {
        const updateData = { fields: {} };
        
        if (item.epicName && jiraIssueType === 'Epic') {
          updateData.fields.customfield_10011 = item.epicName;
        }
        
        if (item.epicLink && jiraIssueType !== 'Epic') {
          updateData.fields.customfield_10014 = item.epicLink;
        }
        
        if (item.storyPoints) {
          updateData.fields.customfield_10016 = parseFloat(item.storyPoints);
        }
        
        if (Object.keys(updateData.fields).length > 0) {
          try {
            await axios.put(
              `${jiraConfig.url}/rest/api/2/issue/${response.data.key}`,
              updateData,
              {
                headers: {
                  'Authorization': `Bearer ${jiraConfig.apiToken}`,
                  'Accept': 'application/json',
                  'Content-Type': 'application/json'
                }
              }
            );
            console.log('Successfully updated custom fields after creation');
          } catch (updateError) {
            console.log('Could not update custom fields, but issue created successfully');
          }
        }
      }
    }

    res.json({
      key: response.data.key,
      id: response.data.id,
      self: response.data.self
    });
  } catch (error) {
    console.error('Error creating Jira issue:', error.response?.data || error.message);
    res.status(400).json({ 
      error: error.response?.data?.errors || error.message 
    });
  }
});

// Get epics from Jira for selective import
app.post('/api/jira/get-epics', async (req, res) => {
  const { jiraConfig, searchQuery = '' } = req.body;
  
  try {
    // Try Greenhopper API first (Jira Agile / BofA Jira 3)
    let response;
    let epics = [];
    
    try {
      // Greenhopper API endpoint with search
      response = await axios.get(
        `${jiraConfig.url}/rest/greenhopper/1.0/epics`,
        {
          params: {
            searchQuery: searchQuery,  // Search query for server-side filtering
            projectKey: jiraConfig.defaultProject,
            maxResults: 1000,  // Increased to get more results
            hideDone: false
          },
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json'
          }
        }
      );
      
      // Greenhopper API response format
      if (response.data.epicLists) {
        const allEpics = [];
        response.data.epicLists.forEach(list => {
          if (list.epicNames) {
            list.epicNames.forEach(epic => {
              allEpics.push({
                key: epic.key,
                id: epic.epicId || epic.key,
                name: epic.name,
                done: epic.isDone || false,
                created: epic.created || new Date().toISOString() // Use epic created date if available
              });
            });
          }
        });
        // Sort by created date, newest first
        epics = allEpics.sort((a, b) => new Date(b.created) - new Date(a.created));
      }
    } catch (greenhopperError) {
      console.log('Greenhopper API failed, trying standard Jira API v2...');
      
      // Fallback to standard Jira API v2 with JQL search
      let jql = `project = ${jiraConfig.defaultProject} AND issuetype = Epic`;
      
      // Add search filter to JQL if provided
      if (searchQuery && searchQuery.trim()) {
        jql += ` AND (summary ~ "${searchQuery}" OR key ~ "${searchQuery}")`;
      }
      
      jql += ` ORDER BY created DESC`;
      
      response = await axios.get(
        `${jiraConfig.url}/rest/api/2/search`,
        {
          params: {
            jql: jql,
            maxResults: 1000,  // Increased to get more results
            fields: 'summary,status,priority,assignee,created,updated'
          },
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json'
          }
        }
      );

      epics = response.data.issues.map(issue => ({
        key: issue.key,
        id: issue.id,
        name: issue.fields.summary,
        status: issue.fields.status.name,
        assignee: issue.fields.assignee?.displayName || 'Unassigned',
        created: issue.fields.created,
        hasChildren: false
      }));
    }

    res.json({ epics });
  } catch (error) {
    console.error('Error getting epics from Jira:', error.message);
    console.error('Error details:', error.response?.data);
    res.status(400).json({ 
      error: error.response?.data?.message || error.message 
    });
  }
});

// Get stories by epic from Jira
app.post('/api/jira/get-stories-by-epic', async (req, res) => {
  const { jiraConfig, epicKey } = req.body;
  
  try {
    let stories = [];
    
    // Try Greenhopper custom field first (BofA Jira)
    try {
      const jql = `cf[10014] = ${epicKey} AND issuetype in (Story, Task) ORDER BY created DESC`;
      
      const response = await axios.get(
        `${jiraConfig.url}/rest/api/2/search`,
        {
          params: {
            jql: jql,
            maxResults: 1000,
            fields: 'summary,status,priority,assignee,created,updated,issuetype'
          },
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json'
          }
        }
      );
      
      stories = response.data.issues.map(issue => ({
        key: issue.key,
        id: issue.id,
        name: issue.fields.summary,
        type: issue.fields.issuetype.name.toLowerCase(),
        status: issue.fields.status.name,
        assignee: issue.fields.assignee?.displayName || 'Unassigned',
        created: issue.fields.created,
        updated: issue.fields.updated
      }));
    } catch (greenhopperError) {
      console.log('Greenhopper custom field failed, trying standard Epic Link...');
      
      // Fallback to standard Epic Link field
      const jql = `"Epic Link" = ${epicKey} AND issuetype in (Story, Task) ORDER BY created DESC`;
      
      const response = await axios.get(
        `${jiraConfig.url}/rest/api/2/search`,
        {
          params: {
            jql: jql,
            maxResults: 1000,
            fields: 'summary,status,priority,assignee,created,updated,issuetype'
          },
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json'
          }
        }
      );
      
      stories = response.data.issues.map(issue => ({
        key: issue.key,
        id: issue.id,
        name: issue.fields.summary,
        type: issue.fields.issuetype.name.toLowerCase(),
        status: issue.fields.status.name,
        assignee: issue.fields.assignee?.displayName || 'Unassigned',
        created: issue.fields.created,
        updated: issue.fields.updated
      }));
    }
    
    res.json({ stories, epicKey });
  } catch (error) {
    console.error('Error getting stories from Jira:', error.message);
    console.error('Error details:', error.response?.data);
    res.status(400).json({ 
      error: error.response?.data?.message || error.message 
    });
  }
});

// Import selected epics and their children
app.post('/api/jira/import-epics', async (req, res) => {
  const { jiraConfig, epicKeys } = req.body;
  
  try {
    const allItems = [];
    
    for (const epicKey of epicKeys) {
      const epicResponse = await axios.get(
        `${jiraConfig.url}/rest/api/2/issue/${epicKey}`,
        {
          params: {
            fields: 'summary,status,priority,assignee,duedate,created,updated,timetracking'
          },
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json'
          }
        }
      );
      
      const epic = epicResponse.data;
      const epicItem = formatJiraIssue(epic, 'epic');
      allItems.push(epicItem);
      
      // Try to get children using multiple methods (for different Jira versions)
      // Method 1: Greenhopper/BofA Jira custom field
      let childrenResponse;
      try {
        const greenhopperJql = `cf[10014] = ${epicKey} OR parent = ${epicKey}`;
        childrenResponse = await axios.get(
          `${jiraConfig.url}/rest/api/2/search`,
          {
            params: {
              jql: greenhopperJql,
              maxResults: 100,
              fields: 'summary,status,priority,assignee,duedate,issuetype,created,updated,timetracking,customfield_10014'
            },
            headers: {
              'Authorization': `Bearer ${jiraConfig.apiToken}`,
              'Accept': 'application/json'
            }
          }
        );
      } catch (error) {
        // Method 2: Standard Jira Epic Link field name
        console.log('Greenhopper query failed, trying standard Epic Link...');
        const standardJql = `"Epic Link" = ${epicKey} OR parent = ${epicKey}`;
        childrenResponse = await axios.get(
          `${jiraConfig.url}/rest/api/2/search`,
          {
            params: {
              jql: standardJql,
              maxResults: 100,
              fields: 'summary,status,priority,assignee,duedate,issuetype,created,updated,timetracking'
            },
            headers: {
              'Authorization': `Bearer ${jiraConfig.apiToken}`,
              'Accept': 'application/json'
            }
          }
        );
      }
      
      childrenResponse.data.issues.forEach(child => {
        const childItem = formatJiraIssue(child, null, epicItem.id);
        epicItem.children.push(childItem.id);
        allItems.push(childItem);
      });
    }
    
    res.json({ items: allItems });
  } catch (error) {
    console.error('Error importing epics from Jira:', error.message);
    res.status(400).json({ 
      error: error.response?.data?.message || error.message 
    });
  }
});

// Update Jira issue status
app.post('/api/jira/update-status', async (req, res) => {
  const { jiraConfig, issueKey, newStatus } = req.body;
  
  try {
    const transitionsResponse = await axios.get(
      `${jiraConfig.url}/rest/api/2/issue/${issueKey}/transitions`,
      {
        headers: {
          'Authorization': `Bearer ${jiraConfig.apiToken}`,
          'Accept': 'application/json'
        }
      }
    );

    const statusMap = {
      'pending': ['To Do', 'Open', 'Backlog', 'New'],
      'in-progress': ['In Progress', 'Start Progress', 'In Development', 'Developing'],
      'review': ['Done', 'Resolved', 'Closed', 'Complete', 'Review']
    };

    const targetStatuses = statusMap[newStatus] || [];
    const transition = transitionsResponse.data.transitions.find(t => 
      targetStatuses.some(status => 
        t.name.toLowerCase().includes(status.toLowerCase()) ||
        status.toLowerCase().includes(t.name.toLowerCase())
      )
    );

    if (transition) {
      await axios.post(
        `${jiraConfig.url}/rest/api/2/issue/${issueKey}/transitions`,
        {
          transition: { id: transition.id }
        },
        {
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          }
        }
      );

      logSync('status_update', issueKey, 'success');
      res.json({ success: true, transitionName: transition.name });
    } else {
      logSync('status_update', issueKey, 'no_transition');
      res.json({ success: false, message: 'No matching transition found' });
    }
  } catch (error) {
    logSync('status_update', req.body.issueKey, 'error', error.message);
    console.error('Error updating Jira status:', error.message);
    res.status(400).json({ 
      error: error.response?.data?.message || error.message 
    });
  }
});

// Add comment to Jira
app.post('/api/jira/add-comment', async (req, res) => {
  const { jiraConfig, issueKey, comment } = req.body;
  
  try {
    const response = await axios.post(
      `${jiraConfig.url}/rest/api/2/issue/${issueKey}/comment`,
      {
        body: comment
      },
      {
        headers: {
          'Authorization': `Bearer ${jiraConfig.apiToken}`,
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      }
    );

    logSync('comment_add', issueKey, 'success');
    res.json({ 
      success: true, 
      commentId: response.data.id,
      created: response.data.created 
    });
  } catch (error) {
    logSync('comment_add', req.body.issueKey, 'error', error.message);
    console.error('Error adding Jira comment:', error.message);
    res.status(400).json({ 
      error: error.response?.data?.message || error.message 
    });
  }
});

// Update Jira issue fields
app.post('/api/jira/update-issue', async (req, res) => {
  const { jiraConfig, issueKey, updates } = req.body;
  
  try {
    const fieldsToUpdate = {};
    
    if (updates.assignee) {
      const userResponse = await axios.get(
        `${jiraConfig.url}/rest/api/2/user/search`,
        {
          params: { query: updates.assignee },
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json'
          }
        }
      );
      
      if (userResponse.data.length > 0) {
        fieldsToUpdate.assignee = { accountId: userResponse.data[0].accountId };
      }
    }
    
    if (updates.priority) {
      fieldsToUpdate.priority = { name: updates.priority };
    }
    
    if (updates.duedate) {
      fieldsToUpdate.duedate = updates.duedate;
    }
    
    if (updates.summary) {
      fieldsToUpdate.summary = updates.summary;
    }
    
    if (updates.estimatedHours !== undefined) {
      fieldsToUpdate.timetracking = {
        originalEstimate: `${updates.estimatedHours}h`
      };
    }

    await axios.put(
      `${jiraConfig.url}/rest/api/2/issue/${issueKey}`,
      { fields: fieldsToUpdate },
      {
        headers: {
          'Authorization': `Bearer ${jiraConfig.apiToken}`,
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      }
    );

    logSync('issue_update', issueKey, 'success');
    res.json({ success: true });
  } catch (error) {
    logSync('issue_update', req.body.issueKey, 'error', error.message);
    console.error('Error updating Jira issue:', error.message);
    res.status(400).json({ 
      error: error.response?.data?.message || error.message 
    });
  }
});

// Sync from Jira
app.post('/api/jira/sync-issue', async (req, res) => {
  const { jiraConfig, issueKey } = req.body;
  
  try {
    const response = await axios.get(
      `${jiraConfig.url}/rest/api/2/issue/${issueKey}`,
      {
        params: {
          fields: 'summary,status,priority,assignee,duedate,issuetype,created,updated,comment,timetracking'
        },
        headers: {
          'Authorization': `Bearer ${jiraConfig.apiToken}`,
          'Accept': 'application/json'
        }
      }
    );

    const issue = response.data;
    const syncedData = formatJiraIssueForSync(issue);
    
    logSync('sync_from_jira', issueKey, 'success');
    res.json(syncedData);
  } catch (error) {
    logSync('sync_from_jira', req.body.issueKey, 'error', error.message);
    console.error('Error syncing from Jira:', error.message);
    res.status(400).json({ 
      error: error.response?.data?.message || error.message 
    });
  }
});

// Helper functions
function formatJiraIssue(issue, typeOverride = null, parentId = null) {
  const fields = issue.fields;
  const typeMap = {
    'Epic': 'epic',
    'Story': 'story',
    'Task': 'task',
    'Sub-task': 'subtask',
    'Bug': 'task'
  };

  const statusMap = {
    'To Do': 'pending',
    'Open': 'pending',
    'New': 'pending',
    'Backlog': 'pending',
    'In Progress': 'in-progress',
    'In Development': 'in-progress',
    'Done': 'review',
    'Resolved': 'review',
    'Closed': 'review'
  };

  const type = typeOverride || typeMap[fields.issuetype?.name] || 'task';
  const level = type === 'epic' ? 1 : type === 'story' ? 2 : type === 'task' ? 3 : 4;

  return {
    id: Date.now() + Math.random() * 1000,
    name: fields.summary,
    type: type,
    level: level,
    parentId: parentId,
    children: [],
    status: statusMap[fields.status?.name] || 'pending',
    priority: fields.priority?.name?.toLowerCase() || 'medium',
    assignee: fields.assignee?.displayName || '',
    startDate: fields.created?.split('T')[0] || new Date().toISOString().split('T')[0],
    endDate: fields.duedate || new Date(Date.now() + 30*24*60*60*1000).toISOString().split('T')[0],
    estimatedHours: fields.timetracking?.originalEstimateSeconds ? 
      Math.round(fields.timetracking.originalEstimateSeconds / 3600) : 0,
    actualHours: fields.timetracking?.timeSpentSeconds ? 
      Math.round(fields.timetracking.timeSpentSeconds / 3600) : 0,
    comments: [],
    jira: {
      issueKey: issue.key,
      issueId: issue.id,
      issueUrl: `${issue.self.split('/rest/')[0]}/browse/${issue.key}`,
      issueType: fields.issuetype?.name,
      lastSynced: new Date().toISOString()
    }
  };
}

function formatJiraIssueForSync(issue) {
  const fields = issue.fields;
  const statusMap = {
    'To Do': 'pending',
    'Open': 'pending',
    'In Progress': 'in-progress',
    'Done': 'review',
    'Resolved': 'review',
    'Closed': 'review'
  };

  return {
    name: fields.summary,
    status: statusMap[fields.status.name] || 'pending',
    priority: fields.priority?.name?.toLowerCase() || 'medium',
    assignee: fields.assignee?.displayName || '',
    endDate: fields.duedate || new Date(Date.now() + 30*24*60*60*1000).toISOString().split('T')[0],
    estimatedHours: fields.timetracking?.originalEstimateSeconds ? 
      Math.round(fields.timetracking.originalEstimateSeconds / 3600) : 0,
    actualHours: fields.timetracking?.timeSpentSeconds ? 
      Math.round(fields.timetracking.timeSpentSeconds / 3600) : 0,
    comments: fields.comment?.comments?.map(c => ({
      id: c.id,
      text: c.body,
      author: c.author.displayName,
      timestamp: c.created,
      fromJira: true
    })) || [],
    lastSynced: new Date().toISOString()
  };
}

function logSync(action, issueKey, status, error = null) {
  try {
    const log = {
      timestamp: new Date().toISOString(),
      action,
      issueKey,
      status,
      error
    };
    
    addSyncLog(log);
  } catch (e) {
    console.error('Failed to write sync log:', e);
  }
}

// Start server
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Node version: ${process.version}`);
});