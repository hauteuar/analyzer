const express = require('express');
const cors = require('cors');
const low = require('lowdb');
const FileSync = require('lowdb/adapters/FileSync');
const axios = require('axios');
const Holidays = require('date-holidays');
const path = require('path');

const app = express();
const adapter = new FileSync('db.json');
const db = low(adapter);

// Initialize holidays
const hd = new Holidays('US');

// Initialize database with default structure
db.defaults({ 
  projects: [],
  syncLog: []
}).write();

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
    const projects = db.get('projects').value();
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
    
    const existingProject = db.get('projects')
      .find({ id: projectId })
      .value();
    
    if (existingProject) {
      db.get('projects')
        .find({ id: projectId })
        .assign(project)
        .write();
    } else {
      db.get('projects')
        .push(project)
        .write();
    }
    
    res.json(project);
  } catch (error) {
    console.error('Error saving project:', error);
    res.status(500).json({ error: 'Failed to save project' });
  }
});

app.delete('/api/projects/:id', (req, res) => {
  try {
    const projectId = parseInt(req.params.id);
    db.get('projects')
      .remove({ id: projectId })
      .write();
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
    
    // Add custom fields if provided
    if (item.storyPoints) {
      issueData.fields.customfield_10016 = parseFloat(item.storyPoints);
    }
    
    if (item.labels && Array.isArray(item.labels) && item.labels.length > 0) {
      issueData.fields.labels = item.labels;
    }
    
    // Epic handling
    if (item.epicName && jiraIssueType === 'Epic') {
      issueData.fields.customfield_10011 = item.epicName; // Epic Name field
    }
    
    // Epic Link - for Stories/Tasks that should be linked to an epic
    if (item.epicLink && jiraIssueType !== 'Epic') {
      issueData.fields.customfield_10014 = item.epicLink; // Epic Link field
    }
    
    // Parent Link - for Sub-tasks
    if (item.parentKey && jiraIssueType === 'Sub-task') {
      issueData.fields.parent = { key: item.parentKey };
    }

    const response = await axios.post(
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
  const { jiraConfig } = req.body;
  
  try {
    // Try Greenhopper API first (Jira Agile / BofA Jira 3)
    let response;
    let epics = [];
    
    try {
      // Greenhopper API endpoint
      response = await axios.get(
        `${jiraConfig.url}/rest/greenhopper/1.0/epics`,
        {
          params: {
            searchQuery: '',
            projectKey: jiraConfig.defaultProject,
            maxResults: 100,
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
                done: epic.isDone || false
              });
            });
          }
        });
        epics = allEpics;
      }
    } catch (greenhopperError) {
      console.log('Greenhopper API failed, trying standard Jira API v2...');
      
      // Fallback to standard Jira API v2
      const jql = `project = ${jiraConfig.defaultProject} AND issuetype = Epic ORDER BY created DESC`;
      
      response = await axios.get(
        `${jiraConfig.url}/rest/api/2/search`,
        {
          params: {
            jql: jql,
            maxResults: 50,
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
      
      const childrenJql = `"Epic Link" = ${epicKey} OR parent = ${epicKey}`;
      const childrenResponse = await axios.get(
        `${jiraConfig.url}/rest/api/2/search`,
        {
          params: {
            jql: childrenJql,
            maxResults: 100,
            fields: 'summary,status,priority,assignee,duedate,issuetype,created,updated,timetracking'
          },
          headers: {
            'Authorization': `Bearer ${jiraConfig.apiToken}`,
            'Accept': 'application/json'
          }
        }
      );
      
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
    
    db.get('syncLog')
      .push(log)
      .write();
    
    const logs = db.get('syncLog').value();
    if (logs.length > 100) {
      db.set('syncLog', logs.slice(-100)).write();
    }
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