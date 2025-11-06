// Popup script for Email Categorizer extension

document.addEventListener('DOMContentLoaded', function() {
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');
  const statsDiv = document.getElementById('stats');
  const categoriesDiv = document.getElementById('categories');
  
  // Check service status
  checkServiceStatus();
  
  // Button event listeners
  document.getElementById('categorizeCurrent').addEventListener('click', categorizeCurrentEmail);
  document.getElementById('openDashboard').addEventListener('click', openDashboard);
  document.getElementById('startService').addEventListener('click', startService);
  
  // Load statistics
  loadStatistics();
});

async function checkServiceStatus() {
  try {
    const response = await fetch('http://localhost:8501/api/status');
    if (response.ok) {
      updateStatus(true, 'Service Online');
    } else {
      updateStatus(false, 'Service Offline');
    }
  } catch (error) {
    updateStatus(false, 'Service Offline');
  }
}

function updateStatus(isOnline, text) {
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');
  
  statusIndicator.className = `status-indicator ${isOnline ? 'status-online' : 'status-offline'}`;
  statusText.textContent = text;
}

async function categorizeCurrentEmail() {
  try {
    // Get current Gmail tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab.url.includes('mail.google.com')) {
      alert('Please open Gmail first');
      return;
    }
    
    // Inject script to get email content
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: getCurrentEmailContent
    });
    
    const emailContent = results[0].result;
    
    if (!emailContent) {
      alert('No email selected or content not found');
      return;
    }
    
    // Send to categorization service
    const response = await fetch('http://localhost:8501/api/categorize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: emailContent })
    });
    
    if (response.ok) {
      const result = await response.json();
      showCategorizationResult(result);
    } else {
      alert('Error categorizing email');
    }
    
  } catch (error) {
    console.error('Error:', error);
    alert('Error categorizing email. Make sure the service is running.');
  }
}

function getCurrentEmailContent() {
  // This function runs in the Gmail page context
  const emailElements = document.querySelectorAll('[data-message-id]');
  
  for (const element of emailElements) {
    if (element.querySelector('.a3s')) {
      const subject = element.querySelector('h2')?.textContent || '';
      const body = element.querySelector('.a3s')?.textContent || '';
      return `${subject} ${body}`.trim();
    }
  }
  
  return null;
}

function showCategorizationResult(result) {
  const categoryNames = {
    'promotions': '📂 Promotions',
    'notifications': '🔔 Notifications',
    'important': '📌 Important',
    'jobs': '💼 Jobs',
    'spam': '🚫 Spam',
    'alerts': '⚠️ Alerts'
  };
  
  const category = categoryNames[result.category] || result.category;
  const confidence = Math.round(result.confidence * 100);
  
  alert(`Email Categorized!\n\nCategory: ${category}\nConfidence: ${confidence}%`);
}

function openDashboard() {
  chrome.tabs.create({ url: 'http://localhost:8501' });
}

async function startService() {
  try {
    // This would typically start the background service
    // For now, just open the dashboard
    openDashboard();
    alert('Please start the background service manually:\n\npython background_service.py');
  } catch (error) {
    console.error('Error starting service:', error);
    alert('Error starting service');
  }
}

async function loadStatistics() {
  try {
    const response = await fetch('http://localhost:8501/api/stats');
    if (response.ok) {
      const stats = await response.json();
      displayStatistics(stats);
    }
  } catch (error) {
    console.error('Error loading statistics:', error);
  }
}

function displayStatistics(stats) {
  document.getElementById('totalProcessed').textContent = stats.total_processed || 0;
  document.getElementById('lastRun').textContent = stats.last_run ? 
    new Date(stats.last_run).toLocaleString() : 'Never';
  document.getElementById('categoryCount').textContent = Object.keys(stats.categories || {}).length;
  
  statsDiv.style.display = 'block';
  categoriesDiv.style.display = 'block';
}

