// Background service worker for Email Categorizer extension

chrome.runtime.onInstalled.addListener(() => {
  console.log('Email Categorizer extension installed');
});

// Handle messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'categorizeEmail') {
    categorizeEmail(request.text)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true; // Keep message channel open for async response
  }
});

async function categorizeEmail(emailText) {
  try {
    const response = await fetch('http://localhost:8501/api/categorize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: emailText })
    });
    
    if (response.ok) {
      return await response.json();
    } else {
      throw new Error('Categorization service not available');
    }
  } catch (error) {
    console.error('Error categorizing email:', error);
    throw error;
  }
}

// Check service status periodically
setInterval(async () => {
  try {
    const response = await fetch('http://localhost:8501/api/status');
    const isOnline = response.ok;
    
    // Update badge
    chrome.action.setBadgeText({
      text: isOnline ? 'ON' : 'OFF'
    });
    
    chrome.action.setBadgeBackgroundColor({
      color: isOnline ? '#28a745' : '#dc3545'
    });
    
  } catch (error) {
    chrome.action.setBadgeText({ text: 'OFF' });
    chrome.action.setBadgeBackgroundColor({ color: '#dc3545' });
  }
}, 30000); // Check every 30 seconds

