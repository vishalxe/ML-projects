"""
Email Service for Real-time Email Categorization
Integrates with Gmail API to automatically categorize emails
"""

import os
import pickle
import base64
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Gmail API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ML imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Email processing
import email
from email.mime.text import MIMEText
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailCategorizer:
    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.model = None
        self.vectorizer = None
        self.categories = {
            'promotions': '📂 Promotions',
            'notifications': '🔔 Notifications',
            'important': '📌 Important',
            'jobs': '💼 Jobs',
            'spam': '🚫 Spam',
            'alerts': '⚠️ Alerts'
        }
        self.alert_keywords = [
            "expire", "expiring", "ending", "last date", "final day",
            "today only", "due in", "offer ends", "deadline", "last chance"
        ]
        
    def authenticate_gmail(self):
        """Authenticate with Gmail API"""
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
                 'https://www.googleapis.com/auth/gmail.modify',
                 'https://www.googleapis.com/auth/gmail.labels']
        
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(f"Gmail credentials file not found: {self.credentials_file}")
                
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail API authenticated successfully")
        
    def load_model(self, model_path='email_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load trained ML model and vectorizer"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("ML model loaded successfully")
        except FileNotFoundError:
            logger.warning("Model files not found, will train new model")
            self.train_model()
    
    def train_model(self, dataset_path='email_dataset.csv'):
        """Train the ML model on the dataset"""
        logger.info("Training ML model...")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Prepare data
        X = df['text']
        y = df['category']
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_vec = self.vectorizer.fit_transform(X)
        
        # Train model
        self.model = MultinomialNB()
        self.model.fit(X_vec, y)
        
        # Save model
        with open('email_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info("Model trained and saved successfully")
    
    def predict_category(self, email_text: str) -> tuple:
        """Predict category for email text"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Check for alerts first
        if self._detect_alert(email_text):
            return 'alerts', 0.95
        
        # Predict using ML model
        email_vec = self.vectorizer.transform([email_text])
        prediction = self.model.predict(email_vec)[0]
        probabilities = self.model.predict_proba(email_vec)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def _detect_alert(self, text: str) -> bool:
        """Detect if email contains alert keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.alert_keywords)
    
    def get_recent_emails(self, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get recent emails from Gmail"""
        try:
            # Get recent messages
            results = self.service.users().messages().list(
                userId='me', 
                maxResults=max_results,
                q='in:inbox'
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me', 
                    id=message['id'],
                    format='full'
                ).execute()
                
                email_data = self._parse_email(msg)
                if email_data:
                    emails.append(email_data)
            
            return emails
            
        except HttpError as error:
            logger.error(f"Error fetching emails: {error}")
            return []
    
    def _parse_email(self, msg) -> Dict[str, Any]:
        """Parse Gmail message into structured data"""
        try:
            headers = msg['payload'].get('headers', [])
            
            # Extract basic info
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            # Extract body
            body = self._extract_body(msg['payload'])
            
            # Clean and prepare text for analysis
            email_text = f"{subject} {body}".strip()
            
            return {
                'id': msg['id'],
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body,
                'text': email_text,
                'thread_id': msg['threadId']
            }
            
        except Exception as e:
            logger.error(f"Error parsing email: {e}")
            return None
    
    def _extract_body(self, payload) -> str:
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data', '')
                    if data:
                        body += base64.urlsafe_b64decode(data).decode('utf-8')
                elif part['mimeType'] == 'text/html':
                    data = part['body'].get('data', '')
                    if data:
                        html_body = base64.urlsafe_b64decode(data).decode('utf-8')
                        # Simple HTML to text conversion
                        body += re.sub(r'<[^>]+>', '', html_body)
        else:
            if payload['mimeType'] == 'text/plain':
                data = payload['body'].get('data', '')
                if data:
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        return body
    
    def create_label(self, category: str) -> str:
        """Create Gmail label for category"""
        label_name = self.categories[category]
        
        try:
            # Check if label exists
            labels = self.service.users().labels().list(userId='me').execute()
            existing_labels = [label['name'] for label in labels.get('labels', [])]
            
            if label_name not in existing_labels:
                # Create new label
                label_object = {
                    'name': label_name,
                    'labelListVisibility': 'labelShow',
                    'messageListVisibility': 'show'
                }
                
                created_label = self.service.users().labels().create(
                    userId='me', 
                    body=label_object
                ).execute()
                
                logger.info(f"Created label: {label_name}")
                return created_label['id']
            else:
                # Find existing label ID
                for label in labels.get('labels', []):
                    if label['name'] == label_name:
                        return label['id']
                        
        except HttpError as error:
            logger.error(f"Error creating label: {error}")
            return None
    
    def apply_label(self, message_id: str, label_id: str):
        """Apply label to email message"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            logger.info(f"Applied label to message {message_id}")
        except HttpError as error:
            logger.error(f"Error applying label: {error}")
    
    def auto_categorize_emails(self, max_emails: int = 50):
        """Automatically categorize recent emails"""
        logger.info("Starting auto-categorization...")
        
        # Get recent emails
        emails = self.get_recent_emails(max_emails)
        
        categorized_count = 0
        category_stats = {}
        
        for email_data in emails:
            try:
                # Predict category
                category, confidence = self.predict_category(email_data['text'])
                
                # Create/get label
                label_id = self.create_label(category)
                
                if label_id:
                    # Apply label
                    self.apply_label(email_data['id'], label_id)
                    categorized_count += 1
                    
                    # Update stats
                    category_stats[category] = category_stats.get(category, 0) + 1
                    
                    logger.info(f"Categorized: {email_data['subject'][:50]}... -> {category} ({confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Error categorizing email {email_data['id']}: {e}")
        
        logger.info(f"Auto-categorization complete. Processed {categorized_count} emails.")
        logger.info(f"Category distribution: {category_stats}")
        
        return {
            'processed': categorized_count,
            'categories': category_stats
        }
    
    def get_categorized_emails(self, category: str = None) -> List[Dict[str, Any]]:
        """Get emails by category"""
        try:
            # Get all labels
            labels = self.service.users().labels().list(userId='me').execute()
            label_map = {label['name']: label['id'] for label in labels.get('labels', [])}
            
            if category:
                label_name = self.categories.get(category)
                if not label_name or label_name not in label_map:
                    return []
                
                # Get messages with specific label
                results = self.service.users().messages().list(
                    userId='me',
                    labelIds=[label_map[label_name]],
                    maxResults=50
                ).execute()
            else:
                # Get all categorized messages
                category_labels = [label_map[name] for name in self.categories.values() if name in label_map]
                if not category_labels:
                    return []
                
                results = self.service.users().messages().list(
                    userId='me',
                    labelIds=category_labels,
                    maxResults=100
                ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()
                
                email_data = self._parse_email(msg)
                if email_data:
                    # Add labels to email data
                    email_data['labels'] = [label['name'] for label in msg.get('labelIds', [])]
                    emails.append(email_data)
            
            return emails
            
        except HttpError as error:
            logger.error(f"Error fetching categorized emails: {error}")
            return []

def main():
    """Main function to run email categorization"""
    categorizer = EmailCategorizer()
    
    try:
        # Authenticate
        categorizer.authenticate_gmail()
        
        # Load or train model
        categorizer.load_model()
        
        # Auto-categorize emails
        results = categorizer.auto_categorize_emails(max_emails=100)
        
        print(f"✅ Successfully categorized {results['processed']} emails")
        print(f"📊 Category distribution: {results['categories']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

