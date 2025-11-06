"""
Background Service for Real-time Email Categorization
Runs continuously to monitor and categorize new emails
"""

import time
import schedule
import logging
from datetime import datetime, timedelta
from email_service import EmailCategorizer
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_categorizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailCategorizerService:
    def __init__(self):
        self.categorizer = EmailCategorizer()
        self.last_check = None
        self.stats = {
            'total_processed': 0,
            'last_run': None,
            'categories': {},
            'errors': 0
        }
        
    def initialize(self):
        """Initialize the service"""
        try:
            logger.info("Initializing Email Categorizer Service...")
            
            # Authenticate with Gmail
            self.categorizer.authenticate_gmail()
            
            # Load or train model
            self.categorizer.load_model()
            
            logger.info("Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            return False
    
    def process_new_emails(self):
        """Process new emails since last check"""
        try:
            logger.info("Processing new emails...")
            
            # Get emails from last 10 minutes
            if self.last_check:
                # Get emails since last check
                query = f"after:{int(self.last_check.timestamp())}"
                emails = self._get_emails_with_query(query)
            else:
                # First run - get last 50 emails
                emails = self.categorizer.get_recent_emails(50)
            
            processed_count = 0
            category_stats = {}
            
            for email_data in emails:
                try:
                    # Skip if already categorized (has any of our labels)
                    if self._is_already_categorized(email_data):
                        continue
                    
                    # Predict category
                    category, confidence = self.categorizer.predict_category(email_data['text'])
                    
                    # Only process if confidence is high enough
                    if confidence > 0.6:
                        # Create/get label
                        label_id = self.categorizer.create_label(category)
                        
                        if label_id:
                            # Apply label
                            self.categorizer.apply_label(email_data['id'], label_id)
                            processed_count += 1
                            
                            # Update stats
                            category_stats[category] = category_stats.get(category, 0) + 1
                            
                            logger.info(f"Categorized: {email_data['subject'][:50]}... -> {category} ({confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error processing email {email_data['id']}: {e}")
                    self.stats['errors'] += 1
            
            # Update stats
            self.stats['total_processed'] += processed_count
            self.stats['last_run'] = datetime.now().isoformat()
            for category, count in category_stats.items():
                self.stats['categories'][category] = self.stats['categories'].get(category, 0) + count
            
            self.last_check = datetime.now()
            
            logger.info(f"Processed {processed_count} new emails")
            self._save_stats()
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error in process_new_emails: {e}")
            self.stats['errors'] += 1
            return 0
    
    def _get_emails_with_query(self, query):
        """Get emails using Gmail query"""
        try:
            results = self.categorizer.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=50
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                msg = self.categorizer.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()
                
                email_data = self.categorizer._parse_email(msg)
                if email_data:
                    emails.append(email_data)
            
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching emails with query: {e}")
            return []
    
    def _is_already_categorized(self, email_data):
        """Check if email is already categorized"""
        try:
            # Get message details to check labels
            msg = self.categorizer.service.users().messages().get(
                userId='me',
                id=email_data['id'],
                format='metadata',
                metadataHeaders=['Labels']
            ).execute()
            
            # Check if any of our category labels are applied
            category_labels = list(self.categorizer.categories.values())
            message_labels = msg.get('labelIds', [])
            
            # Get all labels to check names
            labels = self.categorizer.service.users().labels().list(userId='me').execute()
            label_map = {label['id']: label['name'] for label in labels.get('labels', [])}
            
            for label_id in message_labels:
                if label_map.get(label_id) in category_labels:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if email is categorized: {e}")
            return False
    
    def _save_stats(self):
        """Save statistics to file"""
        try:
            with open('service_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
    
    def load_stats(self):
        """Load statistics from file"""
        try:
            if os.path.exists('service_stats.json'):
                with open('service_stats.json', 'r') as f:
                    self.stats = json.load(f)
        except Exception as e:
            logger.error(f"Error loading stats: {e}")
    
    def get_stats(self):
        """Get current statistics"""
        return self.stats
    
    def run_full_categorization(self):
        """Run full categorization on recent emails"""
        logger.info("Running full categorization...")
        results = self.categorizer.auto_categorize_emails(max_emails=200)
        
        # Update stats
        self.stats['total_processed'] += results['processed']
        self.stats['last_run'] = datetime.now().isoformat()
        for category, count in results['categories'].items():
            self.stats['categories'][category] = self.stats['categories'].get(category, 0) + count
        
        self._save_stats()
        return results

def run_service():
    """Main service runner"""
    service = EmailCategorizerService()
    
    # Initialize service
    if not service.initialize():
        logger.error("Failed to initialize service. Exiting.")
        return
    
    # Load previous stats
    service.load_stats()
    
    # Schedule tasks
    schedule.every(5).minutes.do(service.process_new_emails)
    schedule.every(1).hours.do(service.run_full_categorization)
    
    logger.info("Email Categorizer Service started")
    logger.info("Scheduled tasks:")
    logger.info("- Process new emails: every 5 minutes")
    logger.info("- Full categorization: every 1 hour")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")

if __name__ == "__main__":
    run_service()

