"""
Setup script for Live Email Categorizer
Automates the setup process for real-time email categorization
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header():
    print("=" * 60)
    print("📧 LIVE EMAIL CATEGORIZER SETUP")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_live.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ["logs", "models", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_config_files():
    """Create configuration files"""
    print("\n⚙️ Creating configuration files...")
    
    # Create .env template
    env_content = """# Email Categorizer Configuration
GMAIL_CREDENTIALS_FILE=credentials.json
GMAIL_TOKEN_FILE=token.json
LOG_LEVEL=INFO
PROCESSING_INTERVAL=300
CONFIDENCE_THRESHOLD=0.6
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Created .env file")
    
    # Create service config
    service_config = {
        "processing_interval": 300,  # 5 minutes
        "confidence_threshold": 0.6,
        "max_emails_per_run": 50,
        "categories": {
            "promotions": "📂 Promotions",
            "notifications": "🔔 Notifications",
            "important": "📌 Important",
            "jobs": "💼 Jobs",
            "spam": "🚫 Spam",
            "alerts": "⚠️ Alerts"
        }
    }
    
    with open("service_config.json", "w") as f:
        json.dump(service_config, f, indent=2)
    print("✅ Created service_config.json")

def check_gmail_setup():
    """Check Gmail API setup"""
    print("\n🔑 Checking Gmail API setup...")
    
    if os.path.exists("credentials.json"):
        print("✅ Gmail credentials found")
        return True
    else:
        print("⚠️ Gmail credentials not found")
        print("\nTo set up Gmail API:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable Gmail API")
        print("4. Create OAuth 2.0 credentials")
        print("5. Download as 'credentials.json'")
        print("6. Place it in the project directory")
        return False

def create_startup_scripts():
    """Create startup scripts"""
    print("\n🚀 Creating startup scripts...")
    
    # Windows batch file
    windows_script = """@echo off
echo Starting Email Categorizer Service...
python background_service.py
pause
"""
    with open("start_service.bat", "w") as f:
        f.write(windows_script)
    print("✅ Created start_service.bat")
    
    # Linux/Mac shell script
    unix_script = """#!/bin/bash
echo "Starting Email Categorizer Service..."
python3 background_service.py
"""
    with open("start_service.sh", "w") as f:
        f.write(unix_script)
    
    # Make executable
    os.chmod("start_service.sh", 0o755)
    print("✅ Created start_service.sh")
    
    # Web app launcher
    web_script = """@echo off
echo Starting Email Categorizer Web App...
streamlit run live_app.py
pause
"""
    with open("start_web_app.bat", "w") as f:
        f.write(web_script)
    print("✅ Created start_web_app.bat")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import streamlit
        import pandas
        import sklearn
        import google.auth
        print("✅ All required packages imported successfully")
        
        # Test model training
        from email_service import EmailCategorizer
        categorizer = EmailCategorizer()
        print("✅ Email categorizer initialized")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set up Gmail API credentials (if not done already)")
    print("2. Run the web app: python live_app.py")
    print("3. Start the background service: python background_service.py")
    print("\nOr use the provided scripts:")
    print("• Windows: start_web_app.bat and start_service.bat")
    print("• Linux/Mac: ./start_service.sh")
    print("\nFor more help, check the README.md file")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Create directories and files
    create_directories()
    create_config_files()
    create_startup_scripts()
    
    # Check Gmail setup
    gmail_ready = check_gmail_setup()
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed at testing")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()
    
    if not gmail_ready:
        print("\n⚠️ Remember to set up Gmail API credentials before running the service!")

if __name__ == "__main__":
    main()

