# 📧 Live Email Categorizer

A real-time email categorization system that automatically categorizes your Gmail emails using machine learning. The system runs continuously in the background, monitoring new emails and applying appropriate labels based on content analysis.

## ✨ Features

- **Real-time Processing**: Automatically categorizes new emails as they arrive
- **Gmail Integration**: Direct integration with Gmail API
- **Machine Learning**: Uses trained ML model for accurate categorization
- **Web Dashboard**: Beautiful web interface for monitoring and control
- **Background Service**: Runs continuously without user intervention
- **Multiple Categories**: Promotions, Notifications, Important, Jobs, Spam, Alerts
- **Confidence Scoring**: Only processes emails with high confidence scores
- **Statistics Tracking**: Detailed analytics and reporting

## 🚀 Quick Start

### 1. Setup

Run the automated setup script:

```bash
python setup_live.py
```

This will:

- Install all required dependencies
- Create necessary directories
- Set up configuration files
- Create startup scripts

### 2. Gmail API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API
4. Create OAuth 2.0 credentials
5. Download credentials as `credentials.json`
6. Place it in the project directory

### 3. Start the System

**Option A: Web Interface Only**

```bash
python live_app.py
```

**Option B: Background Service + Web Interface**

```bash
# Terminal 1: Start background service
python background_service.py

# Terminal 2: Start web interface
python live_app.py
```

**Option C: Use provided scripts**

- Windows: `start_web_app.bat` and `start_service.bat`
- Linux/Mac: `./start_service.sh`

## 📊 Web Dashboard

Access the web interface at `http://localhost:8501` to:

- **Dashboard**: View real-time statistics and charts
- **Live Categorization**: Manually categorize emails
- **Categorized Emails**: Browse emails by category
- **Settings**: Configure the system

## 🔧 Configuration

### Service Settings

Edit `service_config.json`:

```json
{
  "processing_interval": 300,
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
```

### Environment Variables

Create `.env` file:

```env
GMAIL_CREDENTIALS_FILE=credentials.json
GMAIL_TOKEN_FILE=token.json
LOG_LEVEL=INFO
PROCESSING_INTERVAL=300
CONFIDENCE_THRESHOLD=0.6
```

## 📁 Project Structure

```
email_auto_categorizer/
├── live_app.py              # Web interface
├── email_service.py         # Gmail API integration
├── background_service.py    # Background processing service
├── setup_live.py           # Automated setup script
├── requirements_live.txt   # Dependencies for live version
├── gmail_addon/            # Gmail Add-on files
│   ├── manifest.json
│   └── Code.gs
├── service_config.json     # Service configuration
├── service_stats.json      # Runtime statistics
└── logs/                   # Log files
```

## 🤖 Machine Learning Model

The system uses a Naive Bayes classifier with TF-IDF vectorization:

- **Training Data**: 500 email samples across 6 categories
- **Features**: TF-IDF with 5000 most frequent terms
- **Algorithm**: Multinomial Naive Bayes
- **Confidence Threshold**: 60% (configurable)

### Model Training

The model is automatically trained on first run using `email_dataset.csv`. You can retrain it by deleting the model files:

```bash
rm email_model.pkl vectorizer.pkl
python email_service.py
```

## 📈 Monitoring

### Service Statistics

The service tracks:

- Total emails processed
- Category distribution
- Processing errors
- Last run timestamp

View stats in the web dashboard or check `service_stats.json`.

### Logs

Service logs are stored in `email_categorizer.log`:

- Processing activities
- Error messages
- Performance metrics

## 🔒 Security & Privacy

- **OAuth 2.0**: Secure Gmail API authentication
- **Local Processing**: All email processing happens locally
- **No Data Storage**: Emails are not stored permanently
- **Token Management**: Automatic token refresh

## 🛠️ Troubleshooting

### Common Issues

1. **Gmail API Authentication Error**

   - Ensure `credentials.json` is in the project directory
   - Check that Gmail API is enabled in Google Cloud Console
   - Delete `token.json` and re-authenticate

2. **Service Not Processing Emails**

   - Check service logs in `email_categorizer.log`
   - Verify Gmail API permissions
   - Ensure confidence threshold is appropriate

3. **Model Not Found**
   - Delete model files to retrain: `rm email_model.pkl vectorizer.pkl`
   - Run the service again to retrain

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance

- **Processing Speed**: ~50 emails per minute
- **Memory Usage**: ~100MB base + 50MB per 1000 emails
- **Accuracy**: ~85% on test dataset
- **Latency**: Real-time processing (5-minute intervals)

## 🔄 Updates & Maintenance

### Updating the Model

1. Add new training data to `email_dataset.csv`
2. Delete existing model files
3. Restart the service

### Adding New Categories

1. Update `service_config.json`
2. Add training data for new category
3. Retrain the model

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review service logs
3. Check Gmail API documentation
4. Verify configuration settings

## 🎯 Future Enhancements

- [ ] Support for other email providers (Outlook, Yahoo)
- [ ] Advanced ML models (BERT, GPT)
- [ ] Email thread analysis
- [ ] Custom category training
- [ ] Mobile app interface
- [ ] Email summarization
- [ ] Sentiment analysis

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Email Categorizing! 📧✨**

