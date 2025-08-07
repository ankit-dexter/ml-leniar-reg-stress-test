# GDP growth predictor : ML (Leniar regression)

## 🌟 Features Overview

This complete system provides:
- **📊 CSV Upload & Dynamic Training**: Upload your economic data and train the model
- **📈 5-Year Future Predictions**: Generate GDP forecasts with visual charts
- **⚡ Natural Language Stress Testing**: Ask questions like "What if export growth increases by 2 points?"
- **🤖 AI-Powered Insights**: Get economic analysis using Perplexity AI
- **📋 Real-time Performance Tracking**: Monitor model accuracy and performance

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements_complete.txt
```

### 2. Configuration
1. Get your Perplexity API key from: https://www.perplexity.ai/settings/api
2. Replace "your-perplexity-api-key-here" in:
   - `complete_mcp_server.py` (line ~580)
   - `fastapi_complete.py` (line ~27)

### 3. Launch the Application
```bash
python fastapi_complete.py
```

### 4. Access the System
- **Main Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive Features**: All available through the web interface

## 📊 CSV Data Format

Your CSV file must have exactly these columns:

```csv
Year,Export_Growth_Score,FDI_Flows_Pct,CO2_Oil_Change_Pct,GDP_Growth_Score
2000,3.2,15.5,2.1,2.8
2001,3.5,18.2,1.8,3.1
...
```

### Data Requirements:
- **Year**: 2000-2025
- **Export_Growth_Score**: 1-5 (economic performance score)
- **FDI_Flows_Pct**: 0-50% (foreign direct investment as % of fixed investment)
- **CO2_Oil_Change_Pct**: -30% to +30% (year-over-year change in CO2 emissions from oil)
- **GDP_Growth_Score**: 1-5 (GDP growth performance score)
- **Minimum**: 15 data points for reliable training

## 🎯 How to Use

### Step 1: Upload Training Data
1. Click the upload area or drag & drop your CSV file
2. Wait for model training to complete
3. Check the model performance metrics

### Step 2: Generate 5-Year Predictions
1. Choose number of years to predict (1-10)
2. Click "Generate Predictions & Chart"
3. View forecasts and trend analysis

### Step 3: Run Stress Tests
Use natural language queries like:
- "What if export growth increases by 2 points?"
- "What if FDI decreases by 10%?"
- "What if CO2 emissions increase by 15%?"

### Step 4: Get AI Insights
Ask economic questions:
- "What factors most influence GDP growth?"
- "How reliable are these predictions?"
- "What policy recommendations emerge from this data?"

## 🔧 API Endpoints

- `POST /api/upload-csv` - Upload CSV and train model
- `GET /api/predict-future` - Generate future predictions
- `POST /api/stress-test` - Natural language stress testing
- `POST /api/ai-insights` - AI economic analysis
- `GET /api/model-status` - Model performance and status

## 📈 Understanding Results

### Model Performance Metrics
- **R² Score**: Higher is better (0-1), indicates how well the model explains variance
- **RMSE**: Lower is better, measures prediction error
- **Data Points**: Number of historical records used for training

### Predictions
- **GDP Growth Score**: 1-5 scale prediction for future years
- **Trend Analysis**: Increasing, decreasing, or stable growth pattern
- **Confidence**: Based on model performance and historical patterns

### Stress Testing
- **Base Scenario**: Current economic conditions
- **Stressed Scenario**: Modified conditions based on your query
- **Impact**: Quantified effect on GDP growth predictions

## 🛡️ Security & Guardrails

The system includes comprehensive guardrails:
- **Input Validation**: All data ranges are validated
- **Question Filtering**: Only economics-related questions are processed
- **Error Handling**: Graceful error messages and recovery
- **File Security**: CSV files are validated and processed safely

## 🎨 Advanced Features

### Custom Scenarios
You can specify custom scenarios for future predictions by modifying the prediction request.

### Batch Processing
The system can handle multiple stress test queries and compare results.

### Export Results
All predictions and analysis results can be exported for further analysis.

## 🔍 Troubleshooting

### Common Issues:
1. **API Key Error**: Ensure Perplexity API key is correctly configured
2. **CSV Format Error**: Check column names and data ranges
3. **Model Training Fails**: Ensure minimum 15 data points
4. **Predictions Disabled**: Upload and train model first

### Data Quality Tips:
- Ensure data consistency across years
- Check for outliers that might affect training
- Validate that scores are within expected ranges
- Include recent years for better trend analysis

## 📊 Sample Data

Use `sample_sea_gdp_data.csv` as a template for your own data. This file contains:
- 25 years of sample data (2000-2024)
- Realistic economic indicators
- Proper formatting and ranges

## 🌏 Economic Context

This system is specifically designed for Southeast Asian economies and considers:
- Export-driven growth patterns
- Foreign investment flows
- Environmental impact on economic growth
- Regional economic trends and relationships

## 📞 Support

For issues or questions:
1. Check the troubleshooting guide
2. Review console logs for detailed error messages
3. Ensure all requirements are installed correctly
4. Verify Perplexity API key has sufficient credits

## 🎯 Next Steps

After successful setup:
1. Upload your historical economic data
2. Explore different stress test scenarios
3. Generate predictions for planning and analysis
4. Use AI insights for policy recommendations

Happy forecasting! 🚀📊
