
#!/usr/bin/env python3
"""
Complete FastAPI Frontend for GDP Prediction System
Integrates with complete MCP server for CSV training, predictions, and stress testing
"""

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import json
import base64
import asyncio
from pathlib import Path
import logging

# Import our complete MCP server
from complete_mcp_server import CompleteMCPServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Southeast Asia GDP Growth Predictor - Complete Edition",
    description="AI-powered economic forecasting with CSV training and stress testing",
    version="2.0.0"
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # frontend origins allowed
    allow_credentials=True,
    allow_methods=["*"],              # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # allow all headers
)

# Initialize Complete MCP Server
PERPLEXITY_API_KEY = "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"  # Replace with actual key
mcp_server = CompleteMCPServer(PERPLEXITY_API_KEY)

# Pydantic models for request validation
class FuturePredictionRequest(BaseModel):
    years_ahead: int = Field(default=5, ge=1, le=10, description="Number of years to predict")
    scenarios: Optional[Dict] = Field(default=None, description="Custom scenarios for specific years")

class StressTestRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=200, description="Natural language stress test query")
    target_year: int = Field(..., ge=2023, le=2030, description="Target year for stress test")

class AIInsightRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=500, description="Economic question for AI analysis")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard with all features"""
    model_status = mcp_server.get_model_status()
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "model_status": model_status
    })

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file and train the model"""
    try:
        logger.info(f"Received file upload: {file.filename}")
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")

        # Read file content
        file_content = await file.read()

        # Process with MCP server
        result = await mcp_server.handle_csv_upload(file_content, file.filename)

        if result['success']:
            logger.info(f"Successfully processed CSV: {file.filename}")
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'CSV processing failed'))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/api/predict-future")
async def predict_future(request: FuturePredictionRequest):
    """Generate future GDP predictions with charts"""
    try:
        result = await mcp_server.handle_future_predictions(
            years_ahead=request.years_ahead,
            scenarios=request.scenarios
        )

        if result['success']:
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Prediction failed'))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Future prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/stress-test")
async def stress_test(request: StressTestRequest):
    try:
        logger.info(f"Received stress test query: {request}")
        result = await mcp_server.handle_stress_test_query(
            query=request.query,
            target_year=request.target_year
        )
        
        if result['success']:
            return {
                "success": True,
                "stress_test_result": result,
                "message": "Stress test completed successfully"
            }
        else:
            return {
                "success": False,
                "error": result['error']
            }
            
    except Exception as e:
        logger.error(f"Stress test endpoint error: {e}")
        return {
            "success": False,
            "error": f"Stress test failed: {str(e)}"
        }

@app.post("/api/ai-insights")
async def ai_insights(request: AIInsightRequest):
    """Get AI-powered economic insights"""
    try:
        result = await mcp_server.handle_ai_insights(
            question=request.question
        )

        return result

    except Exception as e:
        logger.error(f"AI insights error: {e}")
        raise HTTPException(status_code=500, detail=f"AI insights failed: {str(e)}")

@app.get("/api/model-status")
async def get_model_status():
    """Get current model status and performance"""
    try:
        status = mcp_server.get_model_status()
        return status

    except Exception as e:
        logger.error(f"Model status error: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "GDP Growth Predictor Complete Edition",
        "features": [
            "CSV Upload & Training",
            "5-Year Predictions with Charts",
            "Natural Language Stress Testing",
            "AI-Powered Economic Insights"
        ]
    }

# Error handlers
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input parameters",
            "details": "Please check your input values and try again",
            "requirements": {
                "csv_format": "Year,Export_Growth_Score,FDI_Flows_Pct,CO2_Oil_Change_Pct,GDP_Growth_Score",
                "years_ahead": "1 to 10 years",
                "query_length": "10 to 200 characters"
            }
        }
    )

@app.exception_handler(413)
async def file_too_large_handler(request: Request, exc):
    """Handle file size errors"""
    return JSONResponse(
        status_code=413,
        content={
            "error": "File too large",
            "details": "CSV file must be smaller than 10MB"
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Complete GDP Growth Predictor System")
    print("=" * 50)
    print("🎯 Features Available:")
    print("   📊 CSV Upload & Dynamic Model Training")
    print("   📈 5-Year Future Predictions with Charts")
    print("   🔍 Natural Language Stress Testing")
    print("   🤖 AI-Powered Economic Insights")
    print("   📋 Real-time Model Performance Tracking")
    print("")
    print("🌐 Access Dashboard: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("")
    print("⚡ Ready for CSV data upload and economic forecasting!")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
