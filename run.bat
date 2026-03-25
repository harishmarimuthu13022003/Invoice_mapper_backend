@echo off
REM ========================================
REM AI Invoice Mapper - Windows Setup Script
REM ========================================

echo ============================================
echo 🚀 AI Invoice Mapper - Backend Setup
echo ============================================

REM Check if .env exists
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ✅ Created .env
    echo.
    echo ⚠️  IMPORTANT: Please edit .env and configure:
    echo    - MongoDB connection
    echo    - Ollama URL (or use OpenAI)
    echo.
)

REM Check Python
python --version >nul 2>&1 || (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Create directories
echo 📁 Creating directories...
if not exist uploads mkdir uploads
if not exist chroma_db mkdir chroma_db
if not exist data mkdir data

echo.
echo ============================================
echo ✅ Setup complete! Starting API...
echo ============================================
echo.
echo 📌 Next steps:
echo    1. Ensure MongoDB is running
echo    2. Ensure Ollama is running (or use OpenAI)
echo    3. API will be at: http://localhost:8003
echo    4. API docs at: http://localhost:8003/docs
echo.

REM Run the API
python main.py

pause