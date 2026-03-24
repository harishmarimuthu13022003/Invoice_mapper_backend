#!/bin/bash
# ========================================
# AI Invoice Mapper - Setup & Run Script
# ========================================

echo "============================================"
echo "🚀 AI Invoice Mapper - Backend Setup"
echo "============================================"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and configure:"
    echo "   - MongoDB connection"
    echo "   - Ollama URL (or use OpenAI)"
    echo ""
fi

# Check Python version
echo "🐍 Checking Python version..."
python --version || python3 --version

# Create virtual environment if needed
if [ !d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "📥 Activating virtual environment..."
source venv/bin/activate  # Linux/Mac
# source venv\Scripts\activate  # Windows

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check MongoDB
echo ""
echo "🔍 Checking MongoDB..."
if command -v mongod &> /dev/null; then
    echo "   MongoDB found - starting..."
    mongod --dbpath ./data/db &
else
    echo "   ⚠️  MongoDB not found. Please install MongoDB Community Server"
    echo "   Download: https://www.mongodb.com/try/download/community"
fi

# Check Ollama
echo ""
echo "🔍 Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "   Ollama found - downloading Llama3 model..."
    ollama pull llama3
    echo "   Starting Ollama service..."
    ollama serve &
else
    echo "   ⚠️  Ollama not found."
    echo "   Download from: https://ollama.com"
    echo "   Or set OPENAI_API_KEY in .env to use OpenAI instead"
fi

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p chroma_db
mkdir -p data/db

# Start the API
echo ""
echo "============================================"
echo "✅ Setup complete! Starting API..."
echo "============================================"
echo ""
echo "📌 Next steps:"
echo "   1. Ensure MongoDB is running"
echo "   2. Ensure Ollama is running (or use OpenAI)"
echo "   3. API will be at: http://localhost:8001"
echo "   4. API docs at: http://localhost:8001/docs"
echo ""

# Run the API
python main.py