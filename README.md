# AI Smart Prompt Stream

An intelligent AI model routing application that automatically selects the best AI model for your prompts based on task type and priority preferences (accuracy, speed, or cost).

## Features

- 🤖 **Intelligent Model Routing**: Automatically routes prompts to the optimal AI model
- 🎯 **Priority-Based Selection**: Choose between accuracy, speed, or cost optimization
- 📊 **Real-time Analytics**: Monitor model performance and usage statistics
- 💡 **Prompt Enhancement**: AI-powered suggestions to improve your prompts
- 🚀 **Modern UI**: Beautiful, responsive interface built with React and shadcn/ui
- 🔗 **Database Integration**: Supabase database for model scoring and metadata

## Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for fast development
- shadcn/ui components
- Tailwind CSS for styling
- React Query for API state management

### Backend
- FastAPI (Python)
- Groq API for LLM inference
- PostgreSQL (Supabase) for model data
- uvicorn ASGI server

## Prerequisites

- Node.js 18+ 
- Python 3.8+
- Groq API key
- Supabase database (or PostgreSQL)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-smart-prompt-stream
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy `env.example` to `.env` and fill in your credentials:

```bash
cp env.example .env
```

Edit `.env` with your actual values:

```env
# Groq API Configuration
GROQ_API_KEY=your_actual_groq_api_key

# Supabase Database Configuration  
DB_HOST=your_supabase_host
DB_PORT=6543
DB_NAME=postgres
DB_USER=your_supabase_user
DB_PASSWORD=your_supabase_password
```

### 5. Database Setup

Make sure your Supabase database has the required tables:

```sql
-- Models table
CREATE TABLE models (
    model_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL
);

-- Model scores table
CREATE TABLE model_scores (
    model_id VARCHAR REFERENCES models(model_id),
    category VARCHAR NOT NULL,
    accuracy DECIMAL,
    speed DECIMAL,
    cost DECIMAL,
    PRIMARY KEY (model_id, category)
);
```

## Running the Application

### Option 1: Run Everything Together (Recommended)

```bash
npm run dev:full
```

This command will start both the backend (port 8000) and frontend (port 5173) simultaneously.

### Option 2: Run Separately

**Terminal 1 - Backend:**
```bash
npm run backend
# or
python start.py
# or manually:
cd src/backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

## API Endpoints

### Backend (http://localhost:8000)

- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /models` - Get available models from database
- `POST /chat` - Main chat endpoint for AI responses
- `POST /enhance-prompt` - Get prompt enhancement suggestions
- `GET /analytics/models` - Get model analytics

### Frontend (http://localhost:5173)

- `/` - Landing page
- `/chat` - Main chat interface

## Usage

1. Open http://localhost:5173 in your browser
2. Navigate to the chat interface
3. Select your priority (accuracy, speed, or cost)
4. Type your prompt and send
5. The system will:
   - Analyze your prompt type
   - Select the optimal AI model
   - Route your request to Groq API
   - Return the response with metadata

## Project Structure

```
ai-smart-prompt-stream/
├── src/
│   ├── backend/           # Python FastAPI backend
│   │   ├── main.py       # FastAPI server
│   │   ├── infer.py      # LLM inference logic
│   │   ├── scorer.py     # Model scoring logic
│   │   ├── analyzer.py   # Prompt classification
│   │   └── database.py   # Database operations
│   ├── components/       # React components
│   ├── pages/           # React pages
│   ├── lib/             # Utilities and API client
│   └── hooks/           # React hooks
├── requirements.txt     # Python dependencies
├── package.json        # Node.js dependencies
├── env.example         # Environment variables template
└── start.py           # Python backend starter script
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Backend Issues

- **Import errors**: Make sure you're running the backend from the `src/backend` directory
- **Database connection**: Verify your Supabase credentials in `.env`
- **Groq API errors**: Check your `GROQ_API_KEY` is valid

### Frontend Issues

- **API connection errors**: Ensure backend is running on port 8000
- **CORS issues**: Backend is configured for localhost:5173

### Common Solutions

1. **Port conflicts**: Change ports in package.json scripts if needed
2. **Environment variables**: Double-check `.env` file exists and has correct values
3. **Dependencies**: Run `npm install` and `pip install -r requirements.txt`

## License

[Your chosen license]

## Support

For issues and questions, please open a GitHub issue or contact [your contact info].
