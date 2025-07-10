# Apanto - AI Model Router

Apanto is an intelligent AI model routing application that automatically selects the best AI model for your prompts based on task type and priority preferences. Host your own Hugging Face models and let Apanto route intelligently between them and premium AI models.

## Features

- **Multi-Provider AI Support**: Access models from OpenAI, DeepSeek, Google Gemini, Anthropic, and Groq
- **Intelligent Model Routing**: Automatically routes prompts to the optimal AI model across all providers
- **Host Custom Models**: Add your own Hugging Face models to the platform
- **Priority-Based Selection**: Choose between accuracy, speed, or cost optimization
- **Real-time Analytics**: Monitor model performance and usage statistics across providers
- **Prompt Enhancement**: AI-powered suggestions to improve your prompts using multiple models
- **Modern UI**: Beautiful, responsive interface built with React and shadcn/ui
- **Database Integration**: Supabase database for model scoring and metadata
- **Cost Optimization**: Automatic cost estimation and optimization across different providers

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
- Hugging Face Transformers for custom model hosting
- PostgreSQL (Supabase) for model data
- uvicorn ASGI server

## Prerequisites

- Node.js 18+ 
- Python 3.8+
- API keys for your preferred providers:
  - OpenAI API key (optional)
  - DeepSeek API key (optional)
  - Google Gemini API key (optional)
  - Anthropic API key (optional)
  - Groq API key (optional)
- Supabase database (or PostgreSQL)
- GPU recommended for hosting larger Hugging Face models

## Quick Setup (2 Steps)

### 1. Clone and Install

```bash
git clone <repository-url>
cd Apanto
npm install
```

### 2. Configure Environment

Create a `.env` file with your credentials:

```env
# Multi-Provider AI API Keys (all optional - only add the ones you want to use)
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key

# Supabase Database Configuration  
DB_HOST=aws-0-us-east-2.pooler.supabase.com
DB_PORT=6543
DB_NAME=postgres
DB_USER=postgres.jqvayaoaqjkytejrypxs
DB_PASSWORD=your_supabase_password
```

### 3. Run Application

```bash
npm run dev:full
```

That's it! The application will automatically:
- Install missing Node.js dependencies if needed
- Install all required Python dependencies 
- Clean up any conflicting processes
- Start backend and wait for it to be ready
- Start frontend once backend is connected

## Python Dependencies

All Python dependencies are managed through `requirements.txt`. 

The following packages are automatically installed when you run `npm run dev:full`:
- FastAPI & Uvicorn (API server)
- Pydantic (data validation)
- Requests & python-dotenv (HTTP & environment)
- psycopg2-binary (database connection)
- transformers, torch, accelerate (ML models)
- huggingface_hub (AI model hosting)
- openai (OpenAI API client)
- anthropic (Anthropic API client)
- google-generativeai (Google Gemini API client)

## Hugging Face Model Hosting

Apanto allows you to host your own Hugging Face models alongside premium AI models:

### Adding Models

1. Navigate to the chat interface
2. Click the "Host Model" button
3. Enter your Hugging Face model URL (e.g., `microsoft/DialoGPT-medium`)
4. Provide a custom name (optional)
5. Click "Host Model"

### Supported Models

- Text Generation (GPT, LLaMA, etc.)
- Text Classification  
- Question Answering
- Summarization
- Translation
- Sentiment Analysis
- Custom Fine-tuned Models

### API Endpoints for Custom Models

- `POST /host-model` - Add a Hugging Face model
- `GET /models` - List all available models (including hosted and provider models)
- `GET /providers` - Get all available AI providers and their models
- `GET /providers/{provider}/models` - Get models for a specific provider
- `POST /chat` - Chat with any model (auto-routed or specific)
- `DELETE /huggingface/models/{model_id}` - Remove hosted model

## API Endpoints

### Backend (http://localhost:8000)

- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /models` - Get all available models (including provider models)
- `GET /providers` - Get all AI providers and their models
- `GET /providers/{provider}/models` - Get models for specific provider
- `POST /chat` - Main chat endpoint for AI responses (auto-routed)
- `POST /enhance-prompt` - Get prompt enhancement suggestions
- `POST /improve-prompt` - Improve prompts using multiple AI models
- `GET /analytics/models` - Get model analytics across providers

### Frontend (http://localhost:5173)

- `/` - Landing page with feature overview
- `/chat` - Main chat interface with model hosting

## Supported AI Providers

Apanto supports multiple AI providers, automatically routing to the best model for your needs:

### OpenAI
- **GPT-4o**: Latest and most capable model
- **GPT-4o Mini**: Fast and cost-effective
- **GPT-4 Turbo**: High performance for complex tasks
- **GPT-3.5 Turbo**: Fast and economical

### Anthropic (Claude)
- **Claude 3.5 Sonnet**: Advanced reasoning and analysis
- **Claude 3.5 Haiku**: Fast and efficient
- **Claude 3 Opus**: Highest performance for complex tasks
- **Claude 3 Sonnet**: Balanced performance and cost

### Google Gemini
- **Gemini Pro**: Versatile and powerful
- **Gemini Pro Vision**: Multimodal capabilities

### DeepSeek
- **DeepSeek Chat**: General conversation and tasks
- **DeepSeek Coder**: Specialized for coding tasks

### Groq (Llama Models)
- **Llama 3 70B**: High-performance open model
- **Llama 3 8B**: Fast and efficient
- **Mixtral 8x7B**: Excellent reasoning capabilities
- **Gemma 2 9B**: Google's efficient model

## Usage

1. Open http://localhost:5173 in your browser
2. Navigate to the chat interface
3. Optionally add your own Hugging Face models
4. Type your prompt and send
5. Apanto will:
   - Analyze your prompt type
   - Select the optimal AI model across all providers
   - Route your request appropriately
   - Return the response with metadata and cost information

## Project Structure

```
ai-smart-prompt-stream/
├── src/
│   ├── backend/           # Python FastAPI backend
│   │   ├── main.py       # FastAPI server with HF model support
│   │   ├── infer.py      # LLM inference logic
│   │   ├── scorer.py     # Model scoring logic
│   │   ├── analyzer.py   # Prompt classification
│   │   └── database.py   # Database operations
│   ├── components/       # React components
│   ├── pages/           # React pages
│   │   ├── Index.tsx    # Landing page
│   │   └── Chat.tsx     # Chat interface with model hosting
│   ├── lib/             # Utilities and API client
│   └── hooks/           # React hooks
├── requirements.txt     # Python dependencies (including transformers)
├── package.json        # Node.js dependencies
├── env.example         # Environment variables template
└── start.py           # Python backend starter script
```

## Performance Considerations

- **Memory Usage**: Custom models are loaded into memory when first used
- **GPU Support**: Automatically detects and uses CUDA when available
- **Model Caching**: Models stay loaded until manually removed
- **Response Times**: Initial model load may take time for larger models
