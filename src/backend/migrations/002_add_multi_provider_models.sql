-- Add models from multiple AI providers
-- OpenAI Models
INSERT INTO models (name, model_id, is_huggingface, huggingface_url) VALUES
('GPT-4o', 'gpt-4o', FALSE, NULL),
('GPT-4o Mini', 'gpt-4o-mini', FALSE, NULL),
('GPT-4 Turbo', 'gpt-4-turbo', FALSE, NULL),
('GPT-4', 'gpt-4', FALSE, NULL),
('GPT-3.5 Turbo', 'gpt-3.5-turbo', FALSE, NULL)
ON CONFLICT (model_id) DO NOTHING;

-- DeepSeek Models
INSERT INTO models (name, model_id, is_huggingface, huggingface_url) VALUES
('DeepSeek Chat', 'deepseek-chat', FALSE, NULL),
('DeepSeek Coder', 'deepseek-coder', FALSE, NULL)
ON CONFLICT (model_id) DO NOTHING;

-- Google Gemini Models
INSERT INTO models (name, model_id, is_huggingface, huggingface_url) VALUES
('Gemini 2.5 Pro', 'gemini-2.5-pro', FALSE, NULL),
('Gemini 2.5 Flash', 'gemini-2.5-flash', FALSE, NULL)
ON CONFLICT (model_id) DO NOTHING;

-- Anthropic Claude Models
INSERT INTO models (name, model_id, is_huggingface, huggingface_url) VALUES
('Claude 3.5 Sonnet', 'claude-3-5-sonnet-20241022', FALSE, NULL),
('Claude 3.5 Haiku', 'claude-3-5-haiku-20241022', FALSE, NULL),
('Claude 3 Opus', 'claude-3-opus-20240229', FALSE, NULL),
('Claude 3 Sonnet', 'claude-3-sonnet-20240229', FALSE, NULL),
('Claude 3 Haiku', 'claude-3-haiku-20240307', FALSE, NULL)
ON CONFLICT (model_id) DO NOTHING;

-- Groq Models (Llama, Mixtral, etc.)
INSERT INTO models (name, model_id, is_huggingface, huggingface_url) VALUES
('Llama 3 70B', 'llama3-70b-8192', FALSE, NULL),
('Llama 3 8B', 'llama3-8b-8192', FALSE, NULL),
('Mixtral 8x7B', 'mixtral-8x7b-32768', FALSE, NULL),
('Gemma 2 9B', 'gemma2-9b-it', FALSE, NULL)
ON CONFLICT (model_id) DO NOTHING;

-- Add scores for OpenAI models
INSERT INTO model_scores (model_id, category, accuracy, speed, cost) VALUES
-- GPT-4o
('gpt-4o', 'qa', 0.95, 0.8, 0.3),
('gpt-4o', 'coding', 0.92, 0.7, 0.3),
('gpt-4o', 'reasoning', 0.94, 0.8, 0.3),
('gpt-4o', 'creative', 0.93, 0.8, 0.3),
-- GPT-4o Mini
('gpt-4o-mini', 'qa', 0.88, 0.9, 0.1),
('gpt-4o-mini', 'coding', 0.85, 0.8, 0.1),
('gpt-4o-mini', 'reasoning', 0.87, 0.9, 0.1),
('gpt-4o-mini', 'creative', 0.86, 0.9, 0.1),
-- GPT-4 Turbo
('gpt-4-turbo', 'qa', 0.93, 0.7, 0.5),
('gpt-4-turbo', 'coding', 0.90, 0.6, 0.5),
('gpt-4-turbo', 'reasoning', 0.92, 0.7, 0.5),
('gpt-4-turbo', 'creative', 0.91, 0.7, 0.5),
-- GPT-3.5 Turbo
('gpt-3.5-turbo', 'qa', 0.82, 0.95, 0.05),
('gpt-3.5-turbo', 'coding', 0.78, 0.9, 0.05),
('gpt-3.5-turbo', 'reasoning', 0.80, 0.95, 0.05),
('gpt-3.5-turbo', 'creative', 0.79, 0.95, 0.05)
ON CONFLICT (model_id, category) DO NOTHING;

-- Add scores for DeepSeek models
INSERT INTO model_scores (model_id, category, accuracy, speed, cost) VALUES
-- DeepSeek Chat
('deepseek-chat', 'qa', 0.89, 0.85, 0.08),
('deepseek-chat', 'coding', 0.91, 0.8, 0.08),
('deepseek-chat', 'reasoning', 0.88, 0.85, 0.08),
('deepseek-chat', 'creative', 0.87, 0.85, 0.08),
-- DeepSeek Coder
('deepseek-coder', 'qa', 0.85, 0.8, 0.08),
('deepseek-coder', 'coding', 0.93, 0.75, 0.08),
('deepseek-coder', 'reasoning', 0.84, 0.8, 0.08),
('deepseek-coder', 'creative', 0.83, 0.8, 0.08)
ON CONFLICT (model_id, category) DO NOTHING;

-- Add scores for Gemini models
INSERT INTO model_scores (model_id, category, accuracy, speed, cost) VALUES
-- Gemini 2.5 Pro
('gemini-2.5-pro', 'qa', 0.90, 0.9, 0.1),
('gemini-2.5-pro', 'coding', 0.88, 0.85, 0.1),
('gemini-2.5-pro', 'reasoning', 0.89, 0.9, 0.1),
('gemini-2.5-pro', 'creative', 0.88, 0.9, 0.1),
-- Gemini 2.5 Flash
('gemini-2.5-flash', 'qa', 0.87, 0.95, 0.08),
('gemini-2.5-flash', 'coding', 0.85, 0.9, 0.08),
('gemini-2.5-flash', 'reasoning', 0.86, 0.95, 0.08),
('gemini-2.5-flash', 'creative', 0.85, 0.95, 0.08)
ON CONFLICT (model_id, category) DO NOTHING;

-- Add scores for Anthropic models
INSERT INTO model_scores (model_id, category, accuracy, speed, cost) VALUES
-- Claude 3.5 Sonnet
('claude-3-5-sonnet-20241022', 'qa', 0.94, 0.75, 0.4),
('claude-3-5-sonnet-20241022', 'coding', 0.91, 0.7, 0.4),
('claude-3-5-sonnet-20241022', 'reasoning', 0.93, 0.75, 0.4),
('claude-3-5-sonnet-20241022', 'creative', 0.92, 0.75, 0.4),
-- Claude 3.5 Haiku
('claude-3-5-haiku-20241022', 'qa', 0.86, 0.95, 0.08),
('claude-3-5-haiku-20241022', 'coding', 0.83, 0.9, 0.08),
('claude-3-5-haiku-20241022', 'reasoning', 0.85, 0.95, 0.08),
('claude-3-5-haiku-20241022', 'creative', 0.84, 0.95, 0.08),
-- Claude 3 Opus
('claude-3-opus-20240229', 'qa', 0.96, 0.6, 0.8),
('claude-3-opus-20240229', 'coding', 0.93, 0.55, 0.8),
('claude-3-opus-20240229', 'reasoning', 0.95, 0.6, 0.8),
('claude-3-opus-20240229', 'creative', 0.94, 0.6, 0.8),
-- Claude 3 Sonnet
('claude-3-sonnet-20240229', 'qa', 0.92, 0.8, 0.4),
('claude-3-sonnet-20240229', 'coding', 0.89, 0.75, 0.4),
('claude-3-sonnet-20240229', 'reasoning', 0.91, 0.8, 0.4),
('claude-3-sonnet-20240229', 'creative', 0.90, 0.8, 0.4),
-- Claude 3 Haiku
('claude-3-haiku-20240307', 'qa', 0.84, 0.95, 0.08),
('claude-3-haiku-20240307', 'coding', 0.81, 0.9, 0.08),
('claude-3-haiku-20240307', 'reasoning', 0.83, 0.95, 0.08),
('claude-3-haiku-20240307', 'creative', 0.82, 0.95, 0.08)
ON CONFLICT (model_id, category) DO NOTHING;

-- Add scores for Groq models
INSERT INTO model_scores (model_id, category, accuracy, speed, cost) VALUES
-- Llama 3 70B
('llama3-70b-8192', 'qa', 0.89, 0.7, 0.15),
('llama3-70b-8192', 'coding', 0.86, 0.65, 0.15),
('llama3-70b-8192', 'reasoning', 0.88, 0.7, 0.15),
('llama3-70b-8192', 'creative', 0.87, 0.7, 0.15),
-- Llama 3 8B
('llama3-8b-8192', 'qa', 0.82, 0.9, 0.05),
('llama3-8b-8192', 'coding', 0.79, 0.85, 0.05),
('llama3-8b-8192', 'reasoning', 0.81, 0.9, 0.05),
('llama3-8b-8192', 'creative', 0.80, 0.9, 0.05),
-- Mixtral 8x7B
('mixtral-8x7b-32768', 'qa', 0.87, 0.75, 0.15),
('mixtral-8x7b-32768', 'coding', 0.84, 0.7, 0.15),
('mixtral-8x7b-32768', 'reasoning', 0.86, 0.75, 0.15),
('mixtral-8x7b-32768', 'creative', 0.85, 0.75, 0.15),
-- Gemma 2 9B
('gemma2-9b-it', 'qa', 0.83, 0.85, 0.08),
('gemma2-9b-it', 'coding', 0.80, 0.8, 0.08),
('gemma2-9b-it', 'reasoning', 0.82, 0.85, 0.08),
('gemma2-9b-it', 'creative', 0.81, 0.85, 0.08)
ON CONFLICT (model_id, category) DO NOTHING; 