-- Add Hugging Face specific columns to models table
ALTER TABLE models 
ADD COLUMN IF NOT EXISTS is_huggingface BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS huggingface_url TEXT;

-- Update existing records to have default values
UPDATE models 
SET is_huggingface = FALSE 
WHERE is_huggingface IS NULL; 