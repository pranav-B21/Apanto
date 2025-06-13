const API_BASE_URL = 'http://localhost:8000';

export interface ChatRequest {
  prompt: string;
  priority: 'accuracy' | 'speed' | 'cost';
  model_id?: string;
}

export interface ChatResponse {
  response: string;
  model_used: string;
  task_type: string;
  confidence: number;
  response_time: number;
  tokens_used: number;
  estimated_cost: number;
  top_3_models: string[];
  is_local?: boolean;
}

export interface Enhancement {
  id: string;
  suggestion: string;
  type: string;
  confidence: number;
}

export interface ModelInfo {
  name: string;
  model_id: string;
  scores: Record<string, Record<string, number>>;
  is_local?: boolean;
}

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    const config = { ...defaultOptions, ...options };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Network error occurred');
    }
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getEnhancementSuggestions(prompt: string): Promise<Enhancement[]> {
    const response = await this.request<{ enhancements: Enhancement[] }>('/enhance-prompt', {
      method: 'POST',
      body: JSON.stringify({ prompt }),
    });
    return response.enhancements;
  }

  async hostModel(modelUrl: string, customName?: string): Promise<{ success: boolean; message: string; model_id?: string }> {
    return this.request<{ success: boolean; message: string; model_id?: string }>('/host-model', {
      method: 'POST',
      body: JSON.stringify({ model_url: modelUrl, custom_name: customName }),
    });
  }

  async getAvailableModels(): Promise<{ models: ModelInfo[]; count: number }> {
    return this.request<{ models: ModelInfo[]; count: number }>('/models');
  }

  async getAnalytics(): Promise<any> {
    return this.request<any>('/analytics/models');
  }

  async healthCheck(): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>('/health');
  }
}

export const apiService = new ApiService(); 