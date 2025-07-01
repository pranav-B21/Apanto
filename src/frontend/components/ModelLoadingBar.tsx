import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { X, Download, Brain, CheckCircle, AlertCircle } from 'lucide-react';

interface ModelProgress {
  stage: string;
  message: string;
  progress: number;
  memory_usage?: number;
  model_info?: {
    type: string;
    parameters: number;
  };
  error?: string;
}

interface ModelLoadingBarProps {
  isVisible: boolean;
  onClose: () => void;
}

const ModelLoadingBar: React.FC<ModelLoadingBarProps> = ({ isVisible, onClose }) => {
  const [progress, setProgress] = useState<ModelProgress | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    if (!isVisible) return;

    // Connect to WebSocket
    const websocket = new WebSocket('ws://localhost:8000/ws');
    
    websocket.onopen = () => {
      console.log('WebSocket connected for model loading progress');
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'model_progress') {
          setProgress(data.data);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, [isVisible]);

  if (!isVisible || !progress) return null;

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case 'starting':
        return <Download className="h-4 w-4" />;
      case 'tokenizer':
      case 'tokenizer_complete':
        return <Brain className="h-4 w-4" />;
      case 'config':
      case 'config_complete':
        return <Brain className="h-4 w-4" />;
      case 'model_download':
      case 'model_complete':
        return <Download className="h-4 w-4" />;
      case 'complete':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Brain className="h-4 w-4" />;
    }
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'complete':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-blue-600';
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80">
      <Card className="bg-background/95 backdrop-blur-sm border-border shadow-lg">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              {getStageIcon(progress.stage)}
              <span className={`font-medium text-sm ${getStageColor(progress.stage)}`}>
                Model Loading
              </span>
            </div>
            <button
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="space-y-3">
            {/* Progress Bar */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{progress.message}</span>
                <span>{progress.progress}%</span>
              </div>
              <Progress value={progress.progress} className="h-2" />
            </div>

            {/* Memory Usage */}
            {progress.memory_usage && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Memory Usage:</span>
                <Badge variant="outline" className="text-xs">
                  {progress.memory_usage.toFixed(1)} MB
                </Badge>
              </div>
            )}

            {/* Model Info */}
            {progress.model_info && (
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Model Type:</span>
                  <span>{progress.model_info.type}</span>
                </div>
                {progress.model_info.parameters && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Parameters:</span>
                    <span>{progress.model_info.parameters.toLocaleString()}</span>
                  </div>
                )}
              </div>
            )}

            {/* Error Message */}
            {progress.error && (
              <div className="text-xs text-red-600 bg-red-50 dark:bg-red-900/20 p-2 rounded">
                {progress.error}
              </div>
            )}

            {/* Stage Badge */}
            <div className="flex justify-end">
              <Badge variant="secondary" className="text-xs">
                {progress.stage.replace('_', ' ')}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelLoadingBar; 