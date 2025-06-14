import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Link } from 'react-router-dom';
import { 
  Send, 
  Settings, 
  Menu,
  X,
  Home,
  MessageSquare,
  Clock,
  Target,
  Zap,
  DollarSign,
  ChevronUp,
  ChevronDown,
  Copy,
  RotateCcw,
  BarChart3,
  Plus,
  CheckCircle,
  Brain,
  TrendingUp,
  Upload
} from 'lucide-react';
import { apiService, type ChatRequest, type ChatResponse, type Enhancement, type ModelInfo } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  model?: string;
  taskType?: string;
  confidence?: number;
  tokens?: number;
  cost?: number; 
  responseTime?: number;
  isLocal?: boolean;
}

const Chat = () => {
  const { toast } = useToast();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI Smart Prompt Stream assistant. I automatically route your prompts to the best AI model and provide enhancement suggestions. What can I help you with today?',
      isUser: false,
      timestamp: new Date(),
      model: 'Smart Prompt Stream System',
      taskType: 'greeting',
      confidence: 100,
      tokens: 25,
      cost: 0,
      responseTime: 0.1
    }
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [enhancements, setEnhancements] = useState<Enhancement[]>([]);
  const [showEnhancements, setShowEnhancements] = useState(false);
  const [showHostForm, setShowHostForm] = useState(false);
  const [modelUrl, setModelUrl] = useState('');
  const [customName, setCustomName] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const conversations = [
    { id: '1', title: 'AI Model Routing', timestamp: '2 minutes ago', active: true },
    { id: '2', title: 'Code Generation Help', timestamp: '1 hour ago', active: false },
    { id: '3', title: 'Writing Assistant', timestamp: 'Yesterday', active: false },
  ];

  // Get real-time prompt enhancements from backend
  useEffect(() => {
    const getEnhancements = async () => {
      if (currentMessage.length > 10) {
        try {
          const suggestions = await apiService.getEnhancementSuggestions(currentMessage);
          setEnhancements(suggestions);
          setShowEnhancements(true);
        } catch (error) {
          console.error('Failed to get enhancement suggestions:', error);
          setShowEnhancements(false);
        }
      } else {
        setShowEnhancements(false);
      }
    };

    const timeoutId = setTimeout(getEnhancements, 500); // Debounce API calls
    return () => clearTimeout(timeoutId);
  }, [currentMessage]);

  useEffect(() => {
    // Load available models on component mount
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const { models } = await apiService.getAvailableModels();
      setAvailableModels(models);
    } catch (error) {
      console.error('Failed to load models:', error);
      toast({
        title: "Error",
        description: "Failed to load available models",
        variant: "destructive",
      });
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: currentMessage,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const messageToSend = currentMessage;
    setCurrentMessage('');
    setIsLoading(true);
    setShowEnhancements(false);

    try {
      const chatRequest: ChatRequest = {
        prompt: messageToSend,
        priority: 'accuracy',
        model_id: selectedModel || undefined
      };

      const response: ChatResponse = await apiService.chat(chatRequest);

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.response,
        isUser: false,
        timestamp: new Date(),
        model: response.model_used,
        taskType: response.task_type,
        confidence: Math.round(response.confidence),
        tokens: response.tokens_used,
        cost: response.estimated_cost,
        responseTime: response.response_time,
        isLocal: response.is_local
      };

      setMessages(prev => [...prev, aiMessage]);
      
      toast({
        title: "Response Generated",
        description: `Used ${response.model_used}${response.is_local ? ' (Local)' : ''} for ${response.task_type} task`,
      });

    } catch (error) {
      console.error('Failed to send message:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to generate response",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const applyEnhancement = (enhancement: Enhancement) => {
    const enhancedPrompt = `${currentMessage}\n\n[Enhancement: ${enhancement.suggestion}]`;
    setCurrentMessage(enhancedPrompt);
    setShowEnhancements(false);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied",
      description: "Message copied to clipboard",
    });
  };

  const formatTaskType = (taskType: string) => {
    return taskType.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const handleHostModel = async () => {
    if (!modelUrl.trim()) return;
    
    try {
      const response = await apiService.hostModel(modelUrl, customName);
      
      if (!response.success) {
        throw new Error(response.message || 'Failed to host model');
      }

      // Reset form
      setModelUrl('');
      setCustomName('');
      setShowHostForm(false);
      
      // Show success message
      toast({
        title: "Model Added",
        description: response.message || "Your model has been successfully added for hosting",
      });
      
      // Refresh models list
      loadAvailableModels();
      
    } catch (error) {
      console.error('Failed to host model:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to host model",
        variant: "destructive",
      });
    }
  };

  // Get the last AI message for routing insights
  const lastAiMessage = [...messages].reverse().find(m => !m.isUser);

  return (
    <div className="min-h-screen bg-background text-foreground flex">
      {/* Left Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 bg-background/80 backdrop-blur-md border-r border-border overflow-hidden`}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <Link to="/" className="flex items-center space-x-2">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Apanto
              </h1>
            </Link>
            <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          <Button className="w-full mb-6 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700">
            <MessageSquare className="h-4 w-4 mr-2" />
            New Chat
          </Button>

          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground mb-3">Recent Conversations</h3>
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`p-3 rounded-lg cursor-pointer transition-colors ${
                  conv.active ? 'bg-purple-100 border border-purple-200 dark:bg-purple-900/40 dark:border-purple-700' : 'hover:bg-muted dark:hover:bg-muted/40'
                }`}
              >
                <div className="font-medium text-sm text-foreground truncate">{conv.title}</div>
                <div className="text-xs text-muted-foreground flex items-center mt-1">
                  <Clock className="h-3 w-3 mr-1" />
                  {conv.timestamp}
                </div>
              </div>
            ))}
          </div>

          {/* Model Selection */}
          <div className="p-4 border-t border-border">
            <h3 className="text-sm font-medium text-foreground mb-2">Select Model</h3>
            <select
              value={selectedModel || ''}
              onChange={(e) => setSelectedModel(e.target.value || null)}
              className="w-full p-2 border border-border rounded-md text-sm bg-background text-foreground"
            >
              <option value="">Auto-select (Recommended)</option>
              {availableModels.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.name} {model.is_local ? '(Local)' : ''}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="absolute bottom-4 left-4 right-4">
          <Button variant="outline" className="w-full">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-background/80 backdrop-blur-md border-b border-border p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {!sidebarOpen && (
                <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(true)}>
                  <Menu className="h-4 w-4" />
                </Button>
              )}
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-blue-50 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-700">
                <Zap className="h-3 h-3 mr-1" />
                {lastAiMessage?.model || 'LLaMA3-70B'}
              </Badge>
              <Button 
                variant="outline" 
                className="bg-gradient-to-r from-purple-600 to-blue-600 text-white border-none hover:from-purple-700 hover:to-blue-700 hover:text-white"
                onClick={() => setShowHostForm(!showHostForm)}
              >
                <Plus className="h-4 w-4 mr-2" />
                Host Model
              </Button>
              <Badge variant="outline" className="bg-green-50 dark:bg-green-900/40 text-green-700 dark:text-green-300 border-green-200 dark:border-green-700">
                ● Online
              </Badge>
            </div>
          </div>
        </div>

        {/* Intelligent Routing Insights Panel */}
        {lastAiMessage && (
          <div className="p-4 border-b border-border">
            <Card className="bg-card/90 backdrop-blur-sm border-purple-200 dark:border-purple-700">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <CardTitle className="text-lg">Intelligent Routing Active</CardTitle>
                  </div>
                  <Badge variant="outline" className="bg-green-50 dark:bg-green-900/40 text-green-700 dark:text-green-300 border-green-200 dark:border-green-700">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Optimized
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-8">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/40 rounded-lg flex items-center justify-center">
                      <Target className="w-5 h-5 text-purple-600 dark:text-purple-300" />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Task Detected</p>
                      <p className="font-medium text-foreground">{formatTaskType(lastAiMessage.taskType || 'General Chat')}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-green-100 dark:bg-green-900/40 rounded-lg flex items-center justify-center">
                      <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-300" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-muted-foreground mb-2">Confidence</p>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-muted rounded-full h-2">
                          <div 
                            className="bg-green-500 dark:bg-green-400 h-2 rounded-full transition-all duration-300" 
                            style={{ width: `${lastAiMessage.confidence}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-foreground">{lastAiMessage.confidence}%</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/40 rounded-lg flex items-center justify-center">
                      <Brain className="w-5 h-5 text-blue-600 dark:text-blue-300" />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Selected Model</p>
                      <p className="font-medium text-foreground">{lastAiMessage.model}</p>
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 pt-4 border-t border-border">
                  <p className="text-sm text-muted-foreground">
                    <span className="text-purple-600 dark:text-purple-300 font-medium">Reasoning:</span> Default routing for general conversation
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Host Model Form */}
        {showHostForm && (
          <div className="p-4 border-b border-border">
            <Card className="bg-card/90 backdrop-blur-sm border-border">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Host Your Model</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowHostForm(false)}
                    className="text-muted-foreground hover:text-foreground"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="model-url" className="text-muted-foreground text-sm">
                      Hugging Face Model URL
                    </Label>
                    <Input
                      id="model-url"
                      type="text"
                      placeholder="https://huggingface.co/username/model-name"
                      value={modelUrl}
                      onChange={(e) => setModelUrl(e.target.value)}
                      className="bg-muted border-border text-foreground placeholder:text-muted-foreground focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="custom-name" className="text-muted-foreground text-sm">
                      Custom Model Name
                    </Label>
                    <Input
                      id="custom-name"
                      type="text"
                      placeholder="My Custom Model"
                      value={customName}
                      onChange={(e) => setCustomName(e.target.value)}
                      className="bg-muted border-border text-foreground placeholder:text-muted-foreground focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>

                  <div className="flex justify-end">
                    <Button
                      onClick={handleHostModel}
                      disabled={!modelUrl.trim()}
                      className="bg-blue-600 hover:bg-blue-700 text-white font-medium"
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      Host Model
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}
              >
                <div className={`flex space-x-3 max-w-3xl ${message.isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
                  <Avatar className="h-8 w-8">
                    {message.isUser ? (
                      <AvatarFallback className="bg-purple-500 text-white">U</AvatarFallback>
                    ) : (
                      <AvatarFallback className="bg-gradient-to-r from-purple-500 to-blue-500 text-white">AI</AvatarFallback>
                    )}
                  </Avatar>
                  <div className={`rounded-2xl p-4 ${
                    message.isUser 
                      ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white' 
                      : 'bg-card text-foreground border border-border'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      {!message.isUser && (
                        <>
                          <span className="text-sm font-medium">
                            {message.model}
                            {message.isLocal && (
                              <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">
                                Local
                              </span>
                            )}
                          </span>
                        </>
                      )}
                    </div>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    {!message.isUser && (
                      <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-200">
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>{message.model}</span>
                          <span>•</span>
                          <span>{message.responseTime?.toFixed(1)}s</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Button variant="ghost" size="sm" onClick={() => copyToClipboard(message.content)}>
                            <Copy className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start animate-fade-in">
                <div className="flex space-x-3 max-w-3xl">
                  <Avatar className="h-8 w-8">
                    <AvatarFallback className="bg-gradient-to-r from-purple-500 to-blue-500 text-white">AI</AvatarFallback>
                  </Avatar>
                  <div className="rounded-2xl p-4 bg-card text-foreground border border-border">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className="text-sm text-gray-600">Analyzing and routing...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Prompt Enhancements */}
        {showEnhancements && (
          <div className="mx-4 mb-2">
            <Card className="bg-white/90 backdrop-blur-sm border-purple-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center">
                  <Zap className="h-4 w-4 mr-2 text-purple-500" />
                  Prompt Enhancement Suggestions
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-2">
                  {enhancements.map((enhancement) => (
                    <div
                      key={enhancement.id}
                      className="flex items-center justify-between p-2 rounded-lg hover:bg-purple-50 cursor-pointer transition-colors"
                      onClick={() => applyEnhancement(enhancement)}
                    >
                      <span className="text-sm">{enhancement.suggestion}</span>
                      <Badge variant="secondary" className="text-xs">
                        {enhancement.confidence}%
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Input Area */}
        <div className="p-4 bg-card border-t border-border">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <Textarea
                ref={textareaRef}
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                placeholder="Ask me anything... I'll route it to the perfect AI model"
                className="pr-12 min-h-[60px] resize-none bg-card text-foreground border-border placeholder:text-muted-foreground focus:border-purple-500 focus:ring-purple-500"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
              />
              <Button
                onClick={handleSendMessage}
                disabled={!currentMessage.trim() || isLoading}
                className="absolute right-2 bottom-2 h-8 w-8 p-0 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
              <span>{currentMessage.length} characters</span>
              <span>Press Enter to send, Shift+Enter for new line</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
