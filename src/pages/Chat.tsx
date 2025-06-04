import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
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
  TrendingUp
} from 'lucide-react';
import { apiService, type ChatRequest, type ChatResponse, type Enhancement } from '@/lib/api';
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
}

const Chat = () => {
  const { toast } = useToast();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI Insert Name assistant. I automatically route your prompts to the best AI model and provide enhancement suggestions. What can I help you with today?',
      isUser: false,
      timestamp: new Date(),
      model: 'Insert Name System',
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
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

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
        priority: 'accuracy' // Default to accuracy since we removed selection
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
        responseTime: response.response_time
      };

      setMessages(prev => [...prev, aiMessage]);
      
      toast({
        title: "Response Generated",
        description: `Used ${response.model_used} for ${response.task_type} task`,
      });

    } catch (error) {
      console.error('Chat error:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please make sure the backend is running and try again.`,
        isUser: false,
        timestamp: new Date(),
        model: 'Error Handler',
        taskType: 'error',
        confidence: 0,
        tokens: 0,
        cost: 0,
        responseTime: 0
      };

      setMessages(prev => [...prev, errorMessage]);
      
      toast({
        title: "Error",
        description: "Failed to get response from AI. Check if backend is running.",
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

  // Get the last AI message for routing insights
  const lastAiMessage = [...messages].reverse().find(m => !m.isUser);

  return (
    <div className="min-h-screen bg-slate-900 flex text-white">
      {/* Left Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 bg-slate-800 border-r border-slate-700 overflow-hidden`}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <Link to="/" className="flex items-center space-x-2">
              <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Insert Name
              </h1>
            </Link>
            <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(false)} className="text-gray-400 hover:text-white">
              <X className="h-4 w-4" />
            </Button>
          </div>

          <Button className="w-full mb-6 bg-blue-600 hover:bg-blue-700 text-white">
            <Plus className="h-4 w-4 mr-2" />
            New Chat
          </Button>

          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Recent Conversations</h3>
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`p-3 rounded-lg cursor-pointer transition-colors ${
                  conv.active ? 'bg-slate-700 border border-slate-600' : 'hover:bg-slate-700'
                }`}
              >
                <div className="font-medium text-sm text-white truncate">{conv.title}</div>
                <div className="text-xs text-gray-400 flex items-center mt-1">
                  <Clock className="h-3 w-3 mr-1" />
                  {conv.timestamp}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="absolute bottom-4 left-4 right-4">
          <Button variant="outline" className="w-full bg-transparent border-slate-600 text-gray-300 hover:bg-slate-700">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-slate-800 border-b border-slate-700 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {!sidebarOpen && (
                <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(true)} className="text-gray-400 hover:text-white">
                  <Menu className="h-4 w-4" />
                </Button>
              )}
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                <Zap className="h-3 h-3 mr-1" />
                {lastAiMessage?.model || 'Llama 3.1'}
              </Badge>
              <Button variant="outline" className="bg-gradient-to-r from-purple-600 to-blue-600 text-white border-none hover:from-purple-700 hover:to-blue-700">
                <Plus className="h-4 w-4 mr-2" />
                Host Model
              </Button>
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                ● Online
              </Badge>
            </div>
          </div>
        </div>

        {/* Intelligent Routing Insights Panel */}
        {lastAiMessage && (
          <div className="p-4 border-b border-white/20">
            <Card className="bg-white/90 backdrop-blur-sm border-purple-200">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <CardTitle className="text-lg">Intelligent Routing Active</CardTitle>
                  </div>
                  <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Optimized
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-8">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                      <Target className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Task Detected</p>
                      <p className="font-medium text-gray-900">{formatTaskType(lastAiMessage.taskType || 'General Chat')}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                      <TrendingUp className="w-5 h-5 text-green-600" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-gray-600 mb-2">Confidence</p>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-green-500 h-2 rounded-full transition-all duration-300" 
                            style={{ width: `${lastAiMessage.confidence}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-gray-900">{lastAiMessage.confidence}%</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                      <Brain className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Selected Model</p>
                      <p className="font-medium text-gray-900">{lastAiMessage.model}</p>
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <p className="text-sm text-gray-600">
                    <span className="text-purple-600 font-medium">Reasoning:</span> Default routing for general conversation
                  </p>
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
                      <AvatarFallback className="bg-blue-600 text-white">U</AvatarFallback>
                    ) : (
                      <AvatarFallback className="bg-purple-600 text-white">
                        <Zap className="h-4 w-4" />
                      </AvatarFallback>
                    )}
                  </Avatar>
                  <div className={`rounded-2xl p-4 ${
                    message.isUser 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-slate-800 border border-slate-700 text-white'
                  }`}>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    {!message.isUser && (
                      <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-700">
                        <div className="flex items-center space-x-4 text-xs text-gray-400">
                          <span className="flex items-center space-x-1">
                            <Zap className="h-3 w-3" />
                            <span>{message.model}</span>
                          </span>
                          <span>•</span>
                          <span>{message.taskType}</span>
                          <span>•</span>
                          <span>{message.confidence}% confidence</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Button variant="ghost" size="sm" onClick={() => copyToClipboard(message.content)} className="text-gray-400 hover:text-white">
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
                    <AvatarFallback className="bg-purple-600 text-white">
                      <Zap className="h-4 w-4" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="rounded-2xl p-4 bg-slate-800 border border-slate-700">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className="text-sm text-gray-400">Analyzing and routing...</span>
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
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center text-white">
                  <Zap className="h-4 w-4 mr-2 text-blue-400" />
                  Prompt Enhancement Suggestions
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-2">
                  {enhancements.map((enhancement) => (
                    <div
                      key={enhancement.id}
                      className="flex items-center justify-between p-2 rounded-lg hover:bg-slate-700 cursor-pointer transition-colors"
                      onClick={() => applyEnhancement(enhancement)}
                    >
                      <span className="text-sm text-gray-300">{enhancement.suggestion}</span>
                      <Badge variant="secondary" className="text-xs bg-slate-700 text-gray-300">
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
        <div className="p-4 bg-slate-800 border-t border-slate-700">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <Textarea
                ref={textareaRef}
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                placeholder="Ask me anything... I'll route it to the perfect AI model"
                className="pr-12 min-h-[60px] resize-none bg-slate-900 border-slate-600 focus:border-blue-500 focus:ring-blue-500 text-white placeholder-gray-400"
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
                className="absolute right-2 bottom-2 h-8 w-8 p-0 bg-blue-600 hover:bg-blue-700"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex items-center justify-between mt-2 text-xs text-gray-400">
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
