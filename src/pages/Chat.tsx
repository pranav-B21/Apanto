
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
      content: 'Hello! I\'m your AI <insert name> assistant. I automatically route your prompts to the best AI model and provide enhancement suggestions. What can I help you with today?',
      isUser: false,
      timestamp: new Date(),
      model: '<insert name> System',
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
  const [priority, setPriority] = useState<'accuracy' | 'speed' | 'cost'>('accuracy');
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
      // For now, simulate a response
      await new Promise(resolve => setTimeout(resolve, 1000));

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `I've routed your request to Llama 3.1 for optimal results. This model excels at general chat tasks. Here's a comprehensive response to your query...`,
        isUser: false,
        timestamp: new Date(),
        model: 'Llama 3.1',
        taskType: 'General Chat',
        confidence: 75,
        tokens: 150,
        cost: 0.002,
        responseTime: 1.2
      };

      setMessages(prev => [...prev, aiMessage]);
      
      toast({
        title: "Response Generated",
        description: `Used ${aiMessage.model} for ${aiMessage.taskType} task`,
      });

    } catch (error) {
      console.error('Chat error:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Sorry, I encountered an error. Please try again.`,
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
        description: "Failed to get response from AI.",
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

  const getCurrentTime = () => {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="min-h-screen bg-slate-900 flex text-white">
      {/* Left Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 bg-slate-800 border-r border-slate-700 overflow-hidden`}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Zap className="h-5 w-5 text-white" />
              </div>
              <h1 className="text-lg font-semibold text-white">
                <insert name>
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

          {/* Priority Selection */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Priority</h3>
            <div className="space-y-2">
              {(['accuracy', 'speed', 'cost'] as const).map((p) => (
                <Button
                  key={p}
                  variant={priority === p ? 'default' : 'outline'}
                  size="sm"
                  className={`w-full justify-start ${
                    priority === p 
                      ? 'bg-blue-600 text-white border-blue-600' 
                      : 'bg-transparent text-gray-300 border-slate-600 hover:bg-slate-700'
                  }`}
                  onClick={() => setPriority(p)}
                >
                  {p === 'accuracy' && <Target className="h-4 w-4 mr-2" />}
                  {p === 'speed' && <Zap className="h-4 w-4 mr-2" />}
                  {p === 'cost' && <DollarSign className="h-4 w-4 mr-2" />}
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </Button>
              ))}
            </div>
          </div>

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
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                  <Zap className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold text-white"><insert name></h2>
                  <p className="text-sm text-gray-400">AI-powered intelligent model selection</p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge className="bg-blue-600 text-white border-0">
                <Zap className="h-3 w-3 mr-1" />
                Llama 3.1
              </Badge>
              <Button variant="outline" size="sm" className="bg-transparent border-slate-600 text-gray-300 hover:bg-slate-700">
                <Plus className="h-4 w-4 mr-1" />
                Host Model
              </Button>
            </div>
          </div>
        </div>

        {/* Routing Status Panel */}
        {messages.length > 1 && (() => {
          const lastAiMessage = [...messages].reverse().find(m => !m.isUser);
          if (!lastAiMessage) return null;
          
          return (
            <div className="mx-4 mt-4">
              <Card className="bg-slate-800 border-slate-700">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 bg-blue-600 rounded flex items-center justify-center">
                        <Zap className="h-4 w-4 text-white" />
                      </div>
                      <CardTitle className="text-sm text-white">Intelligent Routing Active</CardTitle>
                    </div>
                    <Badge className="bg-green-600 text-white border-0">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      Optimized
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="flex items-center justify-center space-x-2 mb-1">
                        <Target className="h-4 w-4 text-blue-400" />
                        <span className="text-xs text-gray-400">Task Detected</span>
                      </div>
                      <div className="text-sm font-medium text-white">{lastAiMessage.taskType}</div>
                    </div>
                    <div className="text-center">
                      <div className="flex items-center justify-center space-x-2 mb-1">
                        <TrendingUp className="h-4 w-4 text-green-400" />
                        <span className="text-xs text-gray-400">Confidence</span>
                      </div>
                      <div className="text-sm font-medium text-white">{lastAiMessage.confidence}%</div>
                    </div>
                    <div className="text-center">
                      <div className="flex items-center justify-center space-x-2 mb-1">
                        <Zap className="h-4 w-4 text-purple-400" />
                        <span className="text-xs text-gray-400">Selected Model</span>
                      </div>
                      <div className="text-sm font-medium text-white">{lastAiMessage.model}</div>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-slate-700">
                    <div className="flex items-center space-x-2 text-xs text-gray-400">
                      <span className="text-blue-400">Reasoning:</span>
                      <span>Default routing for general conversation</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          );
        })()}

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
                      <AvatarFallback className="bg-blue-600 text-white">hu</AvatarFallback>
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
                    <div className="text-xs text-gray-400 mt-2">
                      <Clock className="h-3 w-3 inline mr-1" />
                      {getCurrentTime()}
                    </div>
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
