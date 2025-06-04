
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
  ThumbsUp,
  ThumbsDown,
  RotateCcw
} from 'lucide-react';

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

interface Enhancement {
  id: string;
  suggestion: string;
  type: string;
  confidence: number;
}

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m RouterAI. I automatically route your prompts to the best AI model and provide enhancement suggestions. What can I help you with today?',
      isUser: false,
      timestamp: new Date(),
      model: 'RouterAI System',
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
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [enhancements, setEnhancements] = useState<Enhancement[]>([]);
  const [showEnhancements, setShowEnhancements] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const conversations = [
    { id: '1', title: 'AI Model Routing', timestamp: '2 minutes ago', active: true },
    { id: '2', title: 'Code Generation Help', timestamp: '1 hour ago', active: false },
    { id: '3', title: 'Writing Assistant', timestamp: 'Yesterday', active: false },
  ];

  // Simulate real-time prompt enhancements
  useEffect(() => {
    if (currentMessage.length > 10) {
      const mockEnhancements: Enhancement[] = [
        {
          id: '1',
          suggestion: 'Add output format specification',
          type: 'format',
          confidence: 85
        },
        {
          id: '2', 
          suggestion: 'Include example of desired result',
          type: 'example',
          confidence: 78
        },
        {
          id: '3',
          suggestion: 'Specify programming language',
          type: 'context',
          confidence: 92
        }
      ];
      setEnhancements(mockEnhancements);
      setShowEnhancements(true);
    } else {
      setShowEnhancements(false);
    }
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
    setCurrentMessage('');
    setIsLoading(true);
    setShowEnhancements(false);

    // Simulate AI response with routing decision
    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `I've analyzed your request and routed it to the most suitable model. Based on the content, this appears to be a ${getTaskType(currentMessage)} task, so I've selected ${getSelectedModel(currentMessage)} for optimal results.\n\nHere's my response: This is a simulated AI response that would come from the selected model. The actual implementation would connect to real AI APIs and provide genuine responses based on intelligent routing decisions.`,
        isUser: false,
        timestamp: new Date(),
        model: getSelectedModel(currentMessage),
        taskType: getTaskType(currentMessage),
        confidence: Math.floor(Math.random() * 20) + 80,
        tokens: Math.floor(Math.random() * 200) + 50,
        cost: Math.random() * 0.05,
        responseTime: Math.random() * 2 + 0.5
      };

      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);
    }, 2000);
  };

  const getTaskType = (message: string): string => {
    if (message.toLowerCase().includes('code') || message.toLowerCase().includes('programming')) {
      return 'coding';
    } else if (message.toLowerCase().includes('write') || message.toLowerCase().includes('essay')) {
      return 'writing';
    } else if (message.toLowerCase().includes('math') || message.toLowerCase().includes('calculate')) {
      return 'math';
    } else if (message.toLowerCase().includes('creative') || message.toLowerCase().includes('story')) {
      return 'creative';
    }
    return 'general';
  };

  const getSelectedModel = (message: string): string => {
    const taskType = getTaskType(message);
    const models = {
      coding: 'GPT-4 Turbo',
      writing: 'Claude-3.5 Sonnet',
      math: 'GPT-4',
      creative: 'Claude-3 Opus',
      general: 'GPT-4'
    };
    return models[taskType as keyof typeof models] || 'GPT-4';
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
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-teal-50 flex">
      {/* Left Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 bg-white/80 backdrop-blur-md border-r border-white/20 overflow-hidden`}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <Link to="/" className="flex items-center space-x-2">
              <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                RouterAI
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
            <h3 className="text-sm font-medium text-gray-700 mb-3">Recent Conversations</h3>
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`p-3 rounded-lg cursor-pointer transition-colors ${
                  conv.active ? 'bg-purple-100 border border-purple-200' : 'hover:bg-gray-100'
                }`}
              >
                <div className="font-medium text-sm text-gray-900 truncate">{conv.title}</div>
                <div className="text-xs text-gray-500 flex items-center mt-1">
                  <Clock className="h-3 w-3 mr-1" />
                  {conv.timestamp}
                </div>
              </div>
            ))}
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
        <div className="bg-white/80 backdrop-blur-md border-b border-white/20 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {!sidebarOpen && (
                <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(true)}>
                  <Menu className="h-4 w-4" />
                </Button>
              )}
              <div>
                <h2 className="font-semibold text-gray-900">AI Model Router</h2>
                <p className="text-sm text-gray-600">Intelligent routing to the perfect AI model</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                ● Online
              </Badge>
            </div>
          </div>
        </div>

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
                      : 'bg-white/80 backdrop-blur-sm border border-white/20'
                  }`}>
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
                          <Button variant="ghost" size="sm">
                            <ThumbsUp className="h-3 w-3" />
                          </Button>
                          <Button variant="ghost" size="sm">
                            <ThumbsDown className="h-3 w-3" />
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
                  <div className="rounded-2xl p-4 bg-white/80 backdrop-blur-sm border border-white/20">
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
        <div className="p-4 bg-white/80 backdrop-blur-md border-t border-white/20">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <Textarea
                ref={textareaRef}
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                placeholder="Ask me anything... I'll route it to the perfect AI model"
                className="pr-12 min-h-[60px] resize-none bg-white/90 backdrop-blur-sm border-gray-200 focus:border-purple-500 focus:ring-purple-500"
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

      {/* Right Panel - Routing Insights */}
      <div className={`${rightPanelOpen ? 'w-80' : 'w-0'} transition-all duration-300 bg-white/80 backdrop-blur-md border-l border-white/20 overflow-hidden`}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-semibold text-gray-900">Routing Insights</h3>
            <Button variant="ghost" size="sm" onClick={() => setRightPanelOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          {messages.length > 1 && (
            <div className="space-y-4">
              {(() => {
                const lastAiMessage = [...messages].reverse().find(m => !m.isUser);
                if (!lastAiMessage) return null;

                return (
                  <>
                    <Card className="border-purple-200 bg-purple-50/50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm flex items-center">
                          <Target className="h-4 w-4 mr-2 text-purple-500" />
                          Current Request
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Task Type:</span>
                          <Badge variant="outline">{lastAiMessage.taskType}</Badge>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Selected Model:</span>
                          <span className="font-medium">{lastAiMessage.model}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Confidence:</span>
                          <span className="font-medium">{lastAiMessage.confidence}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Est. Time:</span>
                          <span className="font-medium">{lastAiMessage.responseTime?.toFixed(1)}s</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="border-green-200 bg-green-50/50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm flex items-center">
                          <BarChart3 className="h-4 w-4 mr-2 text-green-500" />
                          Performance
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Response Time:</span>
                          <span className="font-medium text-green-600">{lastAiMessage.responseTime?.toFixed(1)}s</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Tokens Used:</span>
                          <span className="font-medium">{lastAiMessage.tokens}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Cost:</span>
                          <span className="font-medium">${lastAiMessage.cost?.toFixed(4)}</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm">Alternative Models</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between text-gray-600">
                            <span>Claude-3.5 Sonnet</span>
                            <span>2.1s</span>
                          </div>
                          <div className="flex justify-between text-gray-600">
                            <span>GPT-4 Turbo</span>
                            <span>1.8s</span>
                          </div>
                          <div className="flex justify-between text-gray-600">
                            <span>Gemini Pro</span>
                            <span>2.5s</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </>
                );
              })()}
            </div>
          )}
        </div>
      </div>

      {/* Floating button to reopen right panel */}
      {!rightPanelOpen && (
        <Button
          onClick={() => setRightPanelOpen(true)}
          className="fixed right-4 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 shadow-lg"
          size="sm"
        >
          <ChevronUp className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
};

export default Chat;
