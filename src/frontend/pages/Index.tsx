import React, { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Link } from 'react-router-dom';
import { 
  ArrowRight, 
  Zap, 
  Target, 
  Globe, 
  BarChart3, 
  Code, 
  DollarSign,
  CheckCircle,
  Star,
  Users,
  Shield
} from 'lucide-react';
import { apiService, ModelInfo } from '@/lib/api';
import DarkModeToggle from "@/components/DarkModeToggle";

interface IndexProps {
  darkMode: boolean;
  toggleDarkMode: () => void;
}

const Index: React.FC<IndexProps> = ({ darkMode, toggleDarkMode }) => {
  const features = [
    {
      icon: Target,
      title: "Smart Model Routing",
      description: "Auto-selects the best AI model for your specific task type and requirements"
    },
    {
      icon: Zap,
      title: "Real-time Prompt Enhancement",
      description: "Get instant suggestions to improve your prompts for better results"
    },
    {
      icon: Globe,
      title: "Multi-Provider Support",
      description: "Access OpenAI, Anthropic, Groq, and more - all in one unified interface"
    },
    {
      icon: BarChart3,
      title: "Performance Tracking",
      description: "See which models work best for your use cases with detailed analytics"
    },
    {
      icon: Code,
      title: "No-Code Integration",
      description: "Just chat naturally - we handle all the technical complexity behind the scenes"
    },
    {
      icon: DollarSign,
      title: "Cost Optimization",
      description: "Use expensive models only when needed, save money with smart routing"
    }
  ];

  const [models, setModels] = useState<ModelInfo[]>([]);
  useEffect(() => {
    apiService.getAvailableModels().then(res => setModels(res.models)).catch(() => setModels([]));
  }, []);

  const pricingTiers = [
    {
      name: "Free",
      price: "$0",
      period: "/month",
      description: "Perfect for getting started",
      features: [
        "100 requests per day",
        "Access to open-source models",
        "Basic routing intelligence",
        "Community support"
      ],
      cta: "Start Free",
      popular: false
    },
    {
      name: "Pro",
      price: "$29",
      period: "/month",
      description: "For power users and professionals",
      features: [
        "Unlimited requests",
        "All premium models",
        "Priority routing",
        "Advanced analytics",
        "Custom model preferences",
        "Email support"
      ],
      cta: "Start Pro Trial",
      popular: true
    },
    {
      name: "Enterprise",
      price: "Custom",
      period: "",
      description: "For teams and organizations",
      features: [
        "Custom deployment",
        "SLA guarantees",
        "Dedicated support",
        "Custom integrations",
        "Advanced security",
        "Usage analytics"
      ],
      cta: "Contact Sales",
      popular: false
    }
  ];

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Navigation */}
      <nav className="bg-background/80 backdrop-blur-md border-b border-border sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link to="/" className="flex items-center space-x-2">
                <h1 className="text-2xl font-bold text-blue-600">
                  Apanto
                </h1>
              </Link>
            </div>
            <div className="flex items-center space-x-4">
              <Link to="/chat" className="text-foreground hover:text-purple-600 dark:hover:text-blue-400 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                Chat
              </Link>
              <a href="#models" className="text-foreground hover:text-purple-600 dark:hover:text-blue-400 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                Models
              </a>
              <a href="#pricing" className="text-foreground hover:text-purple-600 dark:hover:text-blue-400 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                Pricing
              </a>
              <DarkModeToggle darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
              <Button variant="outline" className="mr-2">
                Sign In
              </Button>
              <Button className="bg-blue-600 hover:bg-blue-700 text-white">
                Get Started
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <h1 className="text-5xl md:text-7xl font-bold text-foreground mb-6">
              Stop Switching Between{' '}
              <span className="bg-gradient-to-r from-purple-600 via-blue-600 to-teal-600 bg-clip-text text-transparent">
                AI Models
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              Let Apanto choose the perfect AI model for your task - automatically. 
              Intelligent routing, real-time optimization, all in one interface.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link to="/chat">
                <Button 
                  size="lg" 
                  className="bg-blue-600 hover:bg-blue-700 text-lg px-8 py-3 group text-white"
                >
                  Start Chatting
                  <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Button 
                variant="outline" 
                size="lg" 
                className="text-lg px-8 py-3"
                onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
              >
                See How It Works
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Problem Statement */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-card/50 dark:bg-card/80">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
              The Current AI Landscape is Fragmented
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Developers and businesses waste time manually switching between different AI models, 
              trying to figure out which one works best for each task.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <Card className="p-6 border-red-200 bg-red-50/50 dark:bg-red-900/40 dark:border-red-700">
              <CardHeader>
                <CardTitle className="text-red-800 dark:text-red-300 text-xl">Before Apanto</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 dark:bg-red-300 rounded-full mt-2"></div>
                  <span className="text-red-700 dark:text-red-200">Manual model selection for every task</span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 dark:bg-red-300 rounded-full mt-2"></div>
                  <span className="text-red-700 dark:text-red-200">Trial-and-error to find the right AI</span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 dark:bg-red-300 rounded-full mt-2"></div>
                  <span className="text-red-700 dark:text-red-200">Constant switching between platforms</span>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 dark:bg-red-300 rounded-full mt-2"></div>
                  <span className="text-red-700 dark:text-red-200">Suboptimal prompts and results</span>
                </div>
              </CardContent>
            </Card>

            <Card className="p-6 border-green-200 bg-green-50/50 dark:bg-green-900/40 dark:border-green-700">
              <CardHeader>
                <CardTitle className="text-green-800 dark:text-green-300 text-xl">After Apanto</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-300 mt-0.5" />
                  <span className="text-green-700 dark:text-green-200">Automatic model selection</span>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-300 mt-0.5" />
                  <span className="text-green-700 dark:text-green-200">AI-powered prompt optimization</span>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-300 mt-0.5" />
                  <span className="text-green-700 dark:text-green-200">One unified interface for all models</span>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-300 mt-0.5" />
                  <span className="text-green-700 dark:text-green-200">Consistently better results</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
              Why Choose Apanto?
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Experience the future of AI interaction with intelligent routing and optimization.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <Card
                key={index}
                className="p-6 border border-border bg-white/80 dark:bg-zinc-900/80 transition-all duration-300 group hover:bg-gray-100 dark:hover:bg-zinc-800 hover:shadow-xl hover:border-purple-400 dark:hover:border-purple-600 cursor-pointer backdrop-blur-sm"
              >
                <CardHeader>
                  <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <feature.icon className="h-6 w-6 text-foreground" />
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-muted-foreground text-base">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-20 px-4 sm:px-6 lg:px-8 bg-card/50 dark:bg-card/80">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
              How Apanto Works
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Three simple steps to get the perfect AI response every time.
            </p>
          </div>
          <div className="flex flex-col md:flex-row items-center justify-center gap-8 md:gap-0 relative">
            {/* Step 1 */}
            <div className="flex flex-col items-center flex-1 min-w-[220px]">
              <h3 className="text-xl font-semibold mb-3 text-foreground">Type Your Prompt</h3>
              <p className="text-muted-foreground text-center max-w-xs">Write your question or request naturally. Our AI analyzes your intent and task type in real-time.</p>
            </div>
            {/* Arrow 1 */}
            <div className="hidden md:flex flex-col items-center justify-center flex-none mx-4">
              <svg className="w-16 h-16 text-purple-400 dark:text-purple-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 48 48">
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 24h32m0 0l-8-8m8 8l-8 8" />
              </svg>
            </div>
            {/* Step 2 */}
            <div className="flex flex-col items-center flex-1 min-w-[220px]">
              <h3 className="text-xl font-semibold mb-3 text-foreground">AI Routes Intelligently</h3>
              <p className="text-muted-foreground text-center max-w-xs">Apanto selects the best model for your specific task and enhances your prompt for optimal results.</p>
            </div>
            {/* Arrow 2 */}
            <div className="hidden md:flex flex-col items-center justify-center flex-none mx-4">
              <svg className="w-16 h-16 text-purple-400 dark:text-purple-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 48 48">
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 24h32m0 0l-8-8m8 8l-8 8" />
              </svg>
            </div>
            {/* Step 3 */}
            <div className="flex flex-col items-center flex-1 min-w-[220px]">
              <h3 className="text-xl font-semibold mb-3 text-foreground">Get Perfect Results</h3>
              <p className="text-muted-foreground text-center max-w-xs">Receive optimized responses faster than ever, with full transparency into the routing decision.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Supported Models */}
      <section id="models" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
              Supported AI Models
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Access the best AI models organized by task category. More models added weekly.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {models.map((model, index) => (
              <Card
                key={index}
                className="p-6 border border-border bg-white/80 dark:bg-zinc-900/80 transition-all duration-300 hover:bg-gray-100 dark:hover:bg-zinc-800 hover:shadow-xl hover:border-purple-400 dark:hover:border-purple-600 cursor-pointer backdrop-blur-sm"
              >
                <CardHeader>
                  <CardTitle className="text-lg">{model.name}</CardTitle>
                  <div className="text-xs text-muted-foreground">ID: {model.model_id}</div>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {Object.keys(model.scores).map((cat, i) => (
                      <Badge key={i} variant="secondary" className="text-sm">
                        {cat}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <div className="text-center mt-12">
            <Badge variant="outline" className="text-lg px-4 py-2">
              <Star className="w-4 h-4 mr-2" />
              More models added weekly
            </Badge>
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="py-20 px-4 sm:px-6 lg:px-8 bg-card/50 dark:bg-card/80">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
              Simple, Transparent Pricing
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Choose the plan that fits your needs. All plans include intelligent routing and prompt optimization.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {pricingTiers.map((tier, index) => (
              <Card key={index} className={`p-6 relative ${tier.popular ? 'border-purple-500 shadow-lg scale-105' : 'border-border'} bg-card/80 backdrop-blur-sm`}>
                {tier.popular && (
                  <Badge className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-purple-500 to-blue-500">
                    Most Popular
                  </Badge>
                )}
                <CardHeader>
                  <CardTitle className="text-2xl">{tier.name}</CardTitle>
                  <div className="flex items-baseline">
                    <span className="text-4xl font-bold">{tier.price}</span>
                    <span className="text-muted-foreground ml-1">{tier.period}</span>
                  </div>
                  <CardDescription>{tier.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3 mb-6">
                    {tier.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                        <span className="text-muted-foreground">{feature}</span>
                      </li>
                    ))}
                  </ul>
                  <Button 
                    className={`w-full ${tier.popular ? 'bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600' : ''}`}
                    variant={tier.popular ? 'default' : 'outline'}
                  >
                    {tier.cta}
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-background text-foreground py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <h3 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-4">
                Apanto
              </h3>
              <p className="text-muted-foreground">
                Intelligent AI model routing for the modern web.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-muted-foreground">
                <li><a href="#" className="hover:text-foreground transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Pricing</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">API</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Documentation</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-muted-foreground">
                <li><a href="#" className="hover:text-foreground transition-colors">About</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Careers</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Contact</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-muted-foreground">
                <li><a href="#" className="hover:text-foreground transition-colors">Help Center</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Status</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Privacy</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Terms</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-border mt-8 pt-8 text-center text-muted-foreground">
            <p>&copy; 2024 Apanto. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
