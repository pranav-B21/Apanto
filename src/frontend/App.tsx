
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Chat from "./pages/Chat";
import NotFound from "./pages/NotFound";
import { Button } from "@/components/ui/button";
import { Moon, Sun } from "lucide-react";
import { useState, useEffect } from "react";
import DarkModeToggle from "@/components/DarkModeToggle";

const queryClient = new QueryClient();

const App = () => {
  const [darkMode, setDarkMode] = useState(() => {
    // Check localStorage or system preference on initial load
    const saved = localStorage.getItem('darkMode');
    if (saved !== null) {
      return JSON.parse(saved);
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    // Save preference to localStorage
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    
    // Apply dark mode class with smooth transition
    document.documentElement.style.transition = 'background-color 0.3s ease, color 0.3s ease';
    
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }

    // Clean up transition after it completes
    const timeoutId = setTimeout(() => {
      document.documentElement.style.transition = '';
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={
              <Index 
                darkModeToggle={
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label="Toggle dark mode"
                    onClick={() => setDarkMode(!darkMode)}
                    className="ml-2 rounded-full text-xl"
                  >
                    {darkMode ? "☀️" : "🌙"}
                  </Button>
                }
              />
            } />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;