import { Moon, Sun } from "lucide-react";

export default function DarkModeToggle({ darkMode, toggleDarkMode }: { darkMode: boolean, toggleDarkMode: () => void }) {
  return (
    <button
      className="p-2 rounded-full bg-gray-200 dark:bg-zinc-800 hover:bg-gray-300 dark:hover:bg-zinc-700 transition"
      onClick={toggleDarkMode}
      aria-label="Toggle dark mode"
    >
      {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
    </button>
  );
} 