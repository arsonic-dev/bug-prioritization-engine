import { cn } from "@/lib/utils"
import { Bell, Search, User } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function Header() {
  return (
    <header className="h-16 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/50 flex items-center justify-between px-8 sticky top-0 z-40">
      <div className="flex items-center gap-4 flex-1">
        <div className="relative w-96">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
          <input
            type="text"
            placeholder="Search bugs, components, or reporters..."
            className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-10 pr-4 py-2 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/50"
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button className="relative p-2 text-slate-400 hover:text-slate-200 transition-colors">
          <Bell size={20} />
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
        </button>
        <div className="flex items-center gap-3 pl-4 border-l border-slate-800">
          <div className="text-right">
            <p className="text-sm font-medium text-slate-200">Alex Chen</p>
            <p className="text-xs text-slate-500">Engineering Manager</p>
          </div>
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-slate-700 to-slate-600 flex items-center justify-center">
            <User size={18} className="text-slate-300" />
          </div>
        </div>
      </div>
    </header>
  );
}