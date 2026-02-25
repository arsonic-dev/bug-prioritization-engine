import { useState } from 'react';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  List,
  BarChart3,
  Brain,
  Settings,
  Bug
} from 'lucide-react';

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={20} /> },
  { id: 'queue', label: 'Ranked Queue', icon: <List size={20} /> },
  { id: 'analytics', label: 'Analytics', icon: <BarChart3 size={20} /> },
  { id: 'ml', label: 'ML Model', icon: <Brain size={20} /> },
];

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

export function Sidebar({ activeView, onViewChange }: SidebarProps) {
  return (
    <aside className="w-64 h-screen bg-slate-950 border-r border-slate-800/50 fixed left-0 top-0 flex flex-col">
      <div className="p-6 flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
          <Bug className="text-white" size={24} />
        </div>
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">BugRank</h1>
          <p className="text-xs text-slate-400">AI Prioritization</p>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onViewChange(item.id)}
            className={cn(
              'w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200',
              activeView === item.id
                ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            )}
          >
            {item.icon}
            {item.label}
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-800/50">
        <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-900 transition-colors">
          <Settings size={20} />
          Settings
        </button>
      </div>
    </aside>
  );
}