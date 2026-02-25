import { LayoutDashboard, ListOrdered, BarChart3, Brain, Settings } from 'lucide-react';
import { motion } from 'framer-motion';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

const navItems = [
  { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
  { id: 'queue', label: 'Bug Queue', icon: ListOrdered },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'ml', label: 'ML Model', icon: Brain },
];

export function Sidebar({ activeView, onViewChange }: SidebarProps) {
  return (
    <motion.div
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      className="fixed left-4 top-4 bottom-4 w-64 z-50 flex flex-col"
    >
      <div className="flex-1 backdrop-blur-xl bg-slate-950/50 border border-white/10 rounded-3xl p-4 flex flex-col shadow-2xl shadow-black/50">
        <div className="flex items-center gap-3 px-4 py-6 mb-6">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <span className="font-bold text-white text-lg">C</span>
          </div>
          <span className="text-xl font-semibold text-white tracking-tight">COSMOQ</span>
        </div>

        <nav className="space-y-2 flex-1">
          {navItems.map((item) => {
            const isActive = activeView === item.id;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => onViewChange(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group ${
                  isActive
                    ? 'bg-white/10 text-white shadow-lg shadow-white/5'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon size={20} className={isActive ? 'text-blue-400' : 'group-hover:text-blue-300'} />
                <span className="font-medium">{item.label}</span>
                {isActive && (
                  <motion.div
                    layoutId="active-pill"
                    className="absolute left-0 w-1 h-8 bg-blue-500 rounded-r-full"
                  />
                )}
              </button>
            );
          })}
        </nav>

        <div className="mt-auto pt-6 border-t border-white/10">
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 hover:text-white hover:bg-white/5 transition-all">
            <Settings size={20} />
            <span className="font-medium">Settings</span>
          </button>
        </div>
      </div>
    </motion.div>
  );
}