import { Search, Bell, User } from 'lucide-react';

export function Header() {
  return (
    <header className="h-20 flex items-center justify-between px-8 sticky top-0 z-40">
      <div className="flex-1 max-w-xl">
        <div className="relative group">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-blue-400 transition-colors" size={20} />
          <input
            type="text"
            placeholder="Search bugs, insights, or commands..."
            className="w-full bg-slate-900/50 border border-white/10 rounded-xl py-2.5 pl-10 pr-4 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 backdrop-blur-sm transition-all placeholder:text-slate-500"
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button className="relative p-2 text-slate-400 hover:text-white transition-colors rounded-xl hover:bg-white/5">
          <Bell size={20} />
          <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border-2 border-slate-950"></span>
        </button>
        <div className="flex items-center gap-3 pl-4 border-l border-white/10">
          <div className="text-right hidden md:block">
            <p className="text-sm font-medium text-white">Admin User</p>
            <p className="text-xs text-slate-400">Engineering Lead</p>
          </div>
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-slate-700 to-slate-600 border border-white/10 flex items-center justify-center">
            <User size={20} className="text-slate-300" />
          </div>
        </div>
      </div>
    </header>
  );
}