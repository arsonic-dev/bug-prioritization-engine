import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { motion } from 'framer-motion';

interface DashboardLayoutProps {
  children: React.ReactNode;
  activeView: string;
  onViewChange: (view: string) => void;
}

export function DashboardLayout({ children, activeView, onViewChange }: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 overflow-hidden relative selection:bg-blue-500/30">
      {/* Aurora Background Effects */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-20%] left-[-10%] w-[70%] h-[70%] bg-blue-600/20 rounded-full blur-[120px] animate-aurora" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] bg-orange-600/10 rounded-full blur-[100px] animate-aurora" style={{ animationDelay: '-5s' }} />
        <div className="absolute top-[40%] left-[30%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[90px] animate-aurora" style={{ animationDelay: '-10s' }} />
      </div>

      <Sidebar activeView={activeView} onViewChange={onViewChange} />

      <div className="ml-72 relative z-10">
        <Header />
        <main className="p-8 pt-4 max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  );
}