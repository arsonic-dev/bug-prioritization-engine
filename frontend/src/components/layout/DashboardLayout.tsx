import { Sidebar } from './Sidebar';
import { Header } from './Header';

interface DashboardLayoutProps {
  children: React.ReactNode;
  activeView: string;
  onViewChange: (view: string) => void;
}

export function DashboardLayout({ children, activeView, onViewChange }: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <Sidebar activeView={activeView} onViewChange={onViewChange} />
      <div className="ml-64">
        <Header />
        <main className="p-8 max-w-7xl mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}