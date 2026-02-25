import { useState, useEffect } from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { StatCard } from '@/components/dashboard/StatCard';
import { RankedTable } from '@/components/dashboard/RankedTable';
import { BugDetailModal } from '@/components/dashboard/BugDetailModal';
import { AIInsights } from '@/components/dashboard/AIInsights';
import { RetrainPanel } from '@/components/ml/RetrainPanel';
import { AnalyticsView } from '@/components/analytics/AnalyticsView';
import { api } from '@/lib/api';
import type { Bug, DashboardStats, ModelMetrics, Insight } from '@/types';
import {
  Flame,
  AlertTriangle,
  TrendingDown,
  Brain,
  Activity
} from 'lucide-react';
import { Toaster, toast } from 'sonner';

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const [bugs, setBugs] = useState<Bug[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedBug, setSelectedBug] = useState<Bug | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [bugsData, statsData, insightsData, metricsData] = await Promise.all([
        api.getRankedQueue(),
        api.getDashboardStats(),
        api.getInsights(),
        api.getModelMetrics(),
      ]);
      setBugs(bugsData);
      setStats(statsData);
      setInsights(insightsData);
      setMetrics(metricsData);
    } catch (error) {
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const handleBugClick = (bug: Bug) => {
    setSelectedBug(bug);
    setModalOpen(true);
  };

  const handleRetrain = async () => {
    try {
      await api.retrain();
      toast.success('Model retraining completed successfully');
      const newMetrics = await api.getModelMetrics();
      setMetrics(newMetrics);
    } catch (error) {
      toast.error('Failed to retrain model');
    }
  };

  const renderDashboard = () => (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-6">Dashboard Overview</h2>
        <div className="grid grid-cols-4 gap-6">
          <StatCard
            title="Highest Priority"
            value={stats?.highest_priority.jira_key || '-'}
            subtitle={stats?.highest_priority.summary || ''}
            icon={<Flame className="text-red-400" size={24} />}
            variant="critical"
          />
          <StatCard
            title="High Risk Bugs"
            value={stats?.high_risk_count || 0}
            subtitle="Require immediate attention"
            icon={<AlertTriangle className="text-orange-400" size={24} />}
            variant="warning"
            trend="down"
            trendValue="12%"
          />
          <StatCard
            title="Low Impact Bugs"
            value={stats?.low_impact_count || 0}
            subtitle="Can be deprioritized"
            icon={<TrendingDown className="text-green-400" size={24} />}
            variant="success"
          />
          <StatCard
            title="Avg Confidence"
            value={stats ? `${(stats.avg_confidence * 100).toFixed(0)}%` : '-'}
            subtitle="Model prediction confidence"
            icon={<Brain className="text-blue-400" size={24} />}
          />
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold text-white mb-6">AI Insights</h2>
        <AIInsights insights={insights} />
      </div>

      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">Ranked Bug Queue</h2>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Activity size={16} />
            <span>Live updates</span>
          </div>
        </div>
        <RankedTable
          bugs={bugs}
          onBugClick={handleBugClick}
          loading={loading}
        />
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeView) {
      case 'dashboard':
        return renderDashboard();
      case 'queue':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white">Full Bug Queue</h2>
            <RankedTable bugs={bugs} onBugClick={handleBugClick} loading={loading} />
          </div>
        );
      case 'analytics':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white">Analytics & Reporting</h2>
            <AnalyticsView />
          </div>
        );
      case 'ml':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white">ML Model Management</h2>
            {metrics && <RetrainPanel metrics={metrics} onRetrain={handleRetrain} />}
          </div>
        );
      default:
        return renderDashboard();
    }
  };

  return (
    <>
      <DashboardLayout activeView={activeView} onViewChange={setActiveView}>
        {renderContent()}
      </DashboardLayout>
      <BugDetailModal
        bug={selectedBug}
        open={modalOpen}
        onOpenChange={setModalOpen}
      />
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: '#0f172a',
            border: '1px solid #1e293b',
            color: '#e2e8f0',
          },
        }}
      />
    </>
  );
}

export default App;