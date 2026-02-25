import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import type { ModelMetrics } from '@/types';
import { RefreshCw, Clock, Database, Gauge, GitBranch } from 'lucide-react';
import { format } from 'date-fns';

interface RetrainPanelProps {
  metrics: ModelMetrics;
  onRetrain: () => Promise<void>;
}

export function RetrainPanel({ metrics, onRetrain }: RetrainPanelProps) {
  const [isRetraining, setIsRetraining] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleRetrain = async () => {
    setIsRetraining(true);
    setProgress(0);

    const interval = setInterval(() => {
      setProgress(p => {
        if (p >= 90) {
          clearInterval(interval);
          return 90;
        }
        return p + 10;
      });
    }, 200);

    await onRetrain();

    clearInterval(interval);
    setProgress(100);
    setTimeout(() => {
      setIsRetraining(false);
      setProgress(0);
    }, 500);
  };

  return (
    <div className="grid grid-cols-2 gap-6">
      <Card className="p-6 bg-slate-900/50 border-slate-800">
        <h3 className="text-lg font-semibold text-white mb-6">Model Status</h3>

        <div className="space-y-6">
          <div className="flex items-center justify-between p-4 bg-slate-950 rounded-lg">
            <div className="flex items-center gap-3">
              <Clock className="text-slate-400" size={20} />
              <div>
                <p className="text-sm text-slate-400">Last Trained</p>
                <p className="text-white font-medium">
                  {format(new Date(metrics.last_trained), 'MMM d, yyyy HH:mm')}
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-950 rounded-lg">
            <div className="flex items-center gap-3">
              <GitBranch className="text-slate-400" size={20} />
              <div>
                <p className="text-sm text-slate-400">Model Version</p>
                <p className="text-white font-medium">{metrics.version}</p>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-950 rounded-lg">
            <div className="flex items-center gap-3">
              <Gauge className="text-slate-400" size={20} />
              <div>
                <p className="text-sm text-slate-400">F1 Macro Score</p>
                <p className="text-white font-medium">{(metrics.f1_macro * 100).toFixed(1)}%</p>
              </div>
            </div>
            <div className="w-32">
              <Progress value={metrics.f1_macro * 100} className="h-2" />
            </div>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-950 rounded-lg">
            <div className="flex items-center gap-3">
              <Database className="text-slate-400" size={20} />
              <div>
                <p className="text-sm text-slate-400">Training Dataset</p>
                <p className="text-white font-medium">{metrics.dataset_size.toLocaleString()} samples</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-6 border-t border-slate-800">
          <Button
            onClick={handleRetrain}
            disabled={isRetraining}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white"
          >
            <RefreshCw className={`mr-2 ${isRetraining ? 'animate-spin' : ''}`} size={18} />
            {isRetraining ? 'Retraining Model...' : 'Retrain Model'}
          </Button>
          {isRetraining && (
            <div className="mt-4">
              <Progress value={progress} className="h-2" />
              <p className="text-xs text-slate-500 mt-2 text-center">
                Processing new training data and validating model...
              </p>
            </div>
          )}
        </div>
      </Card>

      <Card className="p-6 bg-slate-900/50 border-slate-800">
        <h3 className="text-lg font-semibold text-white mb-6">Data Distribution</h3>
        <div className="h-64 flex items-end justify-around gap-2">
          {[
            { label: 'Critical', value: 15, color: 'bg-red-500' },
            { label: 'High', value: 28, color: 'bg-orange-500' },
            { label: 'Medium', value: 35, color: 'bg-yellow-500' },
            { label: 'Low', value: 22, color: 'bg-green-500' },
          ].map((item) => (
            <div key={item.label} className="flex flex-col items-center gap-2 flex-1">
              <div className="w-full bg-slate-800 rounded-t-lg relative overflow-hidden" style={{ height: `${item.value * 4}px` }}>
                <div className={`absolute bottom-0 w-full ${item.color} opacity-80`} style={{ height: '100%' }} />
              </div>
              <span className="text-xs text-slate-400">{item.label}</span>
              <span className="text-sm text-white font-medium">{item.value}%</span>
            </div>
          ))}
        </div>
        <p className="text-sm text-slate-500 mt-4 text-center">
          Current training data distribution by severity
        </p>
      </Card>
    </div>
  );
}