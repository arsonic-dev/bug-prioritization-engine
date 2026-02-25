import type { ModelMetrics } from '@/types';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { RefreshCw } from 'lucide-react';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface RetrainPanelProps {
  metrics: ModelMetrics;
  onRetrain: () => Promise<void>;
}

export function RetrainPanel({ metrics, onRetrain }: RetrainPanelProps) {
  const [isRetraining, setIsRetraining] = useState(false);

  const handleRetrain = async () => {
    setIsRetraining(true);
    await onRetrain();
    setIsRetraining(false);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <Card className="col-span-2 p-6 bg-white/5 border-white/10 backdrop-blur-md">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h3 className="text-xl font-semibold text-white">Model Performance</h3>
            <p className="text-slate-400 text-sm mt-1">Last trained: {new Date(metrics.last_trained).toLocaleDateString()}</p>
          </div>
          <Button
            onClick={handleRetrain}
            disabled={isRetraining}
            className="bg-blue-600 hover:bg-blue-500 text-white border-0"
          >
            {isRetraining ? (
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            Retrain Model
          </Button>
        </div>

        <div className="space-y-6">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Accuracy</span>
              <span className="text-white font-medium">{(metrics.accuracy * 100).toFixed(1)}%</span>
            </div>
            <Progress value={metrics.accuracy * 100} className="h-2 bg-white/10" />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Precision</span>
              <span className="text-white font-medium">{(metrics.precision * 100).toFixed(1)}%</span>
            </div>
            <Progress value={metrics.precision * 100} className="h-2 bg-white/10" />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Recall</span>
              <span className="text-white font-medium">{(metrics.recall * 100).toFixed(1)}%</span>
            </div>
            <Progress value={metrics.recall * 100} className="h-2 bg-white/10" />
          </div>
        </div>
      </Card>

      <AnimatePresence>
        {isRetraining && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="col-span-1"
          >
            <Card className="h-full p-6 bg-blue-600/10 border-blue-500/20 backdrop-blur-md flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 rounded-full bg-blue-500/20 flex items-center justify-center mb-4 animate-pulse">
                <RefreshCw className="h-8 w-8 text-blue-400 animate-spin" />
              </div>
              <h4 className="text-lg font-medium text-white mb-2">Retraining in Progress</h4>
              <p className="text-sm text-slate-400">Optimizing neural weights with latest dataset...</p>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}