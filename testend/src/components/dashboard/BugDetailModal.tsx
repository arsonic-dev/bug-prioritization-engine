import type { Bug } from '@/types';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

interface BugDetailModalProps {
  bug: Bug | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function BugDetailModal({ bug, open, onOpenChange }: BugDetailModalProps) {
  if (!bug) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg bg-slate-900 border-white/10 text-slate-200 backdrop-blur-xl">
        <DialogHeader>
          <div className="flex items-center gap-3 mb-2">
            <Badge variant="outline" className="text-blue-400 border-blue-400/30 bg-blue-400/10">
              {bug.jira_key}
            </Badge>
            <span className="text-xs text-slate-500">{bug.created_at}</span>
          </div>
          <DialogTitle className="text-xl text-white">{bug.summary}</DialogTitle>
          <DialogDescription className="text-slate-400">
            AI-generated priority analysis and details.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <p className="text-xs text-slate-500 mb-1">AI Confidence</p>
              <p className="text-2xl font-bold text-white">{(bug.ai_confidence * 100).toFixed(0)}%</p>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <p className="text-xs text-slate-500 mb-1">Priority Score</p>
              <p className="text-2xl font-bold text-white">{bug.priority_score}</p>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Severity</span>
              <span className="text-white font-medium">{bug.severity}</span>
            </div>
            <Separator className="bg-white/10" />
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Status</span>
              <span className="text-white font-medium">{bug.status}</span>
            </div>
            <Separator className="bg-white/10" />
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Assignee</span>
              <span className="text-white font-medium">{bug.assignee}</span>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}