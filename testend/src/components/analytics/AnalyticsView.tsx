import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip
} from "recharts"
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const data = [
  { name: 'Mon', bugs: 12, resolved: 8 },
  { name: 'Tue', bugs: 19, resolved: 14 },
  { name: 'Wed', bugs: 15, resolved: 18 },
  { name: 'Thu', bugs: 25, resolved: 22 },
  { name: 'Fri', bugs: 32, resolved: 28 },
  { name: 'Sat', bugs: 20, resolved: 24 },
  { name: 'Sun', bugs: 15, resolved: 19 },
];

export function AnalyticsView() {
  return (
    <div className="space-y-6">
      <Tabs defaultValue="trends" className="w-full">
        <TabsList className="bg-white/5 border border-white/10 mb-6">
          <TabsTrigger value="trends" className="data-[state=active]:bg-white/10 text-slate-400 data-[state=active]:text-white">Trends</TabsTrigger>
          <TabsTrigger value="resolution" className="data-[state=active]:bg-white/10 text-slate-400 data-[state=active]:text-white">Resolution Time</TabsTrigger>
        </TabsList>

        <TabsContent value="trends" className="mt-0">
          <Card className="p-6 bg-white/5 border-white/10 backdrop-blur-md">
            <h3 className="text-lg font-medium text-white mb-6">Bug Inflow vs Resolution</h3>
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data}>
                  <defs>
                    <linearGradient id="colorBugs" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorResolved" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" tick={{fill: '#64748b'}} axisLine={false} tickLine={false} />
                  <YAxis stroke="#64748b" tick={{fill: '#64748b'}} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '12px' }}
                    itemStyle={{ color: '#e2e8f0' }}
                  />
                  <Area type="monotone" dataKey="bugs" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorBugs)" />
                  <Area type="monotone" dataKey="resolved" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorResolved)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}