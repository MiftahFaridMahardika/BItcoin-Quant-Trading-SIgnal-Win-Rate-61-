'use client'

import { MainLayout } from '@/components/layout/MainLayout'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { useSignalStore } from '@/stores/signalStore'
import { cn, formatPrice, formatPercent, getSignalColor, timeAgo } from '@/lib/utils'
import { Activity } from 'lucide-react'

const SIGNAL_LABELS: Record<string, string> = {
  STRONG_LONG: 'Strong Long',
  LONG: 'Long',
  SKIP: 'Skip',
  SHORT: 'Short',
  STRONG_SHORT: 'Strong Short',
}

export default function HistoryPage() {
  const { signalHistory } = useSignalStore()

  const actionable = signalHistory.filter(s => s.signal !== 'SKIP')
  const longCount = signalHistory.filter(s => s.signal === 'LONG' || s.signal === 'STRONG_LONG').length
  const shortCount = signalHistory.filter(s => s.signal === 'SHORT' || s.signal === 'STRONG_SHORT').length

  return (
    <MainLayout>
      <div className="space-y-6">
        <div>
          <h2 className="text-xl font-bold">Signal History</h2>
          <p className="text-text-secondary text-sm mt-1">All generated signals from the current session</p>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Total Signals', value: signalHistory.length, color: 'text-text-primary' },
            { label: 'Actionable', value: actionable.length, color: 'text-accent-blue' },
            { label: 'Long Signals', value: longCount, color: 'text-status-success' },
            { label: 'Short Signals', value: shortCount, color: 'text-status-danger' },
          ].map(({ label, value, color }) => (
            <Card key={label} className="text-center">
              <p className="text-xs text-text-muted mb-1">{label}</p>
              <p className={cn('text-3xl font-bold', color)}>{value}</p>
            </Card>
          ))}
        </div>

        {/* Signal table */}
        <Card>
          <CardHeader>
            <CardTitle>All Signals</CardTitle>
          </CardHeader>
          {signalHistory.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-40 text-text-muted gap-2">
              <Activity className="w-8 h-8" />
              <p>No signals yet — waiting for 4H candle close...</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-text-muted text-xs uppercase">
                    <th className="text-left pb-3 pr-4">Time</th>
                    <th className="text-left pb-3 pr-4">Signal</th>
                    <th className="text-right pb-3 pr-4">Price</th>
                    <th className="text-right pb-3 pr-4">Score</th>
                    <th className="text-right pb-3 pr-4">Confidence</th>
                    <th className="text-right pb-3 pr-4">Entry</th>
                    <th className="text-right pb-3 pr-4">Stop Loss</th>
                    <th className="text-right pb-3">TP1</th>
                  </tr>
                </thead>
                <tbody>
                  {signalHistory.map((signal) => (
                    <tr key={signal.id} className="border-b border-border/30 hover:bg-surface-hover transition-colors">
                      <td className="py-3 pr-4 text-text-muted whitespace-nowrap">{timeAgo(signal.timestamp)}</td>
                      <td className="py-3 pr-4">
                        <span className={cn('font-bold', getSignalColor(signal.signal))}>
                          {SIGNAL_LABELS[signal.signal] ?? signal.signal}
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-right font-mono">{formatPrice(signal.price)}</td>
                      <td className={cn('py-3 pr-4 text-right font-mono font-bold',
                        signal.score > 0 ? 'text-status-success' : signal.score < 0 ? 'text-status-danger' : 'text-text-muted'
                      )}>
                        {signal.score > 0 ? '+' : ''}{signal.score}
                      </td>
                      <td className="py-3 pr-4 text-right font-mono">
                        {(signal.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 pr-4 text-right font-mono text-accent-blue">
                        {signal.entry ? formatPrice(signal.entry) : '—'}
                      </td>
                      <td className="py-3 pr-4 text-right font-mono text-status-danger">
                        {signal.stop_loss ? formatPrice(signal.stop_loss) : '—'}
                      </td>
                      <td className="py-3 text-right font-mono text-status-success">
                        {signal.take_profit_1 ? formatPrice(signal.take_profit_1) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </MainLayout>
  )
}
