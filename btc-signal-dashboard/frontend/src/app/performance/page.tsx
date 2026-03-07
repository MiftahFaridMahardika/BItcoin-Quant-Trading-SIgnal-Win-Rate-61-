'use client'

import { MainLayout } from '@/components/layout/MainLayout'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { useSignalStore } from '@/stores/signalStore'
import { cn, formatPrice } from '@/lib/utils'
import { TrendingUp, TrendingDown, BarChart2, Activity } from 'lucide-react'

export default function PerformancePage() {
  const { signalHistory, currentSignal } = useSignalStore()

  const actionable = signalHistory.filter(s => s.signal !== 'SKIP')
  const longSignals = signalHistory.filter(s => s.signal === 'LONG' || s.signal === 'STRONG_LONG')
  const shortSignals = signalHistory.filter(s => s.signal === 'SHORT' || s.signal === 'STRONG_SHORT')
  const avgConfidence = actionable.length > 0
    ? actionable.reduce((sum, s) => sum + s.confidence, 0) / actionable.length
    : 0
  const avgScore = actionable.length > 0
    ? actionable.reduce((sum, s) => sum + s.score, 0) / actionable.length
    : 0

  return (
    <MainLayout>
      <div className="space-y-6">
        <div>
          <h2 className="text-xl font-bold">Performance</h2>
          <p className="text-text-secondary text-sm mt-1">Signal quality metrics for this session</p>
        </div>

        {/* Session stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Total Signals', value: signalHistory.length, sub: 'this session', icon: Activity, color: 'text-accent-blue' },
            { label: 'Actionable', value: actionable.length, sub: `${signalHistory.length > 0 ? ((actionable.length / signalHistory.length) * 100).toFixed(0) : 0}% of total`, icon: BarChart2, color: 'text-status-success' },
            { label: 'Long Bias', value: longSignals.length, sub: 'long signals', icon: TrendingUp, color: 'text-status-success' },
            { label: 'Short Bias', value: shortSignals.length, sub: 'short signals', icon: TrendingDown, color: 'text-status-danger' },
          ].map(({ label, value, sub, icon: Icon, color }) => (
            <Card key={label}>
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs text-text-muted mb-1 uppercase tracking-wide">{label}</p>
                  <p className={cn('text-3xl font-bold', color)}>{value}</p>
                  <p className="text-xs text-text-muted mt-1">{sub}</p>
                </div>
                <Icon className={cn('w-8 h-8 opacity-20', color)} />
              </div>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Signal quality */}
          <Card>
            <CardHeader>
              <CardTitle>Signal Quality</CardTitle>
            </CardHeader>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-text-secondary">Avg Confidence</span>
                  <span className="font-mono font-bold">{(avgConfidence * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-surface-elevated rounded-full">
                  <div className="h-full bg-accent-blue rounded-full transition-all" style={{ width: `${avgConfidence * 100}%` }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-text-secondary">Avg Score</span>
                  <span className={cn('font-mono font-bold', avgScore > 0 ? 'text-status-success' : 'text-status-danger')}>
                    {avgScore > 0 ? '+' : ''}{avgScore.toFixed(1)}
                  </span>
                </div>
                <div className="h-2 bg-surface-elevated rounded-full">
                  <div className={cn('h-full rounded-full transition-all', avgScore > 0 ? 'bg-status-success' : 'bg-status-danger')}
                    style={{ width: `${Math.min(Math.abs(avgScore / (currentSignal?.max_score || 19)) * 100, 100)}%` }} />
                </div>
              </div>

              <div className="pt-2 border-t border-border space-y-2">
                {[
                  { label: 'SKIP signals', value: signalHistory.filter(s => s.signal === 'SKIP').length },
                  { label: 'Strong signals', value: signalHistory.filter(s => s.signal === 'STRONG_LONG' || s.signal === 'STRONG_SHORT').length },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between">
                    <span className="text-sm text-text-secondary">{label}</span>
                    <span className="font-mono font-bold">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          {/* Current signal context */}
          <Card>
            <CardHeader>
              <CardTitle>Current Market Context</CardTitle>
            </CardHeader>
            {currentSignal ? (
              <div className="space-y-3">
                {[
                  { label: 'Price', value: formatPrice(currentSignal.price) },
                  { label: 'ATR', value: `$${currentSignal.atr.toFixed(0)}` },
                  { label: 'ATR %', value: currentSignal.indicators?.atr_pct ? `${(currentSignal.indicators.atr_pct).toFixed(2)}%` : '—' },
                  { label: 'RSI', value: currentSignal.indicators?.rsi ? currentSignal.indicators.rsi.toFixed(1) : '—' },
                  { label: 'MACD Hist', value: currentSignal.indicators?.macd_hist ? currentSignal.indicators.macd_hist.toFixed(1) : '—' },
                  { label: 'Z-Score', value: currentSignal.indicators?.zscore_20 ? currentSignal.indicators.zscore_20.toFixed(2) : '—' },
                  { label: 'Vol Regime', value: currentSignal.vol_regime ?? '—' },
                  { label: 'Regime', value: currentSignal.regime },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between py-1 border-b border-border/30">
                    <span className="text-sm text-text-secondary">{label}</span>
                    <span className="font-mono text-sm font-semibold">{value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-text-muted text-sm">Waiting for data...</p>
            )}
          </Card>
        </div>
      </div>
    </MainLayout>
  )
}
