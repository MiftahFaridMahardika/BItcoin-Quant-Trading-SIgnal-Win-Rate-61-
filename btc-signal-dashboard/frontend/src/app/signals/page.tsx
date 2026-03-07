'use client'

import { MainLayout } from '@/components/layout/MainLayout'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { useSignalStore } from '@/stores/signalStore'
import { cn, formatPrice, formatPercent, getSignalColor, timeAgo } from '@/lib/utils'
import { TrendingUp, TrendingDown, Minus, Target, Shield, Activity } from 'lucide-react'

export default function SignalsPage() {
  const { currentSignal, signalHistory, currentRegime } = useSignalStore()

  const isLong = currentSignal?.direction === 'LONG'
  const isShort = currentSignal?.direction === 'SHORT'

  return (
    <MainLayout>
      <div className="space-y-6">
        <div>
          <h2 className="text-xl font-bold">Signal Analysis</h2>
          <p className="text-text-secondary text-sm mt-1">Current trading signal with full breakdown</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Current Signal Detail */}
          <Card>
            <CardHeader>
              <CardTitle>Current Signal</CardTitle>
            </CardHeader>
            {currentSignal ? (
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  {isLong ? <TrendingUp className="w-10 h-10 text-status-success" />
                    : isShort ? <TrendingDown className="w-10 h-10 text-status-danger" />
                    : <Minus className="w-10 h-10 text-text-muted" />}
                  <div>
                    <p className={cn('text-4xl font-bold', getSignalColor(currentSignal.signal))}>
                      {currentSignal.signal.replace('_', ' ')}
                    </p>
                    <p className="text-text-secondary text-sm">{timeAgo(currentSignal.timestamp)}</p>
                  </div>
                </div>

                {/* Score bar */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-text-secondary">Score</span>
                    <span className={currentSignal.score > 0 ? 'text-status-success' : currentSignal.score < 0 ? 'text-status-danger' : 'text-text-muted'}>
                      {currentSignal.score > 0 ? '+' : ''}{currentSignal.score} / {currentSignal.max_score}
                    </span>
                  </div>
                  <div className="h-2 bg-surface-elevated rounded-full overflow-hidden">
                    <div
                      className={cn('h-full rounded-full transition-all duration-500',
                        isLong ? 'bg-status-success' : isShort ? 'bg-status-danger' : 'bg-text-muted')}
                      style={{ width: `${Math.abs(currentSignal.score / currentSignal.max_score) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Confidence */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 bg-surface-elevated rounded-lg text-center">
                    <p className="text-xs text-text-muted mb-1">Confidence</p>
                    <p className="text-lg font-bold">{(currentSignal.confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div className="p-3 bg-surface-elevated rounded-lg text-center">
                    <p className="text-xs text-text-muted mb-1">Timeframe</p>
                    <p className="text-lg font-bold">{currentSignal.timeframe}</p>
                  </div>
                  <div className="p-3 bg-surface-elevated rounded-lg text-center">
                    <p className="text-xs text-text-muted mb-1">ATR</p>
                    <p className="text-lg font-bold">{currentSignal.atr.toFixed(0)}</p>
                  </div>
                </div>

                {/* Trade levels */}
                {currentSignal.entry && (
                  <div className="space-y-2 pt-2 border-t border-border">
                    <p className="text-xs text-text-muted uppercase tracking-wide">Trade Levels</p>
                    {[
                      { label: 'Entry', value: currentSignal.entry, icon: Target, color: 'text-accent-blue' },
                      { label: 'Stop Loss', value: currentSignal.stop_loss, icon: Shield, color: 'text-status-danger' },
                      { label: 'Take Profit 1', value: currentSignal.take_profit_1, icon: Target, color: 'text-status-success' },
                      { label: 'Take Profit 2', value: currentSignal.take_profit_2, icon: Target, color: 'text-status-success' },
                      { label: 'Take Profit 3', value: currentSignal.take_profit_3, icon: Target, color: 'text-status-success' },
                    ].map(({ label, value, icon: Icon, color }) => value ? (
                      <div key={label} className="flex items-center justify-between py-1">
                        <div className="flex items-center gap-2">
                          <Icon className={cn('w-4 h-4', color)} />
                          <span className="text-sm text-text-secondary">{label}</span>
                        </div>
                        <span className={cn('font-mono font-semibold', color)}>{formatPrice(value)}</span>
                      </div>
                    ) : null)}
                    {currentSignal.risk_reward && (
                      <div className="flex justify-between pt-2 border-t border-border/50">
                        <span className="text-sm text-text-secondary">Risk/Reward</span>
                        <span className="font-mono font-bold text-accent-blue">{currentSignal.risk_reward.toFixed(2)}x</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-32 text-text-muted">
                <Activity className="w-5 h-5 mr-2" /> Waiting for signal...
              </div>
            )}
          </Card>

          {/* Signal Reasons */}
          <Card>
            <CardHeader>
              <CardTitle>Signal Reasons</CardTitle>
            </CardHeader>
            {currentSignal?.reasons && currentSignal.reasons.length > 0 ? (
              <div className="space-y-2">
                {currentSignal.reasons.map((reason, i) => {
                  const isPositive = reason.startsWith('+')
                  const isNegative = reason.startsWith('-')
                  return (
                    <div key={i} className={cn(
                      'flex items-start gap-3 p-3 rounded-lg',
                      isPositive ? 'bg-status-success/5 border border-status-success/10' :
                      isNegative ? 'bg-status-danger/5 border border-status-danger/10' :
                      'bg-surface-elevated border border-border'
                    )}>
                      <span className={cn('text-sm font-bold w-6 flex-shrink-0',
                        isPositive ? 'text-status-success' : isNegative ? 'text-status-danger' : 'text-text-muted'
                      )}>
                        {isPositive ? '▲' : isNegative ? '▼' : '–'}
                      </span>
                      <span className="text-sm">{reason}</span>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-text-muted text-sm">No reasons available</p>
            )}
          </Card>
        </div>

        {/* Indicators */}
        {currentSignal?.indicators && (
          <Card>
            <CardHeader>
              <CardTitle>Technical Indicators</CardTitle>
            </CardHeader>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Object.entries(currentSignal.indicators).map(([key, value]) => (
                <div key={key} className="p-3 bg-surface-elevated rounded-lg">
                  <p className="text-xs text-text-muted mb-1 uppercase tracking-wide">{key.replace(/_/g, ' ')}</p>
                  <p className="font-mono font-semibold text-sm">
                    {typeof value === 'number' ? (
                      Math.abs(value) > 1000 ? formatPrice(value) :
                      Math.abs(value) < 10 ? value.toFixed(3) : value.toFixed(2)
                    ) : String(value)}
                  </p>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </MainLayout>
  )
}
