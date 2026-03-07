'use client'

import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { RegimeBadge } from '@/components/ui/Badge'
import { cn } from '@/lib/utils'
import { TrendingUp, TrendingDown, Activity, Zap } from 'lucide-react'

interface MarketRegimeProps {
  regime: string
  className?: string
}

const regimeInfo: Record<string, {
  icon: any
  description: string
  strategy: string
  color: string
}> = {
  'BULL': {
    icon: TrendingUp,
    description: 'Uptrend detected. Market showing bullish momentum.',
    strategy: 'Favor LONG positions. Wider TP targets.',
    color: 'text-regime-bull'
  },
  'BEAR': {
    icon: TrendingDown,
    description: 'Downtrend detected. Market showing bearish momentum.',
    strategy: 'Favor SHORT positions or stay flat.',
    color: 'text-regime-bear'
  },
  'SIDEWAYS': {
    icon: Activity,
    description: 'Range-bound market. No clear direction.',
    strategy: 'Tighter stops, reduced position size.',
    color: 'text-regime-sideways'
  },
  'HIGH_VOL': {
    icon: Zap,
    description: 'High volatility detected. Increased risk.',
    strategy: 'SKIP all signals. Wait for stability.',
    color: 'text-regime-high-vol'
  },
  'UNKNOWN': {
    icon: Activity,
    description: 'Analyzing market conditions...',
    strategy: 'Waiting for data.',
    color: 'text-text-muted'
  }
}

export function MarketRegime({ regime, className }: MarketRegimeProps) {
  const info = regimeInfo[regime] || regimeInfo['UNKNOWN']
  const Icon = info.icon

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Market Regime</CardTitle>
        <RegimeBadge regime={regime} />
      </CardHeader>

      <div className="space-y-4">
        <div className={cn('flex items-center gap-3', info.color)}>
          <Icon className="w-8 h-8" />
          <div>
            <p className="font-semibold">{regime.replace('_', ' ')}</p>
            <p className="text-xs text-text-secondary">{info.description}</p>
          </div>
        </div>

        <div className="p-3 bg-surface-elevated rounded-lg border border-border">
          <p className="text-xxs text-text-muted uppercase tracking-wide mb-1">Strategy</p>
          <p className="text-sm">{info.strategy}</p>
        </div>
      </div>
    </Card>
  )
}
