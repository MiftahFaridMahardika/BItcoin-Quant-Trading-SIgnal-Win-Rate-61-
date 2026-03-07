'use client'

import { useSignalStore } from '@/stores/signalStore'
import { usePriceStore } from '@/stores/priceStore'
import { Card } from '@/components/ui/Card'
import { formatPct } from '@/lib/utils'

interface MetricProps {
  label: string
  value: string
  sub?: string
  color?: string
}

function Metric({ label, value, sub, color = 'text-white' }: MetricProps) {
  return (
    <div className="bg-white/5 rounded-lg p-3">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-lg font-semibold font-mono ${color}`}>{value}</p>
      {sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>}
    </div>
  )
}

export function RiskMetrics() {
  const { currentSignal } = useSignalStore()
  const { currentPrice } = usePriceStore()

  const slPct = currentSignal?.stop_loss && currentPrice > 0
    ? Math.abs((currentSignal.stop_loss - currentPrice) / currentPrice)
    : null

  const tp1Pct = currentSignal?.take_profit_1 && currentPrice > 0
    ? Math.abs((currentSignal.take_profit_1 - currentPrice) / currentPrice)
    : null

  return (
    <Card>
      <p className="text-xs text-gray-500 mb-3">RISK METRICS</p>
      <div className="grid grid-cols-2 gap-2">
        <Metric
          label="Confidence"
          value={currentSignal ? formatPct(currentSignal.confidence) : '—'}
          color={
            !currentSignal ? 'text-gray-600'
            : currentSignal.confidence >= 0.7 ? 'text-green-400'
            : currentSignal.confidence >= 0.5 ? 'text-yellow-400'
            : 'text-red-400'
          }
        />
        <Metric
          label="Risk/Reward"
          value={currentSignal?.risk_reward ? `${currentSignal.risk_reward.toFixed(2)}x` : '—'}
          color="text-blue-400"
        />
        <Metric
          label="Stop Loss %"
          value={slPct ? `-${formatPct(slPct)}` : '—'}
          color="text-red-400"
        />
        <Metric
          label="TP1 Target %"
          value={tp1Pct ? `+${formatPct(tp1Pct)}` : '—'}
          color="text-green-400"
        />
      </div>
    </Card>
  )
}
