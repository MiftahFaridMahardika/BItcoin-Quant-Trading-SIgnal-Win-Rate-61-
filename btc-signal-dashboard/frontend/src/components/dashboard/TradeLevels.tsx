'use client'

import { useSignalStore } from '@/stores/signalStore'
import { Card } from '@/components/ui/Card'
import { formatPrice } from '@/lib/utils'
import { Target, Shield, TrendingUp } from 'lucide-react'

interface LevelRowProps {
  label: string
  price?: number | null
  color: string
  icon: React.ReactNode
}

function LevelRow({ label, price, color, icon }: LevelRowProps) {
  if (!price) return null
  return (
    <div className="flex items-center gap-3 py-2.5 border-b border-white/5 last:border-0">
      <div className={`w-8 h-8 rounded-lg bg-${color}-500/10 flex items-center justify-center`}>
        {icon}
      </div>
      <span className="text-sm text-gray-400 flex-1">{label}</span>
      <span className={`font-mono text-sm font-semibold text-${color}-400`}>
        {formatPrice(price)}
      </span>
    </div>
  )
}

export function TradeLevels() {
  const { currentSignal } = useSignalStore()

  if (!currentSignal || currentSignal.signal === 'SKIP' || !currentSignal.entry) {
    return (
      <Card>
        <p className="text-xs text-gray-500 mb-3">TRADE LEVELS</p>
        <p className="text-gray-600 text-sm text-center py-4">No active trade signal</p>
      </Card>
    )
  }

  const { entry, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward } = currentSignal

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs text-gray-500">TRADE LEVELS</p>
        {risk_reward && (
          <span className="text-xs text-blue-400 font-semibold">R/R {risk_reward.toFixed(2)}x</span>
        )}
      </div>

      <LevelRow
        label="Entry"
        price={entry}
        color="blue"
        icon={<Target className="w-4 h-4 text-blue-400" />}
      />
      <LevelRow
        label="Stop Loss"
        price={stop_loss}
        color="red"
        icon={<Shield className="w-4 h-4 text-red-400" />}
      />
      <LevelRow
        label="TP1 (33%)"
        price={take_profit_1}
        color="green"
        icon={<TrendingUp className="w-4 h-4 text-green-400" />}
      />
      <LevelRow
        label="TP2 (33%)"
        price={take_profit_2}
        color="emerald"
        icon={<TrendingUp className="w-4 h-4 text-emerald-400" />}
      />
      <LevelRow
        label="TP3 (34%)"
        price={take_profit_3}
        color="teal"
        icon={<TrendingUp className="w-4 h-4 text-teal-400" />}
      />
    </Card>
  )
}
