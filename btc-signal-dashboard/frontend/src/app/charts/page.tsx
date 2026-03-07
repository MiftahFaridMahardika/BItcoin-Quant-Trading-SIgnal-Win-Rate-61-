'use client'

import { useState } from 'react'
import { MainLayout } from '@/components/layout/MainLayout'
import { PriceChart } from '@/components/dashboard/PriceChart'
import { Card } from '@/components/ui/Card'
import { useSignalStore } from '@/stores/signalStore'
import { cn } from '@/lib/utils'

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'] as const
type Timeframe = typeof TIMEFRAMES[number]

export default function ChartsPage() {
  const [activeTimeframe, setActiveTimeframe] = useState<Timeframe>('4h')
  const { currentSignal, marketInfo } = useSignalStore()

  return (
    <MainLayout>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">Price Charts</h2>
            <p className="text-text-secondary text-sm mt-1">BTC/USDT multi-timeframe analysis</p>
          </div>

          {/* Timeframe selector */}
          <div className="flex gap-1 bg-surface-elevated border border-border rounded-lg p-1">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                onClick={() => setActiveTimeframe(tf)}
                className={cn(
                  'px-3 py-1.5 rounded-md text-sm font-medium transition-all',
                  activeTimeframe === tf
                    ? 'bg-accent-blue text-white'
                    : 'text-text-secondary hover:text-text-primary hover:bg-surface-hover'
                )}
              >
                {tf.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Full-height chart */}
        <Card className="h-[calc(100vh-220px)] min-h-[500px]">
          <PriceChart
            currentPrice={marketInfo?.price}
            signal={currentSignal}
            className="w-full h-full"
          />
        </Card>

        {/* Price summary */}
        {marketInfo && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: '24H High', value: `$${marketInfo.high_24h?.toLocaleString() ?? '—'}`, color: 'text-status-success' },
              { label: '24H Low', value: `$${marketInfo.low_24h?.toLocaleString() ?? '—'}`, color: 'text-status-danger' },
              { label: '24H Volume', value: `${marketInfo.volume_24h?.toFixed(0)} BTC`, color: 'text-text-primary' },
              { label: '24H Change', value: `${marketInfo.change_24h >= 0 ? '+' : ''}${marketInfo.change_24h?.toFixed(2)}%`, color: marketInfo.change_24h >= 0 ? 'text-status-success' : 'text-status-danger' },
            ].map(({ label, value, color }) => (
              <Card key={label} className="text-center">
                <p className="text-xs text-text-muted mb-1">{label}</p>
                <p className={cn('text-lg font-bold font-mono', color)}>{value}</p>
              </Card>
            ))}
          </div>
        )}
      </div>
    </MainLayout>
  )
}
