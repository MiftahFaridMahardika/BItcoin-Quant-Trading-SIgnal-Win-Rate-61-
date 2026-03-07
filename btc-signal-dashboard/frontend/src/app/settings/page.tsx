'use client'

import { MainLayout } from '@/components/layout/MainLayout'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'

export default function SettingsPage() {
  const config = [
    { section: 'Signal Engine', items: [
      { label: 'Primary Timeframe', value: '4H' },
      { label: 'Signal Types', value: 'STRONG_LONG, LONG, SKIP, SHORT, STRONG_SHORT' },
      { label: 'Max Score', value: '19' },
      { label: 'Min Confidence for Trade', value: '> 0%' },
    ]},
    { section: 'Data Source', items: [
      { label: 'Exchange', value: 'Binance (WebSocket)' },
      { label: 'Symbol', value: 'BTCUSDT' },
      { label: 'Historical Bars', value: '500 candles' },
      { label: 'Bootstrap Limit', value: '501 (500 closed bars)' },
    ]},
    { section: 'Regime Detection', items: [
      { label: 'Regimes', value: 'BULL, BEAR, SIDEWAYS, HIGH_VOL, UNKNOWN' },
      { label: 'Vol Regimes', value: 'LOW, NORMAL, HIGH, EXTREME' },
    ]},
    { section: 'Connection', items: [
      { label: 'Backend URL', value: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000' },
      { label: 'WebSocket URL', value: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/stream' },
      { label: 'Reconnect Interval', value: '3s (exponential backoff)' },
      { label: 'Max Reconnect Attempts', value: '10' },
    ]},
  ]

  return (
    <MainLayout>
      <div className="space-y-6">
        <div>
          <h2 className="text-xl font-bold">Settings</h2>
          <p className="text-text-secondary text-sm mt-1">System configuration and parameters</p>
        </div>

        {config.map(({ section, items }) => (
          <Card key={section}>
            <CardHeader>
              <CardTitle>{section}</CardTitle>
            </CardHeader>
            <div className="space-y-1">
              {items.map(({ label, value }) => (
                <div key={label} className="flex justify-between items-center py-2.5 border-b border-border/30 last:border-0">
                  <span className="text-sm text-text-secondary">{label}</span>
                  <span className="font-mono text-sm text-right max-w-[60%] truncate">{value}</span>
                </div>
              ))}
            </div>
          </Card>
        ))}

        <Card>
          <CardHeader>
            <CardTitle>API Endpoints</CardTitle>
          </CardHeader>
          <div className="space-y-1">
            {[
              ['GET /api/market', 'Market info + current signal'],
              ['GET /api/signal/current', 'Latest signal'],
              ['GET /api/signals/history', 'Signal history (DB)'],
              ['GET /api/candles/{timeframe}', 'OHLCV candles'],
              ['GET /api/indicators', 'Latest indicator values'],
              ['WS /ws/stream', 'Real-time stream'],
            ].map(([endpoint, desc]) => (
              <div key={endpoint} className="flex items-center gap-4 py-2.5 border-b border-border/30 last:border-0">
                <code className="text-xs font-mono text-accent-blue bg-accent-blue/5 px-2 py-1 rounded flex-shrink-0">{endpoint}</code>
                <span className="text-sm text-text-secondary">{desc}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </MainLayout>
  )
}
