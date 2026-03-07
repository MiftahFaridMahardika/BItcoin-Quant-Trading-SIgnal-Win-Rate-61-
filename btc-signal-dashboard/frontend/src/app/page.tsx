'use client'

import { MainLayout } from '@/components/layout/MainLayout'
import { SignalPanel } from '@/components/dashboard/SignalPanel'
import { RiskMetrics } from '@/components/dashboard/RiskMetrics'
import { SignalHistory } from '@/components/dashboard/SignalHistory'
import { MarketRegime } from '@/components/dashboard/MarketRegime'
import { LiveTicker } from '@/components/dashboard/LiveTicker'
import { PriceChart } from '@/components/dashboard/PriceChart'
import { MetricCard } from '@/components/ui/MetricCard'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { useSignalStore } from '@/stores/signalStore'
import { BarChart2 } from 'lucide-react'

export default function DashboardPage() {
  const {
    marketInfo,
    currentSignal,
    signalHistory,
    currentRegime,
    connected
  } = useSignalStore()

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Top Metrics Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            title="BTC Price"
            value={marketInfo?.price || 0}
            prefix="$"
            change={marketInfo?.change_24h}
            trend={marketInfo?.change_24h && marketInfo.change_24h >= 0 ? 'up' : 'down'}
          />
          <MetricCard
            title="24H Volume"
            value={marketInfo?.volume_24h ? `${(marketInfo.volume_24h / 1000).toFixed(1)}K` : '—'}
            suffix=" BTC"
          />
          <MetricCard
            title="Signal Score"
            value={currentSignal?.score ?? '—'}
            prefix={currentSignal?.score && currentSignal.score > 0 ? '+' : ''}
            trend={currentSignal?.score && currentSignal.score > 0 ? 'up' :
              currentSignal?.score && currentSignal.score < 0 ? 'down' : 'neutral'}
          />
          <MetricCard
            title="Confidence"
            value={currentSignal?.confidence ? (currentSignal.confidence * 100).toFixed(0) : '—'}
            suffix="%"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Chart */}
          <div className="lg:col-span-2 space-y-6">
            {/* Price Chart */}
            <Card className="h-[400px]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart2 className="w-4 h-4" />
                  BTC/USDT - 4H
                </CardTitle>
                <LiveTicker
                  price={marketInfo?.price}
                  change={marketInfo?.change_24h}
                  connected={connected}
                />
              </CardHeader>
              <div className="h-[320px]">
                <PriceChart
                  currentPrice={marketInfo?.price}
                  signal={currentSignal}
                />
              </div>
            </Card>

            {/* Risk Metrics */}
            <RiskMetrics />

            {/* Signal History Table */}
            <SignalHistory />
          </div>

          {/* Right Column: Signal Panel & Regime */}
          <div className="space-y-6">
            {/* Current Signal Panel */}
            <SignalPanel />

            {/* Market Regime */}
            <MarketRegime regime={currentRegime} />

            {/* Quick Stats */}
            <Card>
              <CardHeader>
                <CardTitle>Today&apos;s Stats</CardTitle>
              </CardHeader>
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b border-border/50">
                  <span className="text-sm text-text-secondary">Signals Generated</span>
                  <span className="font-mono font-bold">{signalHistory.length}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-border/50">
                  <span className="text-sm text-text-secondary">Actionable Signals</span>
                  <span className="font-mono font-bold">
                    {signalHistory.filter(s => s.signal !== 'SKIP').length}
                  </span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-sm text-text-secondary">Connection</span>
                  <span className={connected ? 'text-status-success' : 'text-status-danger'}>
                    {connected ? '● Connected' : '○ Disconnected'}
                  </span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}
