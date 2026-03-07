'use client'

import { useEffect, useState } from 'react'
import { Badge } from '@/components/ui/Badge'
import { formatPrice, formatPercent } from '@/lib/utils'
import { Wifi, WifiOff, Bell, RefreshCw } from 'lucide-react'

interface HeaderProps {
  price?: number
  change24h?: number
  connected?: boolean
  lastUpdate?: string
}

export function Header({
  price = 0,
  change24h = 0,
  connected = false,
  lastUpdate
}: HeaderProps) {
  const [currentTime, setCurrentTime] = useState<string>('')

  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      }))
    }

    updateTime()
    const interval = setInterval(updateTime, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <header className="h-16 bg-surface border-b border-border px-6 flex items-center justify-between">
      {/* Left: Price ticker */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold tabular-nums">
              {formatPrice(price)}
            </span>
            <Badge
              variant={change24h >= 0 ? 'success' : 'danger'}
              size="sm"
            >
              {formatPercent(change24h)}
            </Badge>
          </div>
          <span className="text-text-muted text-sm">BTC/USDT</span>
        </div>

        {/* Separator */}
        <div className="w-px h-8 bg-border" />

        {/* Connection status */}
        <div className="flex items-center gap-2">
          {connected ? (
            <>
              <Wifi className="w-4 h-4 text-status-success" />
              <span className="text-sm text-status-success">Live</span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4 text-status-danger" />
              <span className="text-sm text-status-danger">Disconnected</span>
            </>
          )}
        </div>
      </div>

      {/* Right: Time and actions */}
      <div className="flex items-center gap-4">
        <div className="text-right">
          <div className="text-sm font-mono tabular-nums">{currentTime}</div>
          {lastUpdate && (
            <div className="text-xxs text-text-muted flex items-center gap-1">
              <RefreshCw className="w-3 h-3" />
              Last signal: {lastUpdate}
            </div>
          )}
        </div>

        {/* Notification button */}
        <button className="relative p-2 rounded-lg hover:bg-surface-hover transition-colors">
          <Bell className="w-5 h-5 text-text-secondary" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-status-danger rounded-full" />
        </button>
      </div>
    </header>
  )
}
