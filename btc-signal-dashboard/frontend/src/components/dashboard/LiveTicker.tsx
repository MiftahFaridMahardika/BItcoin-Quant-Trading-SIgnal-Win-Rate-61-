'use client'

import { cn, formatPrice, formatPercent } from '@/lib/utils'
import { Wifi, WifiOff } from 'lucide-react'

interface LiveTickerProps {
  price?: number
  change?: number
  connected?: boolean
}

export function LiveTicker({ price, change, connected }: LiveTickerProps) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <span className="font-mono font-bold tabular-nums">
          {price ? formatPrice(price) : '—'}
        </span>
        {change !== undefined && (
          <span className={cn(
            'text-sm font-medium tabular-nums',
            change >= 0 ? 'text-status-success' : 'text-status-danger'
          )}>
            {formatPercent(change)}
          </span>
        )}
      </div>

      <div className={cn(
        'flex items-center gap-1 text-xs',
        connected ? 'text-status-success' : 'text-status-danger'
      )}>
        {connected ? (
          <>
            <Wifi className="w-3 h-3" />
            <span>Live</span>
          </>
        ) : (
          <>
            <WifiOff className="w-3 h-3" />
            <span>Offline</span>
          </>
        )}
      </div>
    </div>
  )
}
