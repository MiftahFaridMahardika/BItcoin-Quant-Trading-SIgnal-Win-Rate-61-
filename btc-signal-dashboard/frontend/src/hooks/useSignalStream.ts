'use client'

import { useCallback } from 'react'
import { useWebSocket } from './useWebSocket'
import { useSignalStore } from '@/stores/signalStore'
import { usePriceStore } from '@/stores/priceStore'

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/stream'

export function useSignalStream() {
  const { setCurrentSignal, addSignal, setConnected } = useSignalStore()
  const { setCurrentPrice, setPriceChange } = usePriceStore()

  const handleMessage = useCallback(
    (data: unknown) => {
      if (!data || typeof data !== 'object') return
      const msg = data as Record<string, unknown>

      switch (msg.type) {
        case 'initial_state': {
          const d = (msg.data as any) || {}
          if (d.signal) setCurrentSignal(d.signal)
          if (d.market?.price) setCurrentPrice(d.market.price)
          if (d.market?.change_24h != null) setPriceChange(d.market.change_24h)
          setConnected(true)
          break
        }

        case 'new_signal': {
          const signal = (msg.data as any)
          if (signal?.signal && signal.signal !== 'SKIP') addSignal(signal)
          else setCurrentSignal(signal)
          if (signal?.price) setCurrentPrice(signal.price)
          break
        }

        case 'price_update': {
          const p = (msg.data as any)
          if (p?.price) setCurrentPrice(p.price)
          if (p?.change_24h != null) setPriceChange(p.change_24h)
          break
        }

        case 'connection_status':
          setConnected((msg.data as any)?.status === 'connected')
          break

        case 'heartbeat':
        case 'regime_change':
          break
      }
    },
    [setCurrentSignal, addSignal, setConnected, setCurrentPrice, setPriceChange]
  )

  const { connected, send } = useWebSocket({
    url: WS_URL,
    onMessage: handleMessage,
    onConnect: () => setConnected(true),
    onDisconnect: () => setConnected(false),
  })

  return { isConnected: connected, sendMessage: send }
}
