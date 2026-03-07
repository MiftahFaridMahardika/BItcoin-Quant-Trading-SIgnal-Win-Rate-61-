'use client'

import { ReactNode, useCallback } from 'react'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useSignalStore } from '@/stores/signalStore'

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/stream'

export function Providers({ children }: { children: ReactNode }) {
  const { handleWebSocketMessage, setConnected } = useSignalStore()

  const onConnect = useCallback(() => setConnected(true), [setConnected])
  const onDisconnect = useCallback(() => setConnected(false), [setConnected])

  useWebSocket({
    url: WS_URL,
    onMessage: handleWebSocketMessage,
    onConnect,
    onDisconnect,
  })

  return <>{children}</>
}
