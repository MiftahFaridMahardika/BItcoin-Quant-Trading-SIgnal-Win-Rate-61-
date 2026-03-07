'use client'

import { useEffect, useRef, useCallback, useState } from 'react'

interface WebSocketOptions {
  url: string
  onMessage?: (data: any) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  reconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

interface WebSocketState {
  connected: boolean
  connecting: boolean
  error: string | null
}

export function useWebSocket({
  url,
  onMessage,
  onConnect,
  onDisconnect,
  onError,
  reconnect = true,
  reconnectInterval = 3000,
  maxReconnectAttempts = 10,
}: WebSocketOptions) {
  const ws = useRef<WebSocket | null>(null)
  const reconnectAttempts = useRef(0)
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null)

  const [state, setState] = useState<WebSocketState>({
    connected: false,
    connecting: false,
    error: null,
  })

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return

    setState(s => ({ ...s, connecting: true, error: null }))

    try {
      ws.current = new WebSocket(url)

      ws.current.onopen = () => {
        console.log('WebSocket connected')
        reconnectAttempts.current = 0
        setState({ connected: true, connecting: false, error: null })
        onConnect?.()
      }

      ws.current.onclose = () => {
        console.log('WebSocket disconnected')
        setState(s => ({ ...s, connected: false, connecting: false }))
        onDisconnect?.()

        // Attempt reconnect
        if (reconnect && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++
          console.log(`Reconnecting... (${reconnectAttempts.current}/${maxReconnectAttempts})`)

          reconnectTimeout.current = setTimeout(() => {
            connect()
          }, reconnectInterval * Math.min(reconnectAttempts.current, 5))
        }
      }

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setState(s => ({ ...s, error: 'Connection error' }))
        onError?.(error)
      }

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage?.(data)
        } catch (e) {
          console.error('Failed to parse message:', e)
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket:', error)
      setState(s => ({ ...s, connecting: false, error: 'Failed to connect' }))
    }
  }, [url, onMessage, onConnect, onDisconnect, onError, reconnect, reconnectInterval, maxReconnectAttempts])

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current)
    }

    if (ws.current) {
      ws.current.close()
      ws.current = null
    }

    setState({ connected: false, connecting: false, error: null })
  }, [])

  const send = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }, [])

  useEffect(() => {
    connect()

    // Ping interval to keep connection alive
    const pingInterval = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send('ping')
      }
    }, 25000)

    return () => {
      clearInterval(pingInterval)
      disconnect()
    }
  }, [connect, disconnect])

  return { ...state, send, connect, disconnect }
}
