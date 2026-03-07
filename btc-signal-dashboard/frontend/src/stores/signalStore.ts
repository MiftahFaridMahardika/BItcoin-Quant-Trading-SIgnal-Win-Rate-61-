import { create } from 'zustand'

interface MarketInfo {
  symbol: string
  price: number
  change_24h: number
  volume_24h: number
  high_24h: number
  low_24h: number
}

export interface Signal {
  id: string
  timestamp: string
  candle_time: number
  timeframe: string
  price: number
  atr: number
  signal: string
  direction?: string
  score: number
  max_score: number
  confidence: number
  regime: string
  regime_confidence: number
  vol_regime?: string
  entry?: number
  stop_loss?: number
  take_profit_1?: number
  take_profit_2?: number
  take_profit_3?: number
  risk_reward?: number
  position_size_btc?: number
  reasons?: string[]
  indicators?: Record<string, number>
}

interface SignalStore {
  // Connection state
  connected: boolean
  setConnected: (connected: boolean) => void

  // Market data
  marketInfo: MarketInfo | null
  setMarketInfo: (info: MarketInfo) => void
  updatePrice: (data: { price: number; change_24h: number }) => void

  // Current signal
  currentSignal: Signal | null
  setCurrentSignal: (signal: Signal | null) => void

  // Signal history
  signalHistory: Signal[]
  addSignal: (signal: Signal) => void
  setSignalHistory: (signals: Signal[]) => void

  // Regime
  currentRegime: string
  setCurrentRegime: (regime: string) => void

  // Actions
  handleWebSocketMessage: (message: any) => void
  reset: () => void
}

const initialState = {
  connected: false,
  marketInfo: null,
  currentSignal: null,
  signalHistory: [],
  currentRegime: 'UNKNOWN',
}

export const useSignalStore = create<SignalStore>((set, get) => ({
  ...initialState,

  setConnected: (connected) => set({ connected }),

  setMarketInfo: (info) => set({ marketInfo: info }),

  updatePrice: (data) => set((state) => ({
    marketInfo: state.marketInfo
      ? { ...state.marketInfo, price: data.price, change_24h: data.change_24h }
      : null
  })),

  setCurrentSignal: (signal) => set({ currentSignal: signal }),

  addSignal: (signal) => set((state) => ({
    signalHistory: [signal, ...state.signalHistory].slice(0, 100),
    currentSignal: signal,
  })),

  setSignalHistory: (signals) => set({ signalHistory: signals }),

  setCurrentRegime: (regime) => set({ currentRegime: regime }),

  handleWebSocketMessage: (message) => {
    const { type, data } = message

    switch (type) {
      case 'initial_state':
        if (data.market) {
          get().setMarketInfo(data.market)
          // market summary includes current_regime from backend
          if (data.market.regime) get().setCurrentRegime(data.market.regime)
        }
        if (data.signal) {
          get().setCurrentSignal(data.signal)
          // also sync regime from last signal if market didn't have it
          if (data.signal.regime && data.signal.regime !== 'UNKNOWN') {
            get().setCurrentRegime(data.signal.regime)
          }
        }
        break

      case 'price_update':
        get().updatePrice(data)
        break

      case 'new_signal':
        get().addSignal(data)
        break

      case 'regime_change':
        get().setCurrentRegime(data.new)
        break

      case 'connection_status':
        get().setConnected(data.status === 'connected')
        break

      case 'heartbeat':
        // Keep alive
        break

      default:
        console.log('Unknown message type:', type)
    }
  },

  reset: () => set(initialState),
}))
