import { create } from 'zustand'

export interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface PriceState {
  currentPrice: number
  priceChange24h: number
  candles: Candle[]
  lastUpdate: string | null
  setCurrentPrice: (price: number) => void
  setPriceChange: (change: number) => void
  setCandles: (candles: Candle[]) => void
  addCandle: (candle: Candle) => void
  setLastUpdate: (ts: string) => void
}

export const usePriceStore = create<PriceState>((set) => ({
  currentPrice: 0,
  priceChange24h: 0,
  candles: [],
  lastUpdate: null,

  setCurrentPrice: (price) => set({ currentPrice: price }),
  setPriceChange: (change) => set({ priceChange24h: change }),
  setCandles: (candles) => set({ candles }),
  addCandle: (candle) =>
    set((state) => {
      const candles = [...state.candles]
      const idx = candles.findIndex((c) => c.time === candle.time)
      if (idx >= 0) {
        candles[idx] = candle
      } else {
        candles.push(candle)
      }
      return { candles: candles.slice(-500) }
    }),
  setLastUpdate: (ts) => set({ lastUpdate: ts }),
}))
