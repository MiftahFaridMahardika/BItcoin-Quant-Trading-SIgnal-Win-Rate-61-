'use client'

import { useEffect, useRef, useState } from 'react'
import { createChart, IChartApi, ISeriesApi, CandlestickData } from 'lightweight-charts'
import { cn } from '@/lib/utils'
import { fetchCandles } from '@/lib/api'

interface Signal {
  entry?: number
  stop_loss?: number
  take_profit_1?: number
  take_profit_2?: number
  take_profit_3?: number
  direction?: string
}

interface PriceChartProps {
  currentPrice?: number
  signal?: Signal | null
  className?: string
}

function generateDemoData(): CandlestickData[] {
  const data: CandlestickData[] = []
  let price = 65000
  const now = Date.now()

  for (let i = 200; i >= 0; i--) {
    const time = Math.floor((now - i * 4 * 60 * 60 * 1000) / 1000) as any
    const volatility = 500 + Math.random() * 1000
    const open = price
    const close = price + (Math.random() - 0.48) * volatility
    const high = Math.max(open, close) + Math.random() * volatility * 0.5
    const low = Math.min(open, close) - Math.random() * volatility * 0.5
    data.push({ time, open, high, low, close })
    price = close
  }

  return data
}

export function PriceChart({ currentPrice, signal, className }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const priceLineRefs = useRef<any[]>([])
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (!isClient || !containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: 'solid' as any, color: '#141417' },
        textColor: '#8B8B8E',
      },
      grid: {
        vertLines: { color: '#2A2A30' },
        horzLines: { color: '#2A2A30' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: '#3B82F6',
          width: 1,
          style: 2,
          labelBackgroundColor: '#3B82F6',
        },
        horzLine: {
          color: '#3B82F6',
          width: 1,
          style: 2,
          labelBackgroundColor: '#3B82F6',
        },
      },
      rightPriceScale: { borderColor: '#2A2A30' },
      timeScale: {
        borderColor: '#2A2A30',
        timeVisible: true,
        secondsVisible: false,
      },
      handleScale: { mouseWheel: true, pinch: true },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
    })

    chartRef.current = chart

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10B981',
      downColor: '#EF4444',
      borderUpColor: '#10B981',
      borderDownColor: '#EF4444',
      wickUpColor: '#10B981',
      wickDownColor: '#EF4444',
    })

    candleSeriesRef.current = candleSeries

    // Fetch real candles from API; timestamp is in ms, lightweight-charts needs seconds
    fetchCandles('4h', 200)
      .then((res) => {
        const candles: CandlestickData[] = (res.candles || []).map((c: any) => ({
          time: Math.floor(c.timestamp / 1000) as any,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        }))
        if (candles.length > 0) {
          candleSeries.setData(candles)
        } else {
          candleSeries.setData(generateDemoData())
        }
        chart.timeScale().fitContent()
      })
      .catch(() => {
        candleSeries.setData(generateDemoData())
        chart.timeScale().fitContent()
      })

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }

    window.addEventListener('resize', handleResize)
    handleResize()

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [isClient])

  // Update price lines when signal changes
  useEffect(() => {
    if (!candleSeriesRef.current || !signal) return

    // Remove old price lines
    priceLineRefs.current.forEach((line) => {
      try { candleSeriesRef.current?.removePriceLine(line) } catch {}
    })
    priceLineRefs.current = []

    const addLine = (
      price: number | undefined,
      color: string,
      title: string,
      lineWidth: 1 | 2,
      lineStyle: 0 | 2
    ) => {
      if (!price || !candleSeriesRef.current) return
      const line = candleSeriesRef.current.createPriceLine({
        price, color, lineWidth, lineStyle, axisLabelVisible: true, title,
      })
      priceLineRefs.current.push(line)
    }

    addLine(signal.entry, '#3B82F6', 'Entry', 2, 0)
    addLine(signal.stop_loss, '#EF4444', 'SL', 2, 2)
    addLine(signal.take_profit_1, '#10B981', 'TP1', 1, 2)
    addLine(signal.take_profit_2, '#10B981', 'TP2', 1, 2)
    addLine(signal.take_profit_3, '#10B981', 'TP3', 1, 2)
  }, [signal])

  if (!isClient) {
    return (
      <div className={cn('w-full h-full bg-surface flex items-center justify-center', className)}>
        <span className="text-text-muted">Loading chart...</span>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={cn('w-full h-full', className)}
    />
  )
}
