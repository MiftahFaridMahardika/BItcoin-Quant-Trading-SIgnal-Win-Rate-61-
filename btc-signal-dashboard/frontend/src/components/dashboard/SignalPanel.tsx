'use client'

import { useSignalStore } from '@/stores/signalStore'
import { Card } from '@/components/ui/Card'
import { SignalBadge } from '@/components/ui/Badge'
import { formatPrice, formatPct, getSignalColor } from '@/lib/utils'
import { TrendingUp, TrendingDown, Minus, Target, Shield } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

export function SignalPanel() {
  const { currentSignal } = useSignalStore()

  if (!currentSignal) {
    return (
      <Card className="flex items-center justify-center h-48">
        <p className="text-gray-600 text-sm">Waiting for signal...</p>
      </Card>
    )
  }

  const isLong = currentSignal.direction === 'LONG'
  const isShort = currentSignal.direction === 'SHORT'
  const isSkip = currentSignal.signal === 'SKIP'
  const glow: 'green' | 'red' | null = isLong ? 'green' : isShort ? 'red' : null

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={currentSignal.timestamp}
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Card glow={glow}>
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-xs text-gray-500 mb-1">CURRENT SIGNAL</p>
              <div className="flex items-center gap-3">
                {isLong ? (
                  <TrendingUp className="w-6 h-6 text-green-400" />
                ) : isShort ? (
                  <TrendingDown className="w-6 h-6 text-red-400" />
                ) : (
                  <Minus className="w-6 h-6 text-gray-400" />
                )}
                <span className={`text-3xl font-bold ${getSignalColor(currentSignal.signal)}`}>
                  {currentSignal.signal.replace('_', ' ')}
                </span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-500 mb-1">CONFIDENCE</p>
              <p className="text-2xl font-bold text-white">
                {formatPct(currentSignal.confidence * 100)}
              </p>
            </div>
          </div>

          {/* Score bar */}
          <div className="mb-4">
            <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
              <span>Score</span>
              <span className={currentSignal.score > 0 ? 'text-green-400' : currentSignal.score < 0 ? 'text-red-400' : 'text-gray-400'}>
                {currentSignal.score > 0 ? '+' : ''}{currentSignal.score} / {currentSignal.max_score}
              </span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${
                  isLong ? 'bg-green-400' : isShort ? 'bg-red-400' : 'bg-gray-600'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${currentSignal.confidence * 100}%` }}
                transition={{ duration: 0.6, ease: 'easeOut' }}
              />
            </div>
          </div>

          {/* Signal reasons */}
          {currentSignal.reasons && currentSignal.reasons.length > 0 && (
            <div className="mb-4 space-y-1">
              {currentSignal.reasons.slice(0, 4).map((reason, i) => (
                <p key={i} className="text-xs text-gray-500">• {reason}</p>
              ))}
            </div>
          )}

          {/* Trade levels */}
          {!isSkip && currentSignal.entry && (
            <div className="grid grid-cols-2 gap-3 text-sm border-t border-white/5 pt-4">
              <div className="flex items-center gap-2">
                <Target className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-gray-500">Entry</span>
                <span className="ml-auto font-mono text-white">{formatPrice(currentSignal.entry)}</span>
              </div>
              {currentSignal.stop_loss && (
                <div className="flex items-center gap-2">
                  <Shield className="w-3.5 h-3.5 text-red-400" />
                  <span className="text-gray-500">Stop</span>
                  <span className="ml-auto font-mono text-red-400">{formatPrice(currentSignal.stop_loss)}</span>
                </div>
              )}
              {currentSignal.take_profit_1 && (
                <div className="flex items-center gap-2">
                  <Target className="w-3.5 h-3.5 text-green-400" />
                  <span className="text-gray-500">TP1</span>
                  <span className="ml-auto font-mono text-green-400">{formatPrice(currentSignal.take_profit_1)}</span>
                </div>
              )}
              {currentSignal.risk_reward && (
                <div className="flex items-center gap-2">
                  <span className="text-gray-500">R/R</span>
                  <span className="ml-auto font-mono text-blue-400">{currentSignal.risk_reward.toFixed(2)}x</span>
                </div>
              )}
            </div>
          )}
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
