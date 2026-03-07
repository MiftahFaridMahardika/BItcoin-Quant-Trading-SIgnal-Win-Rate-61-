'use client'

import { useSignalStore, type Signal } from '@/stores/signalStore'
import { Card } from '@/components/ui/Card'
import { SignalBadge } from '@/components/ui/Badge'
import { Table } from '@/components/ui/Table'
import { formatPrice, formatPct, timeAgo } from '@/lib/utils'

const columns = [
  {
    key: 'timestamp',
    header: 'Time',
    render: (row: Signal) => (
      <span className="text-gray-500 text-xs">{timeAgo(row.timestamp)}</span>
    ),
  },
  {
    key: 'signal',
    header: 'Signal',
    render: (row: Signal) => (
      <SignalBadge signal={row.signal} />
    ),
  },
  {
    key: 'price',
    header: 'Price',
    render: (row: Signal) => (
      <span className="font-mono text-xs">{formatPrice(row.price)}</span>
    ),
  },
  {
    key: 'confidence',
    header: 'Conf',
    render: (row: Signal) => (
      <span className={`text-xs font-semibold ${
        row.confidence >= 0.7 ? 'text-green-400'
        : row.confidence >= 0.5 ? 'text-yellow-400'
        : 'text-red-400'
      }`}>
        {formatPct(row.confidence * 100)}
      </span>
    ),
  },
  {
    key: 'score',
    header: 'Score',
    render: (row: Signal) => (
      <span className={`text-xs font-mono ${row.score > 0 ? 'text-green-400' : row.score < 0 ? 'text-red-400' : 'text-gray-400'}`}>
        {row.score > 0 ? '+' : ''}{row.score}
      </span>
    ),
  },
]

export function SignalHistory() {
  const { signalHistory } = useSignalStore()

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs text-gray-500">SIGNAL HISTORY</p>
        <span className="text-xs text-gray-600">{signalHistory.length} signals</span>
      </div>
      <Table
        columns={columns as any}
        data={signalHistory.slice(0, 20) as any[]}
        emptyText="No signals yet — waiting for data..."
      />
    </Card>
  )
}
