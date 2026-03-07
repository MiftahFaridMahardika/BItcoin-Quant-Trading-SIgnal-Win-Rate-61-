import { cn } from '@/lib/utils'
import { Card } from './Card'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
    title: string
    value: string | number
    change?: number
    changeLabel?: string
    prefix?: string
    suffix?: string
    trend?: 'up' | 'down' | 'neutral'
    size?: 'sm' | 'md' | 'lg'
    className?: string
}

export function MetricCard({
    title,
    value,
    change,
    changeLabel,
    prefix = '',
    suffix = '',
    trend,
    size = 'md',
    className
}: MetricCardProps) {
    const sizes = {
        sm: { title: 'text-xxs', value: 'text-lg' },
        md: { title: 'text-xs', value: 'text-2xl' },
        lg: { title: 'text-sm', value: 'text-3xl' },
    }

    const trendColors = {
        up: 'text-status-success',
        down: 'text-status-danger',
        neutral: 'text-text-secondary',
    }

    const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus

    return (
        <Card className={cn('', className)}>
            <p className={cn('text-text-secondary uppercase tracking-wide mb-1', sizes[size].title)}>
                {title}
            </p>

            <div className="flex items-baseline gap-2">
                <span className={cn('font-bold tabular-nums', sizes[size].value)}>
                    {prefix}{typeof value === 'number' ? value.toLocaleString() : value}{suffix}
                </span>

                {(change !== undefined || trend) && (
                    <div className={cn('flex items-center gap-1 text-sm', trendColors[trend || 'neutral'])}>
                        {trend && <TrendIcon className="w-4 h-4" />}
                        {change !== undefined && (
                            <span className="tabular-nums">
                                {change > 0 ? '+' : ''}{change.toFixed(2)}%
                            </span>
                        )}
                    </div>
                )}
            </div>

            {changeLabel && (
                <p className="text-xxs text-text-muted mt-1">{changeLabel}</p>
            )}
        </Card>
    )
}
