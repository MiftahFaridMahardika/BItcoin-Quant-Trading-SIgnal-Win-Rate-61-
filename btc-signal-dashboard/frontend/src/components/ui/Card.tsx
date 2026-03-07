import { cn } from '@/lib/utils'
import { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  className?: string
  hover?: boolean
  glow?: 'blue' | 'green' | 'red' | null
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

export function Card({
  children,
  className,
  hover = false,
  glow = null,
  padding = 'md'
}: CardProps) {
  const paddingClasses = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  }

  return (
    <div
      className={cn(
        'bg-surface border border-border rounded-xl',
        'transition-all duration-200',
        paddingClasses[padding],
        hover && 'hover:border-border-light hover:shadow-glow-blue cursor-pointer',
        glow === 'blue' && 'shadow-glow-blue border-accent-blue/30',
        glow === 'green' && 'shadow-glow-green border-status-success/30',
        glow === 'red' && 'shadow-glow-red border-status-danger/30',
        className
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({
  children,
  className
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex items-center justify-between mb-4', className)}>
      {children}
    </div>
  )
}

export function CardTitle({
  children,
  className
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <h3 className={cn('text-sm font-semibold text-text-secondary uppercase tracking-wide', className)}>
      {children}
    </h3>
  )
}
