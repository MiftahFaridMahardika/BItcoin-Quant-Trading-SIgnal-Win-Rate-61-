import { cn } from '@/lib/utils'
import { ReactNode } from 'react'

type BadgeVariant = 'success' | 'danger' | 'warning' | 'info' | 'default' |
  'strong-long' | 'long' | 'skip' | 'short' | 'strong-short' |
  'bull' | 'bear' | 'sideways' | 'high-vol'

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
  size?: 'sm' | 'md' | 'lg'
  pulse?: boolean
  className?: string
}

export function Badge({
  children,
  variant = 'default',
  size = 'md',
  pulse = false,
  className
}: BadgeProps) {
  const variants: Record<BadgeVariant, string> = {
    'success': 'bg-status-success/10 text-status-success border-status-success/20',
    'danger': 'bg-status-danger/10 text-status-danger border-status-danger/20',
    'warning': 'bg-status-warning/10 text-status-warning border-status-warning/20',
    'info': 'bg-status-info/10 text-status-info border-status-info/20',
    'default': 'bg-surface-elevated text-text-secondary border-border',
    // Signal variants
    'strong-long': 'bg-signal-strong-long/10 text-signal-strong-long border-signal-strong-long/20',
    'long': 'bg-signal-long/10 text-signal-long border-signal-long/20',
    'skip': 'bg-signal-skip/10 text-signal-skip border-signal-skip/20',
    'short': 'bg-signal-short/10 text-signal-short border-signal-short/20',
    'strong-short': 'bg-signal-strong-short/10 text-signal-strong-short border-signal-strong-short/20',
    // Regime variants
    'bull': 'bg-regime-bull/10 text-regime-bull border-regime-bull/20',
    'bear': 'bg-regime-bear/10 text-regime-bear border-regime-bear/20',
    'sideways': 'bg-regime-sideways/10 text-regime-sideways border-regime-sideways/20',
    'high-vol': 'bg-regime-high-vol/10 text-regime-high-vol border-regime-high-vol/20',
  }

  const sizes = {
    sm: 'px-2 py-0.5 text-xxs',
    md: 'px-2.5 py-0.5 text-xs',
    lg: 'px-3 py-1 text-sm',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full font-medium border',
        variants[variant],
        sizes[size],
        pulse && 'animate-pulse-slow',
        className
      )}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-current"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
        </span>
      )}
      {children}
    </span>
  )
}

// Convenience components
export function SignalBadge({ signal, pulse = false }: { signal: string, pulse?: boolean }) {
  const variant = signal.toLowerCase().replace('_', '-') as BadgeVariant
  return <Badge variant={variant} pulse={pulse}>{signal.replace('_', ' ')}</Badge>
}

export function RegimeBadge({ regime }: { regime: string }) {
  const variant = regime.toLowerCase().replace('_', '-') as BadgeVariant
  const labels: Record<string, string> = {
    'bull': '🟢 BULL',
    'bear': '🔴 BEAR',
    'sideways': '🟡 SIDEWAYS',
    'high-vol': '🟣 HIGH VOL',
  }
  return <Badge variant={variant}>{labels[regime.toLowerCase()] || regime}</Badge>
}
