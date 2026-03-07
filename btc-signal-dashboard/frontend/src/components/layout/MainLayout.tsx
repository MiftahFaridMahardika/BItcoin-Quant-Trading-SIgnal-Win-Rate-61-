'use client'

import { ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { useSignalStore } from '@/stores/signalStore'

interface MainLayoutProps {
  children: ReactNode
}

export function MainLayout({ children }: MainLayoutProps) {
  const { marketInfo, connected, currentSignal } = useSignalStore()

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <div className="pl-64 transition-all duration-300">
        <Header
          price={marketInfo?.price}
          change24h={marketInfo?.change_24h}
          connected={connected}
          lastUpdate={currentSignal?.timestamp}
        />

        <main className="p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
