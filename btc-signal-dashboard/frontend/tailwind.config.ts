import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Arkham-style dark theme
        background: '#0D0D0F',
        surface: {
          DEFAULT: '#141417',
          elevated: '#1A1A1F',
          hover: '#202025',
        },
        border: {
          DEFAULT: '#2A2A30',
          light: '#3A3A42',
        },
        text: {
          primary: '#FFFFFF',
          secondary: '#8B8B8E',
          muted: '#5A5A5E',
        },
        accent: {
          blue: '#3B82F6',
          cyan: '#22D3EE',
          purple: '#8B5CF6',
        },
        status: {
          success: '#10B981',
          danger: '#EF4444',
          warning: '#F59E0B',
          info: '#3B82F6',
        },
        signal: {
          'strong-long': '#10B981',
          'long': '#34D399',
          'skip': '#6B7280',
          'short': '#F87171',
          'strong-short': '#EF4444',
        },
        regime: {
          bull: '#10B981',
          bear: '#EF4444',
          sideways: '#F59E0B',
          'high-vol': '#8B5CF6',
        }
      },
      fontFamily: {
        sans: ['Inter', 'SF Pro Display', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        'xxs': '0.625rem',
      },
      boxShadow: {
        'glow-blue': '0 0 20px rgba(59, 130, 246, 0.15)',
        'glow-green': '0 0 20px rgba(16, 185, 129, 0.15)',
        'glow-red': '0 0 20px rgba(239, 68, 68, 0.15)',
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.3)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-primary': 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
        'gradient-success': 'linear-gradient(135deg, #10B981, #34D399)',
        'gradient-danger': 'linear-gradient(135deg, #EF4444, #F87171)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(59, 130, 246, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(59, 130, 246, 0.4)' },
        }
      }
    },
  },
  plugins: [],
}

export default config
