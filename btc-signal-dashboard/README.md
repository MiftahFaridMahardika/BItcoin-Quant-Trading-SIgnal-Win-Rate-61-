# BTC Signal Dashboard

Real-time Bitcoin Trading Signal Dashboard with Arkham-style design.

## Quick Start (Development)

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in values
3. Run with Docker Compose:
```bash
docker-compose up -d
```

4. Open http://localhost:3000

## Architecture

- **Frontend**: Next.js 14 + Tailwind CSS
- **Backend**: FastAPI + Python
- **Database**: PostgreSQL
- **Cache**: Redis
- **Data Source**: Binance WebSocket

## Features

- Real-time price updates via WebSocket
- Trading signals with entry/SL/TP levels
- Market regime detection
- Risk metrics dashboard
- Signal history tracking

## Production Deployment

1. Set secure passwords in `.env`
2. Configure SSL certificates in `nginx/ssl/`
3. Run:
```bash
docker-compose -f docker-compose.yml up -d
```
