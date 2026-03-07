"""
FastAPI Main Application — BTC Real-Time Trading Signal API
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from app.config import settings
from app.database import init_db
from app.core.data_fetcher import BinanceDataFetcher
from app.core.signal_generator import RealTimeSignalGenerator, SignalOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
data_fetcher: BinanceDataFetcher | None = None
signal_generator: RealTimeSignalGenerator | None = None
connected_clients: list[WebSocket] = []


# ── Broadcast helpers ─────────────────────────────────────────────────────────

async def broadcast_to_all(message: dict):
    """Send JSON to all live WebSocket clients, drop dead connections."""
    dead: list[WebSocket] = []
    for ws in connected_clients:
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


async def broadcast_signal(signal: SignalOutput):
    await broadcast_to_all({"type": "new_signal", "data": signal.to_dict()})


async def broadcast_price(price_data: dict):
    await broadcast_to_all({"type": "price_update", "data": price_data})


async def broadcast_regime_change(new_regime: str, old_regime: str):
    await broadcast_to_all({
        "type": "regime_change",
        "data": {"new": new_regime, "old": old_regime},
    })


async def broadcast_connection_status(status: str, error: str | None):
    await broadcast_to_all({
        "type": "connection_status",
        "data": {"status": status, "error": error},
    })


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_fetcher, signal_generator

    logger.info("Starting BTC Signal Dashboard Backend...")
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database unavailable — signals will not be persisted: {e}")

    data_fetcher = BinanceDataFetcher("btcusdt")
    signal_generator = RealTimeSignalGenerator(
        data_fetcher=data_fetcher,
        primary_timeframe=settings.BINANCE_TIMEFRAME,
        risk_per_trade=1_000.0,
    )

    # Wire callbacks before starting
    signal_generator.on_new_signal    = broadcast_signal
    signal_generator.on_price_update  = broadcast_price
    signal_generator.on_regime_change = broadcast_regime_change
    data_fetcher.on_connection_change  = broadcast_connection_status

    # Start signal generator (wires into fetcher callbacks)
    await signal_generator.start()

    # Start fetcher as background task (auto-reconnects)
    asyncio.create_task(data_fetcher.start())

    app.state.data_fetcher = data_fetcher
    app.state.signal_generator = signal_generator

    logger.info("Backend started successfully")
    yield

    logger.info("Shutting down...")
    await signal_generator.stop()
    await data_fetcher.stop()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BTC Signal Dashboard API",
    description="Real-time Bitcoin Trading Signal API powered by quant engines",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "running", "service": "BTC Signal Dashboard API"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "fetcher_running": data_fetcher.is_running if data_fetcher else False,
        "clients": len(connected_clients),
    }


@app.get("/api/market")
async def get_market_info():
    if not signal_generator:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    return signal_generator.get_market_summary()


@app.get("/api/signal/current")
async def get_current_signal():
    if not signal_generator:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    signal = signal_generator.get_current_signal()
    if signal:
        return signal
    return {"signal": "WAITING", "message": "Waiting for next candle close"}


@app.get("/api/signals/history")
async def get_signal_history(limit: int = 20):
    if not signal_generator:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    return {"signals": signal_generator.get_signal_history(limit)}


@app.get("/api/candles/{timeframe}")
async def get_candles(timeframe: str, limit: int = 100):
    if not data_fetcher:
        return JSONResponse({"error": "Service not ready"}, status_code=503)
    valid = ["1m", "5m", "15m", "1h", "4h", "1d"]
    if timeframe not in valid:
        return JSONResponse({"error": f"Invalid timeframe. Use: {valid}"}, status_code=400)
    candles = data_fetcher.get_candles(timeframe, limit)
    return {"timeframe": timeframe, "count": len(candles), "candles": candles}


@app.get("/api/indicators")
async def get_indicators():
    if not signal_generator or not signal_generator.current_signal:
        return JSONResponse({"error": "No data available"}, status_code=503)
    return {"indicators": signal_generator.current_signal.indicators}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"WS client connected — total: {len(connected_clients)}")

    try:
        # Send initial state immediately on connect
        await websocket.send_json({
            "type": "initial_state",
            "data": {
                "market":    signal_generator.get_market_summary() if signal_generator else None,
                "signal":    signal_generator.get_current_signal()  if signal_generator else None,
                "connected": data_fetcher.is_running if data_fetcher else False,
            },
        })

        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "ts": time.time()})

    except WebSocketDisconnect:
        logger.info("WS client disconnected")
    except Exception as e:
        logger.error(f"WS error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"WS clients remaining: {len(connected_clients)}")
