# Bitcoin Historical Data — Inspection Report

**Generated:** 2026-03-04
**Tool:** Claude Code Data Inspector

---

## 1. File Discovery

| # | File Name | Size | Format |
|---|-----------|------|--------|
| 1 | `btcusd_1-min_data.csv` | 373 MB (391,146,732 bytes) | CSV |

**Total files found:** 1
**Total raw size:** 373 MB

---

## 2. File Detail: `btcusd_1-min_data.csv`

### 2.1 Column Names

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `Timestamp` | float (Unix epoch) | Waktu candle dalam detik sejak 1970-01-01 UTC |
| 2 | `Open` | float | Harga pembukaan |
| 3 | `High` | float | Harga tertinggi |
| 4 | `Low` | float | Harga terendah |
| 5 | `Close` | float | Harga penutupan |
| 6 | `Volume` | float | Volume trading (dalam BTC) |

### 2.2 Sample Data — First 5 Rows

```
Timestamp,        Open,  High,  Low,   Close, Volume
1325412060.0,     4.58,  4.58,  4.58,  4.58,  0.0
1325412120.0,     4.58,  4.58,  4.58,  4.58,  0.0
1325412180.0,     4.58,  4.58,  4.58,  4.58,  0.0
1325412240.0,     4.58,  4.58,  4.58,  4.58,  0.0
1325412300.0,     4.58,  4.58,  4.58,  4.58,  0.0
```

> **Catatan:** 5 baris pertama menunjukkan harga Bitcoin di awal Januari 2012 sekitar $4.58 dengan zero volume — periode awal likuiditas sangat rendah.

### 2.3 Sample Data — Last 5 Rows

```
Timestamp,        Open,    High,    Low,     Close,   Volume
1772495820.0,     68797,   68801,   68775,   68801,   2.39035319
1772495880.0,     68801,   68801,   68778,   68779,   0.35180718
1772495940.0,     68779,   68821,   68775,   68821,   12.24369073
1772496000.0,     68821,   68847,   68807,   68827,   0.81499942
1772496060.0,     68827,   68827,   68793,   68795,   1.63637616
```

> **Catatan:** 5 baris terakhir menunjukkan harga Bitcoin di awal Maret 2026 sekitar $68,800 dengan volume aktif.

### 2.4 Timestamp Format

| Property | Value |
|----------|-------|
| Format | Unix Epoch (detik), tipe `float` dengan suffix `.0` |
| Granularity | 1 menit (interval = 60 detik) |
| Timezone | UTC |
| Contoh konversi | `1325412060.0` → `2012-01-01 10:01:00 UTC` |

**Cara konversi ke datetime Python:**
```python
import datetime
dt = datetime.datetime.utcfromtimestamp(1325412060.0)
# Output: datetime(2012, 1, 1, 10, 1, 0)
```

---

## 3. Data Statistics

### 3.1 Row Count & Date Range

| Metric | Value |
|--------|-------|
| **Total baris data** | **7,450,241 baris** |
| **Header row** | 1 baris |
| **Total baris file** | 7,450,242 |
| **Tanggal pertama** | 2012-01-01 10:01:00 UTC |
| **Tanggal terakhir** | 2026-03-03 00:01:00 UTC |
| **Rentang waktu** | ±14 tahun 2 bulan |

### 3.2 Price Statistics

| Metric | Value |
|--------|-------|
| **Harga Close minimum** | $3.80 |
| **Harga Close maksimum** | $126,202.00 |
| **Kenaikan total** | ~33,210x dari harga awal |

---

## 4. Data Quality Analysis

### 4.1 Missing Values

| Column | Missing Count | Status |
|--------|--------------|--------|
| Timestamp | 0 | CLEAN |
| Open | 0 | CLEAN |
| High | 0 | CLEAN |
| Low | 0 | CLEAN |
| Close | 0 | CLEAN |
| Volume | 0 | CLEAN |
| **Total** | **0** | **CLEAN** |

### 4.2 Duplicate Timestamps

| Check | Result | Status |
|-------|--------|--------|
| Timestamp duplikat | 0 | CLEAN |

### 4.3 Anomalies Detection

| Anomaly Type | Count | Persentase | Status |
|-------------|-------|-----------|--------|
| Harga negatif (Open/High/Low/Close < 0) | 0 | 0% | CLEAN |
| Invalid OHLC (High < Low) | 0 | 0% | CLEAN |
| Zero Volume | 1,310,680 | 17.59% | EXPECTED |
| Volume > 0 | 6,139,561 | 82.41% | NORMAL |

### 4.4 Completeness (Gap Analysis)

| Metric | Value |
|--------|-------|
| Menit yang seharusnya ada | 7,451,401 |
| Menit yang tersedia | 7,450,241 |
| **Candle yang hilang** | **1,160** |
| **Gap rate** | **0.0156%** |

---

## 5. Data Quality Score

```
╔══════════════════════════════════════════════════╗
║           DATA QUALITY SCORECARD                 ║
╠══════════════════════════════════════════════════╣
║  Missing Values         : 0/7,450,241   → 100%  ║
║  No Duplicates          : 0 duplicate   → 100%  ║
║  No Negative Prices     : 0 anomaly     → 100%  ║
║  Valid OHLC Structure   : 0 violation   → 100%  ║
║  Completeness (candles) : 1160 missing  →  99.98%║
╠══════════════════════════════════════════════════╣
║  OVERALL DATA QUALITY SCORE:      99.84 / 100   ║
║  GRADE:                           A+ (EXCELLENT) ║
╚══════════════════════════════════════════════════╝
```

**Penjelasan score:**
- -0.16 poin karena 1,160 candle yang hilang (gap rate 0.0156%)
- Zero volume 17.59% **tidak mengurangi score** karena ini adalah perilaku normal untuk periode 2012-2015 ketika likuiditas sangat rendah

---

## 6. Rekomendasi Preprocessing

### 6.1 WAJIB (Must-Do)

| # | Action | Alasan |
|---|--------|--------|
| 1 | **Konversi Timestamp** | Ubah dari Unix epoch float ke `datetime64[ns]` pandas untuk kemudahan analisis |
| 2 | **Set Timestamp sebagai Index** | Set sebagai DatetimeIndex untuk time-series operations |
| 3 | **Handle Missing Candles (1,160 gaps)** | Isi gap dengan `resample('1T').asfreq()` lalu forward-fill atau tandai sebagai NaN |

### 6.2 DISARANKAN (Recommended)

| # | Action | Alasan |
|---|--------|--------|
| 4 | **Handle Zero Volume** | Untuk backtesting: pertimbangkan filter/exclude 1.31 juta baris zero-volume untuk mencegah sinyal palsu |
| 5 | **Normalisasi harga** | Jika digunakan untuk ML: pertimbangkan log-transform atau percentage returns karena range harga sangat lebar ($3.80 → $126,202) |
| 6 | **Konversi ke UTC timezone-aware** | Tambahkan `tz_localize('UTC')` untuk menghindari ambiguitas |
| 7 | **Chunked reading** | File 373 MB — gunakan `pd.read_csv(..., chunksize=100000)` jika RAM terbatas |

### 6.3 OPSIONAL (Optional)

| # | Action | Alasan |
|---|--------|--------|
| 8 | **Resample ke timeframe lebih besar** | Buat 5m, 15m, 1H, 4H, 1D OHLCV dari data 1m untuk multi-timeframe analysis |
| 9 | **Split train/test** | Pisahkan: pre-2020 untuk training, 2020+ untuk testing (atau sesuai kebutuhan) |
| 10 | **Simpan ke Parquet** | Convert ke `.parquet` untuk akses 5-10x lebih cepat dan ukuran lebih kecil (~60-80 MB) |

### 6.4 Contoh Kode Preprocessing

```python
import pandas as pd

# 1. Load dengan dtype optimization
df = pd.read_csv(
    'btcusd_1-min_data.csv',
    dtype={
        'Timestamp': 'float64',
        'Open': 'float32',
        'High': 'float32',
        'Low': 'float32',
        'Close': 'float32',
        'Volume': 'float32'
    }
)

# 2. Konversi timestamp dan set index
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True)
df.set_index('Timestamp', inplace=True)
df.sort_index(inplace=True)

# 3. Handle missing candles (isi gap)
df = df.resample('1min').asfreq()  # Insert NaN untuk candle yang hilang
df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].fillna(method='ffill')
df['Volume'] = df['Volume'].fillna(0)

# 4. Filter zero-volume (opsional, untuk trading signals)
df_active = df[df['Volume'] > 0].copy()

# 5. Simpan ke parquet untuk akses cepat
df.to_parquet('btcusd_1min_clean.parquet', engine='pyarrow')

print(f"Data siap: {len(df):,} baris | {df.index[0]} s/d {df.index[-1]}")
```

---

## 7. Summary Report

```
╔═══════════════════════════════════════════════════════════╗
║              BITCOIN DATA INSPECTION SUMMARY              ║
╠═══════════════════════════════════════════════════════════╣
║  File           : btcusd_1-min_data.csv                   ║
║  Size           : 373 MB                                  ║
║  Total Rows     : 7,450,241 candles (1-minute)            ║
║  Date Range     : 2012-01-01 → 2026-03-03 (~14.2 tahun)  ║
║  Columns        : Timestamp, Open, High, Low, Close, Vol  ║
╠═══════════════════════════════════════════════════════════╣
║  Missing Values : 0                    ✓ CLEAN            ║
║  Duplicates     : 0                    ✓ CLEAN            ║
║  Negative Price : 0                    ✓ CLEAN            ║
║  OHLC Violation : 0                    ✓ CLEAN            ║
║  Missing Candles: 1,160 (0.016%)       ✓ EXCELLENT        ║
║  Zero Volume    : 1,310,680 (17.59%)   ~ EXPECTED         ║
╠═══════════════════════════════════════════════════════════╣
║  Min Price      : $3.80 (2012)                            ║
║  Max Price      : $126,202.00 (2026)                      ║
║  Quality Score  : 99.84/100            A+ EXCELLENT       ║
╚═══════════════════════════════════════════════════════════╝
```

---

*Report generated by Claude Code Data Inspector — 2026-03-04*
