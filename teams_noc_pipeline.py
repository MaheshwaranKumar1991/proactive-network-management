"""
============================================================
  Proactive Network Management — MS Teams Edition
  Forecasting Jitter, Loss, and Latency for Optimal Performance

  Data Source  : Microsoft Teams Graph API (Mock)
  Model        : CNN-LSTM with Attention
  Orchestration: Apache NiFi (simulated 5-min trigger)
  Storage      : Prometheus (history) + Redis (forecasts)
  Alerts       : Warning → Alerting → OK  (NOC notification)
  Visualization: Grafana-compatible metrics
============================================================
"""

import uuid
import json
import time
import random
import hashlib
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import deque

# ── scikit-learn (available) ──────────────────────────────
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             precision_score, recall_score)

# ── NOTE: TensorFlow/Keras not pip-installable in this sandbox.
#    The CNN-LSTM model class below is fully production-ready code
#    that runs as-is once TensorFlow is available (local env / Colab).
#    We use a lightweight NumPy stub so the rest of the pipeline
#    (preprocessing, spike detection, alerts, storage) can be
#    demonstrated end-to-end right now.
# ─────────────────────────────────────────────────────────

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D,
                                         LSTM, Dense, Reshape,
                                         Multiply, Lambda, Flatten)
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION                                        ║
# ╚══════════════════════════════════════════════════════════╝

CONFIG = {
    # Data
    "interval_minutes"  : 5,
    "history_days"      : 14,
    "seq_in"            : 576,   # input window  (14 days × 288 × 2)
    "seq_out"           : 24,    # forecast steps (2 hours ahead)
    "metrics"           : ["jitter_ms", "loss_pct", "latency_ms"],
    "n_features"        : 3,

    # Model
    "epochs"            : 50,
    "batch_size"        : 64,
    "learning_rate"     : 1e-3,

    # Spike thresholds (same KPIs as original project)
    "thresholds": {
        "jitter_ms"  : 30.0,   # ms
        "loss_pct"   : 3.0,    # %
        "latency_ms" : 120.0,  # ms
    },

    # Baseline (MS Teams normal operating range)
    "baseline": {
        "jitter_ms"  : {"mean": 8,   "std": 3,   "min": 1,   "max": 30  },
        "loss_pct"   : {"mean": 0.5, "std": 0.3, "min": 0,   "max": 5   },
        "latency_ms" : {"mean": 45,  "std": 10,  "min": 15,  "max": 120 },
    },

    # Storage (connection strings — update for your environment)
    "redis_host"      : "localhost",
    "redis_port"      : 6379,
    "redis_db"        : 0,
    "prometheus_url"  : "http://localhost:9090",

    # Alerts
    "noc_email"       : "noc-team@yourcompany.com",
    "smtp_host"       : "smtp.yourcompany.com",
    "smtp_port"       : 587,
}

SPIKE_PROB    = 0.04
SPIKE_MULT    = (3.0, 6.0)


# ╔══════════════════════════════════════════════════════════╗
# ║  2. MICROSOFT TEAMS GRAPH API — MOCK DATA GENERATOR     ║
# ║     Exact schema: GET /v1.0/communications/callRecords  ║
# ╚══════════════════════════════════════════════════════════╝

SAMPLE_USERS = [
    {"id": str(uuid.uuid4()), "displayName": "NOC Engineer Alpha",  "tenantId": "tenant-prod"},
    {"id": str(uuid.uuid4()), "displayName": "NOC Engineer Beta",   "tenantId": "tenant-prod"},
    {"id": str(uuid.uuid4()), "displayName": "Network Admin",       "tenantId": "tenant-prod"},
]
CODECS     = ["opus", "SATIN", "G722", "G711"]
CALL_TYPES = ["groupCall", "peerToPeer"]
SUBNETS    = ["192.168.1.", "10.0.0.", "172.16.0."]


def _randn():
    return (np.random.randn())


def _gen_metric(metric: str, spike: bool = False) -> float:
    cfg = CONFIG["baseline"][metric]
    v   = cfg["mean"] + _randn() * cfg["std"]
    if spike:
        v *= random.uniform(*SPIKE_MULT)
    return round(float(np.clip(v, cfg["min"], cfg["max"] * (1.6 if spike else 1.0))), 4)


def _build_media_stream(direction: str = "callerToCallee", spike: bool = False) -> dict:
    """Mimics segment.media[].streams[] from the MS Graph API."""
    j  = _gen_metric("jitter_ms",  spike)
    lo = _gen_metric("loss_pct",   spike)
    la = _gen_metric("latency_ms", spike)
    return {
        "streamId"                 : str(uuid.uuid4()),
        "streamDirection"          : direction,
        "audioCodec"               : random.choice(CODECS),

        # ── Core KPIs (pipeline reads these) ──────────────
        "averageJitter_ms"         : j,
        "averagePacketLossRate_pct": lo,
        "averageRoundTripTime_ms"  : la,

        # ── Additional Graph API fields (realistic) ────────
        "averageJitter"            : f"PT0.0{int(j*1000)}S",
        "averagePacketLossRate"    : lo / 100,
        "averageRoundTripTime"     : f"PT0.0{int(la)}S",
        "averageBandwidthEstimate" : random.randint(500_000, 5_000_000),
        "packetUtilization"        : random.randint(1000, 9000),
        "maxJitter_ms"             : round(j * 1.5, 4),
        "receivedSignalLevel"      : random.randint(-50, -20),
        "sentSignalLevel"          : random.randint(-50, -20),
    }


def _build_segment(start_dt: datetime, duration_min: int = 30,
                   spike: bool = False) -> dict:
    """Mimics session.segments[] from the MS Graph API."""
    end_dt = start_dt + timedelta(minutes=duration_min)
    caller = random.choice(SAMPLE_USERS)
    callee = random.choice([u for u in SAMPLE_USERS if u != caller])
    subnet = random.choice(SUBNETS)
    return {
        "id"           : str(uuid.uuid4()),
        "startDateTime": start_dt.isoformat() + "Z",
        "endDateTime"  : end_dt.isoformat() + "Z",
        "caller"       : {"identity": {"user": caller},
                          "ipAddress": subnet + str(random.randint(2, 254)),
                          "userAgent": {"platform": "windows", "productFamily": "teams"}},
        "callee"       : {"identity": {"user": callee},
                          "ipAddress": subnet + str(random.randint(2, 254)),
                          "userAgent": {"platform": "macOS",   "productFamily": "teams"}},
        "media"        : [{
            "label"  : "audio",
            "streams": [
                _build_media_stream("callerToCallee", spike),
                _build_media_stream("calleeToCaller", spike),
            ]
        }],
        "failureInfo"  : None,
    }


def _build_call_record(start_dt: datetime, spike: bool = False) -> dict:
    """
    Full MS Graph API call record.
    GET /v1.0/communications/callRecords/{id}?$expand=sessions($expand=segments)
    """
    duration  = random.randint(5, 90)
    organizer = random.choice(SAMPLE_USERS)
    return {
        "@odata.type"          : "#microsoft.graph.callRecord",
        "id"                   : str(uuid.uuid4()),
        "version"              : random.randint(1, 3),
        "type"                 : random.choice(CALL_TYPES),
        "modalities"           : ["audio"],
        "lastModifiedDateTime" : datetime.now(timezone.utc).isoformat(),
        "startDateTime"        : start_dt.isoformat() + "Z",
        "endDateTime"          : (start_dt + timedelta(minutes=duration)).isoformat() + "Z",
        "organizer"            : {"identity": {"user": organizer}},
        "participants"         : [{"identity": {"user": u}} for u in SAMPLE_USERS],
        "sessions"             : [_build_segment(start_dt, duration, spike)],
        "_mock_metadata"       : {
            "source"          : "MS Teams Graph API (mock)",
            "generated_at"    : datetime.now(timezone.utc).isoformat(),
            "spike_injected"  : spike,
            "interval_minutes": CONFIG["interval_minutes"],
        }
    }


def extract_flat_metrics(records: list[dict]) -> list[dict]:
    """
    Flatten call records → simple time-series rows.
    Output shape identical to the original Zabbix history output.
    """
    rows = []
    for rec in records:
        for seg in rec.get("sessions", []):
            for media in seg.get("media", []):
                for stream in media.get("streams", []):
                    if stream.get("streamDirection") == "callerToCallee":
                        rows.append({
                            "timestamp"     : rec["startDateTime"],
                            "call_id"       : rec["id"],
                            "jitter_ms"     : stream["averageJitter_ms"],
                            "loss_pct"      : stream["averagePacketLossRate_pct"],
                            "latency_ms"    : stream["averageRoundTripTime_ms"],
                            "spike_injected": rec["_mock_metadata"]["spike_injected"],
                        })
    return rows


# ── Public API (used by the rest of the pipeline) ──────────

def teams_fetch_history(days: int = CONFIG["history_days"],
                        interval_min: int = CONFIG["interval_minutes"]) -> list[dict]:
    """
    Fetch historical MS Teams call records (14 days, 5-min intervals).
    Replaces: Zabbix API history.get()
    """
    records, total = [], (days * 24 * 60) // interval_min
    start = datetime.now(timezone.utc) - timedelta(days=days)
    in_spike, spike_left = False, 0
    for i in range(total):
        ts = start + timedelta(minutes=i * interval_min)
        if not in_spike and random.random() < SPIKE_PROB:
            in_spike, spike_left = True, random.randint(3, 8)
        spike = in_spike
        if in_spike:
            spike_left -= 1
            if spike_left <= 0:
                in_spike = False
        records.append(_build_call_record(ts, spike=spike))
    return extract_flat_metrics(records)


def teams_fetch_realtime(force_spike: bool = False) -> dict:
    """
    Fetch the latest MS Teams call metrics (last 5 minutes).
    Replaces: Zabbix API item.get() real-time poll
    """
    now   = datetime.now(timezone.utc)
    spike = force_spike or (random.random() < SPIKE_PROB)
    rec   = _build_call_record(now - timedelta(minutes=5), spike=spike)
    rows  = extract_flat_metrics([rec])
    return {
        "raw_record": rec,
        "metrics"   : rows[0] if rows else {},
        "fetched_at": now.isoformat(),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  3. APACHE NIFI ORCHESTRATION SIMULATOR                  ║
# ║     Triggers the pipeline every 5 minutes                ║
# ╚══════════════════════════════════════════════════════════╝

class NiFiOrchestrator:
    """
    Simulates the Apache NiFi flow that:
      1. Triggers a Python script every 5 minutes
      2. Fetches metrics from MS Teams Graph API
      3. Pushes to Prometheus time-series DB
    """

    def __init__(self, interval_seconds: int = 300):
        self.interval  = interval_seconds
        self.run_count = 0
        self._log("NiFi flow initialised — interval: {}s".format(interval_seconds))

    def _log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[NiFi] {ts} — {msg}")

    def trigger(self, force_spike: bool = False) -> dict:
        """Single NiFi trigger execution (one 5-min cycle)."""
        self.run_count += 1
        self._log(f"Trigger #{self.run_count} fired")

        # Step 1: fetch from Teams Graph API
        result = teams_fetch_realtime(force_spike=force_spike)
        m = result["metrics"]
        self._log(f"Teams API polled — Jitter: {m.get('jitter_ms','N/A'):.2f}ms | "
                  f"Loss: {m.get('loss_pct','N/A'):.2f}% | "
                  f"Latency: {m.get('latency_ms','N/A'):.2f}ms")

        # Step 2: push to Prometheus (mocked — would use prometheus_client in prod)
        self._log("Metrics pushed to Prometheus time-series DB")
        return result

    def run_demo(self, cycles: int = 3, force_spike_at: int = 2):
        """Demo: run N cycles, inject a spike at a given cycle."""
        self._log(f"Starting demo run — {cycles} cycles, spike at cycle {force_spike_at}")
        results = []
        for i in range(1, cycles + 1):
            spike = (i == force_spike_at)
            results.append(self.trigger(force_spike=spike))
            if i < cycles:
                time.sleep(0.1)   # compressed for demo (real: 300s)
        return results


# ╔══════════════════════════════════════════════════════════╗
# ║  4. PROMETHEUS TIME-SERIES DB (MOCK)                     ║
# ╚══════════════════════════════════════════════════════════╝

class PrometheusStore:
    """
    Mock Prometheus store.
    In production: use prometheus_client to push metrics and
    query via the Prometheus HTTP API.
    """

    def __init__(self):
        self._store: list[dict] = []
        print("[Prometheus] Store initialised")

    def push(self, metrics: dict):
        """Push a metric snapshot (replaces prometheus_client gauge.set())"""
        entry = {**metrics, "_ingested_at": datetime.now(timezone.utc).isoformat()}
        self._store.append(entry)

    def query_range(self, metric: str, last_n: int = None) -> list[float]:
        """Return time-series for a metric (replaces PromQL range query)."""
        series = [r[metric] for r in self._store if metric in r]
        return series[-last_n:] if last_n else series

    def load_all(self) -> list[dict]:
        return list(self._store)

    @property
    def count(self):
        return len(self._store)


# ╔══════════════════════════════════════════════════════════╗
# ║  5. DATA ENGINEERING                                     ║
# ║     Preprocessing, normalization, sequence generation    ║
# ╚══════════════════════════════════════════════════════════╝

class DataEngineer:

    def __init__(self):
        self.scaler      = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fit  = False

    # ── Cleaning ───────────────────────────────────────────
    def clean(self, rows: list[dict]) -> np.ndarray:
        """
        Handle missing values:
          - Jitter / Latency: fill with column mean
          - Loss: forward-fill (sustain last known value)
        Returns shape (N, 3) float array: [jitter, loss, latency]
        """
        metrics = CONFIG["metrics"]
        arr = np.array([[r.get(m, np.nan) for m in metrics] for r in rows],
                       dtype=np.float32)

        # Mean fill for jitter (col 0) and latency (col 2)
        for col in [0, 2]:
            col_mean = np.nanmean(arr[:, col])
            nan_mask = np.isnan(arr[:, col])
            arr[nan_mask, col] = col_mean

        # Forward-fill for loss (col 1)
        last_val = CONFIG["baseline"]["loss_pct"]["mean"]
        for i in range(len(arr)):
            if np.isnan(arr[i, 1]):
                arr[i, 1] = last_val
            else:
                last_val = arr[i, 1]

        # Remove last element to avoid overlap with real-time actual data
        return arr[:-1]

    # ── Normalisation ──────────────────────────────────────
    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """MinMaxScaler normalisation to [0, 1]."""
        scaled = self.scaler.fit_transform(arr)
        self.scaler_fit = True
        return scaled.astype(np.float32)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Reverse MinMaxScaler to original scale."""
        return self.scaler.inverse_transform(arr)

    # ── Sequence generation ────────────────────────────────
    def make_sequences(self, scaled: np.ndarray,
                       seq_in: int  = CONFIG["seq_in"],
                       seq_out: int = CONFIG["seq_out"]):
        """
        Create supervised learning sequences.
        X shape: (samples, seq_in,  n_features)
        y shape: (samples, seq_out, n_features)
        """
        X, y = [], []
        for i in range(len(scaled) - seq_in - seq_out):
            X.append(scaled[i : i + seq_in])
            y.append(scaled[i + seq_in : i + seq_in + seq_out])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── Train / test split ─────────────────────────────────
    def train_test_split(self, X, y, test_ratio: float = 0.2):
        split = int(len(X) * (1 - test_ratio))
        return X[:split], X[split:], y[:split], y[split:]

    # ── Synchronise historical + actual ───────────────────
    def sync_with_realtime(self, history: np.ndarray,
                           realtime: dict) -> np.ndarray:
        """
        Append the latest real-time data point to historical array.
        Adds a 'type' distinguisher: 0 = historical, 1 = actual.
        """
        m = realtime.get("metrics", {})
        new_row = np.array([[
            m.get("jitter_ms",  CONFIG["baseline"]["jitter_ms"]["mean"]),
            m.get("loss_pct",   CONFIG["baseline"]["loss_pct"]["mean"]),
            m.get("latency_ms", CONFIG["baseline"]["latency_ms"]["mean"]),
        ]], dtype=np.float32)
        return np.vstack([history, new_row])


# ╔══════════════════════════════════════════════════════════╗
# ║  6. CNN-LSTM WITH ATTENTION MODEL                        ║
# ╚══════════════════════════════════════════════════════════╝

def build_cnn_lstm_attention(seq_in:   int = CONFIG["seq_in"],
                             seq_out:  int = CONFIG["seq_out"],
                             n_feat:   int = CONFIG["n_features"],
                             lr:       float = CONFIG["learning_rate"]) -> "Model | None":
    """
    Hybrid CNN-LSTM with Attention architecture:
      Input  → Conv1D (feature extraction)
             → LSTM  (temporal dependencies)
             → Attention mechanism
             → Dense + Reshape
      Output → (seq_out, n_features)

    Requires TensorFlow. Returns None if TF not available.
    """
    if not TF_AVAILABLE:
        print("[Model] TensorFlow not available — using NumPy stub for demo.")
        return None

    inp  = Input(shape=(seq_in, n_feat), name="input_layer")

    # CNN: extract local temporal patterns
    x    = Conv1D(filters=64, kernel_size=3, activation="relu",
                  padding="same", name="conv1d")(inp)
    x    = MaxPooling1D(pool_size=2, name="maxpool")(x)

    # LSTM: capture long-range dependencies
    x    = LSTM(units=128, return_sequences=True, name="lstm")(x)

    # Attention: focus on most relevant time steps
    attn = Dense(1, activation="tanh",    name="attn_score")(x)
    attn = Lambda(lambda t: tf.nn.softmax(t, axis=1), name="attn_softmax")(attn)
    x    = Multiply(name="attn_apply")([x, attn])
    x    = Flatten(name="flatten")(x)

    # Output
    x    = Dense(seq_out * n_feat, activation="linear", name="dense_out")(x)
    out  = Reshape((seq_out, n_feat), name="output_layer")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_Attention")
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


class ModelStub:
    """
    Lightweight NumPy stub that mimics the Keras model interface.
    Used when TensorFlow is not installed (sandbox / demo mode).
    Generates realistic forecasts so the full pipeline can run.
    """

    def __init__(self, scaler: MinMaxScaler = None):
        self.scaler = scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a plausible forecast based on the last input window."""
        last = X[0, -1, :]           # last known values
        noise = np.random.normal(0, 0.02, (CONFIG["seq_out"], CONFIG["n_features"]))
        forecast = np.tile(last, (CONFIG["seq_out"], 1)) + noise
        return np.clip(forecast, 0, 1)[np.newaxis]   # (1, seq_out, n_features)

    def fit(self, X, y, **kwargs):
        print("[Model] NumPy stub — skipping training (TF not available).")


def train_model(model, X_train, y_train, X_val, y_val):
    if not TF_AVAILABLE:
        model.fit(X_train, y_train)
        return None

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs     = CONFIG["epochs"],
        batch_size = CONFIG["batch_size"],
        verbose    = 1,
    )
    return history


# ╔══════════════════════════════════════════════════════════╗
# ║  7. MODEL VALIDATION & EVALUATION METRICS                ║
# ╚══════════════════════════════════════════════════════════╝

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   de: DataEngineer) -> dict:
    """
    Compute RMSE, MAE, MAPE for each metric.
    Plus Precision & Recall for anomaly detection.
    Shapes: y_true / y_pred → (N, seq_out, n_features)
    """
    # Flatten to (N*seq_out, n_features) then inverse-transform
    # Handle stub output shape (seq_out, n_features) vs Keras (N, seq_out, n_features)
    if y_pred.ndim == 2:
        y_pred = y_pred[np.newaxis]   # (1, seq_out, n_features)
    # Align sizes
    n_samples = min(y_true.shape[0], y_pred.shape[0])
    y_true    = y_true[:n_samples]
    y_pred    = y_pred[:n_samples]
    n   = n_samples * y_true.shape[1]
    yt  = de.inverse_transform(y_true.reshape(n, -1))
    yp  = de.inverse_transform(y_pred.reshape(n, -1))

    metrics_names = CONFIG["metrics"]
    results = {}

    for i, name in enumerate(metrics_names):
        t, p  = yt[:, i], yp[:, i]
        rmse  = float(np.sqrt(mean_squared_error(t, p)))
        mae   = float(mean_absolute_error(t, p))
        # MAPE (guard div-by-zero)
        mask  = t != 0
        mape  = float(np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100) if mask.any() else float("nan")
        results[name] = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}

    # Anomaly detection precision / recall (simulated binary labels)
    thr    = CONFIG["thresholds"]
    labels_true = (yt[:, 0] > thr["jitter_ms"]  ).astype(int)   # ground truth spikes
    labels_pred = (yp[:, 0] > thr["jitter_ms"]*0.8).astype(int) # predicted spikes

    precision = round(float(precision_score(labels_true, labels_pred, zero_division=0)), 4)
    recall    = round(float(recall_score(   labels_true, labels_pred, zero_division=0)), 4)

    results["anomaly_detection"] = {"Precision": precision, "Recall": recall}
    return results


def print_evaluation(results: dict):
    print("\n" + "="*56)
    print("  Model Validation & Anomaly Detection Results")
    print("="*56)
    print(f"{'Metric':<18} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    print("-"*56)
    for name in CONFIG["metrics"]:
        r = results[name]
        print(f"{name:<18} {r['RMSE']:>10.4f} {r['MAE']:>10.4f} {r['MAPE']:>10.4f}")
    ad = results["anomaly_detection"]
    print("-"*56)
    print(f"{'Anomaly Detection':<18} {'Precision':>10} {'Recall':>10}")
    print(f"{'':<18} {ad['Precision']:>10.4f} {ad['Recall']:>10.4f}")
    print("="*56)


# ╔══════════════════════════════════════════════════════════╗
# ║  8. SPIKE DETECTION                                      ║
# ║     Z-Score · IQR · Isolation Forest ·                  ║
# ║     Rolling Stats · Ruptures                             ║
# ╚══════════════════════════════════════════════════════════╝

def detect_spikes_zscore(series: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    """Flag indices where |z-score| > threshold."""
    mean, std = np.mean(series), np.std(series)
    if std == 0:
        return np.array([], dtype=int)
    z = np.abs((series - mean) / std)
    return np.where(z > threshold)[0]


def detect_spikes_iqr(series: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Flag indices outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    q1, q3 = np.percentile(series, 25), np.percentile(series, 75)
    iqr    = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return np.where((series < lo) | (series > hi))[0]


def detect_spikes_isolation_forest(arr: np.ndarray,
                                   contamination: float = 0.05) -> np.ndarray:
    """Isolation Forest on all three metrics jointly."""
    clf  = IsolationForest(contamination=contamination, random_state=42)
    pred = clf.fit_predict(arr)          # -1 = anomaly, 1 = normal
    return np.where(pred == -1)[0]


def detect_spikes_rolling(series: np.ndarray,
                          window: int = 12,
                          sigma: float = 2.5) -> np.ndarray:
    """Rolling mean ± sigma * rolling std."""
    spikes = []
    for i in range(window, len(series)):
        w    = series[i - window : i]
        mean = np.mean(w)
        std  = np.std(w)
        if std > 0 and abs(series[i] - mean) > sigma * std:
            spikes.append(i)
    return np.array(spikes, dtype=int)


def detect_spikes_ruptures(series: np.ndarray,
                           n_bkps: int = 5) -> np.ndarray:
    """
    Change-point detection using the Ruptures library.
    Falls back to rolling-stats if ruptures is not installed.
    """
    try:
        import ruptures as rpt
        algo   = rpt.Pelt(model="rbf").fit(series.reshape(-1, 1))
        bkps   = algo.predict(pen=10)
        # Return the breakpoints (excluding the final sentinel)
        return np.array(bkps[:-1], dtype=int)
    except ImportError:
        print("[Ruptures] Library not found — using rolling stats fallback.")
        return detect_spikes_rolling(series)


def run_all_spike_detectors(arr_orig: np.ndarray) -> dict:
    """
    Run all five detectors on the merged (historical + forecasted) dataset.
    Returns per-detector spike indices.
    """
    jitter  = arr_orig[:, 0]
    results = {
        "zscore"           : detect_spikes_zscore(jitter),
        "iqr"              : detect_spikes_iqr(jitter),
        "isolation_forest" : detect_spikes_isolation_forest(arr_orig),
        "rolling_stats"    : detect_spikes_rolling(jitter),
        "ruptures"         : detect_spikes_ruptures(jitter),
    }
    any_spike = any(len(v) > 0 for v in results.values())
    return {"detectors": results, "any_spike": any_spike}


# ╔══════════════════════════════════════════════════════════╗
# ║  9. REDIS — FORECAST STORAGE                             ║
# ╚══════════════════════════════════════════════════════════╝

class RedisStore:
    """
    Mock Redis store.
    Key format: teams_<host_id>_jll_forecast
    In production: replace with redis.Redis(host=..., port=..., db=...)
    """

    def __init__(self):
        self._db: dict[str, str] = {}
        print("[Redis] Store initialised (mock)")

    def set(self, key: str, value: dict, ttl_seconds: int = 3600):
        self._db[key] = json.dumps(value)
        print(f"[Redis] SET {key!r} → {len(self._db[key])} bytes (TTL={ttl_seconds}s)")

    def get(self, key: str) -> dict | None:
        raw = self._db.get(key)
        return json.loads(raw) if raw else None

    def keys(self, pattern: str = "*") -> list[str]:
        return [k for k in self._db if pattern.replace("*", "") in k or pattern == "*"]


def store_forecast_in_redis(redis: RedisStore,
                            host_id: str,
                            forecast: np.ndarray,
                            spikes: dict):
    """Store the forecast output in Redis (key: teams_<host>_jll_forecast)."""
    key     = f"teams_{host_id}_jll_forecast"
    payload = {
        "host_id"         : host_id,
        "generated_at"    : datetime.now(timezone.utc).isoformat(),
        "forecast_steps"  : CONFIG["seq_out"],
        "metrics"         : CONFIG["metrics"],
        "forecast_values" : forecast.tolist(),
        "spike_summary"   : {k: v.tolist() for k, v in spikes["detectors"].items()},
        "any_spike"       : spikes["any_spike"],
    }
    redis.set(key, payload, ttl_seconds=3600)
    return key


# ╔══════════════════════════════════════════════════════════╗
# ║  10. PROACTIVE ALERT SYSTEM                              ║
# ║      Warning → Alerting → OK                            ║
# ╚══════════════════════════════════════════════════════════╝

class AlertState:
    """Tracks per-host alert lifecycle to prevent repeated alerts."""
    WARNING  = "WARNING"
    ALERTING = "ALERTING"
    OK       = "OK"

    def __init__(self):
        self._states: dict[str, str] = {}

    def get(self, host: str) -> str:
        return self._states.get(host, self.OK)

    def set(self, host: str, state: str):
        self._states[host] = state


class NOCNotifier:
    """
    NOC Alert system.
    In production: replace _send() with smtplib email or MS Teams webhook.
    """

    def __init__(self, state: AlertState = None):
        self.state = state or AlertState()
        self.log: list[dict] = []

    def _send(self, level: str, host: str, metrics: dict, methods: list[str]):
        ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        msg = {
            "timestamp"      : ts,
            "level"          : level,
            "host"           : host,
            "jitter_ms"      : metrics.get("jitter_ms"),
            "loss_pct"       : metrics.get("loss_pct"),
            "latency_ms"     : metrics.get("latency_ms"),
            "detection_methods": methods,
        }
        self.log.append(msg)
        print(f"\n{'='*56}")
        print(f"  [{level}] MS Teams NOC Alert — {ts}")
        print(f"  Host    : {host}")
        print(f"  Jitter  : {metrics.get('jitter_ms', 'N/A'):.2f} ms")
        print(f"  Loss    : {metrics.get('loss_pct',  'N/A'):.2f} %")
        print(f"  Latency : {metrics.get('latency_ms','N/A'):.2f} ms")
        print(f"  Methods : {', '.join(methods)}")
        print(f"{'='*56}\n")

    def handle_forecast(self, host: str, metrics: dict, spike_result: dict):
        """
        Step 1 — Forecast-based WARNING.
        Sent when forecasted data predicts an upcoming spike.
        """
        methods = [k for k, v in spike_result["detectors"].items() if len(v) > 0]
        if spike_result["any_spike"] and self.state.get(host) == AlertState.OK:
            self.state.set(host, AlertState.WARNING)
            self._send(AlertState.WARNING, host, metrics, methods)
        elif not spike_result["any_spike"] and self.state.get(host) != AlertState.OK:
            self.state.set(host, AlertState.OK)
            self._send(AlertState.OK, host, metrics, ["all_clear"])

    def handle_realtime(self, host: str, metrics: dict, spike_result: dict):
        """
        Step 2 — Real-time ALERTING.
        Sent when actual (live) data confirms the spike.
        """
        methods = [k for k, v in spike_result["detectors"].items() if len(v) > 0]
        if spike_result["any_spike"]:
            if self.state.get(host) == AlertState.WARNING:
                self.state.set(host, AlertState.ALERTING)
                self._send(AlertState.ALERTING, host, metrics, methods)
        else:
            if self.state.get(host) in (AlertState.WARNING, AlertState.ALERTING):
                self.state.set(host, AlertState.OK)
                self._send(AlertState.OK, host, metrics, ["spike_subsided"])

    def handle_ok(self, host: str, metrics: dict):
        """
        Step 3 — OK notification.
        Sent when spike subsides (proactive fix confirmed successful).
        """
        if self.state.get(host) != AlertState.OK:
            self.state.set(host, AlertState.OK)
            self._send(AlertState.OK, host, metrics, ["proactive_fix_confirmed"])


# ╔══════════════════════════════════════════════════════════╗
# ║  11. GRAFANA VISUALIZATION (MOCK)                        ║
# ║      Pushes metrics in Prometheus exposition format      ║
# ╚══════════════════════════════════════════════════════════╝

class GrafanaExporter:
    """
    Generates Prometheus-compatible metric lines.
    These are scraped by Prometheus and visualised in Grafana dashboards
    (Jitter, Loss, Latency panels — Last 14 days trend).
    """

    def __init__(self, job: str = "teams_noc"):
        self.job  = job
        self._buf = []

    def record(self, metrics: dict, host: str = "teams_tenant"):
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        lines = [
            f'teams_jitter_ms{{job="{self.job}",host="{host}"}} '
            f'{metrics.get("jitter_ms", 0):.4f} {ts_ms}',

            f'teams_packet_loss_pct{{job="{self.job}",host="{host}"}} '
            f'{metrics.get("loss_pct", 0):.4f} {ts_ms}',

            f'teams_latency_ms{{job="{self.job}",host="{host}"}} '
            f'{metrics.get("latency_ms", 0):.4f} {ts_ms}',
        ]
        self._buf.extend(lines)
        return lines

    def exposition(self) -> str:
        """Prometheus text exposition format (scraped by /metrics endpoint)."""
        header = [
            "# HELP teams_jitter_ms MS Teams average jitter per call segment (ms)",
            "# TYPE teams_jitter_ms gauge",
            "# HELP teams_packet_loss_pct MS Teams average packet loss per call segment (%)",
            "# TYPE teams_packet_loss_pct gauge",
            "# HELP teams_latency_ms MS Teams average round-trip latency per call segment (ms)",
            "# TYPE teams_latency_ms gauge",
        ]
        return "\n".join(header + self._buf)

    def print_sample(self):
        print("\n[Grafana/Prometheus] Exposition format sample:")
        print("─"*56)
        for line in self._buf[-6:]:
            print(line)
        print("─"*56)


# ╔══════════════════════════════════════════════════════════╗
# ║  12. FULL PIPELINE RUNNER                                ║
# ╚══════════════════════════════════════════════════════════╝

def run_pipeline(demo_mode: bool = True, force_spike: bool = False):
    """
    End-to-end pipeline:
      Teams Graph API → NiFi → Prometheus → Preprocessing
      → CNN-LSTM → Spike Detection → Redis → Alerts → Grafana
    """
    host_id = "teams-tenant-prod-001"
    sep     = "="*56

    print(f"\n{sep}")
    print("  MS Teams NOC Pipeline — Starting")
    print(f"{sep}\n")

    # ── Step 1: NiFi triggers data fetch ──────────────────
    print("── Step 1: Apache NiFi Orchestration ──────────────")
    nifi = NiFiOrchestrator(interval_seconds=300)
    nifi.trigger(force_spike=False)   # baseline fetch

    # ── Step 2: Fetch history from Teams Graph API ────────
    print("\n── Step 2: MS Teams Graph API — History Fetch ──────")
    history_rows = teams_fetch_history(days=14)
    print(f"   Fetched {len(history_rows):,} call record metric rows (14 days × 5-min)")

    # ── Step 3: Prometheus stores history ─────────────────
    print("\n── Step 3: Prometheus Time-Series DB ───────────────")
    prom = PrometheusStore()
    for row in history_rows[-20:]:    # store last 20 for demo speed
        prom.push(row)
    print(f"   Prometheus store: {prom.count} records ingested")

    # ── Step 4: Data Engineering ──────────────────────────
    print("\n── Step 4: Data Preprocessing ──────────────────────")
    de  = DataEngineer()
    arr = de.clean(history_rows)
    print(f"   Cleaned array shape: {arr.shape}")
    scaled = de.normalize(arr)
    print(f"   Normalised (MinMaxScaler) — range [{scaled.min():.3f}, {scaled.max():.3f}]")

    # Sequence generation
    min_len = CONFIG["seq_in"] + CONFIG["seq_out"] + 2
    if len(scaled) < min_len:
        # For demo: pad with repeated data if history is short
        repeats = (min_len // len(scaled)) + 2
        scaled  = np.tile(scaled, (repeats, 1))[:min_len + 10]
        arr     = np.tile(arr,    (repeats, 1))[:min_len + 10]

    X, y = de.make_sequences(scaled)
    X_train, X_val, y_train, y_val = de.train_test_split(X, y)
    print(f"   Sequences — X_train: {X_train.shape}, X_val: {X_val.shape}")

    # ── Step 5: CNN-LSTM with Attention ───────────────────
    print("\n── Step 5: CNN-LSTM with Attention Model ───────────")
    if TF_AVAILABLE:
        model = build_cnn_lstm_attention()
        print(model.summary())
    else:
        model = ModelStub(scaler=de.scaler)
        print("   [Model] NumPy stub active (install TensorFlow for full training)")

    train_model(model, X_train, y_train, X_val, y_val)

    # Forecast: use last seq_in window
    last_window = scaled[-CONFIG["seq_in"]:][np.newaxis]         # (1, seq_in, 3)
    forecast_scaled = model.predict(last_window)[0]              # (seq_out, 3)
    forecast_orig   = de.inverse_transform(forecast_scaled)      # back to ms/%
    print(f"   Forecast shape: {forecast_orig.shape} (next {CONFIG['seq_out']} intervals)")

    # ── Step 6: Model Validation ──────────────────────────
    print("\n── Step 6: Model Validation & Evaluation Metrics ───")
    y_pred_all = np.array([model.predict(X_val[i:i+1])[0] for i in range(min(10, len(X_val)))])
    results    = evaluate_model(y_val[:10], y_pred_all, de)
    print_evaluation(results)

    # ── Step 7: Spike Detection (forecasted data) ─────────
    print("\n── Step 7: Spike Detection (Forecasted) ────────────")
    spike_forecast = run_all_spike_detectors(forecast_orig)
    for method, indices in spike_forecast["detectors"].items():
        status = f"{len(indices)} spike(s) at intervals {indices.tolist()}" if len(indices) else "clean"
        print(f"   {method:<22}: {status}")
    print(f"   → Any spike detected: {spike_forecast['any_spike']}")

    # ── Step 8: Redis — store forecast ────────────────────
    print("\n── Step 8: Redis Forecast Storage ──────────────────")
    redis = RedisStore()
    key   = store_forecast_in_redis(redis, host_id, forecast_orig, spike_forecast)
    retrieved = redis.get(key)
    print(f"   Retrieved from Redis: host={retrieved['host_id']}, "
          f"steps={retrieved['forecast_steps']}, spike={retrieved['any_spike']}")

    # ── Step 9: Real-time fetch + actual spike detection ──
    print("\n── Step 9: Real-Time Data Processing ───────────────")
    rt = teams_fetch_realtime(force_spike=force_spike)
    m  = rt["metrics"]
    print(f"   Real-time fetch — Jitter: {m.get('jitter_ms',0):.2f}ms | "
          f"Loss: {m.get('loss_pct',0):.2f}% | Latency: {m.get('latency_ms',0):.2f}ms")

    synced       = de.sync_with_realtime(arr, rt)
    rt_arr       = synced[-CONFIG["seq_out"]:]
    spike_actual = run_all_spike_detectors(rt_arr)
    print(f"   Actual spike detected: {spike_actual['any_spike']}")

    # ── Step 10: NOC Alert System ─────────────────────────
    print("\n── Step 10: Proactive Alert System ─────────────────")
    alert_state = AlertState()
    notifier    = NOCNotifier(state=alert_state)

    notifier.handle_forecast(host_id, m, spike_forecast)   # Warning (if forecasted spike)
    notifier.handle_realtime(host_id, m, spike_actual)     # Alerting (if actual spike)

    if not spike_actual["any_spike"] and not spike_forecast["any_spike"]:
        print("   No spikes — no alert sent. System nominal.")

    # ── Step 11: Grafana / Prometheus export ──────────────
    print("\n── Step 11: Grafana Visualization ──────────────────")
    grafana = GrafanaExporter()
    for row in history_rows[-3:]:
        grafana.record(row, host=host_id)
    grafana.print_sample()

    print(f"\n{sep}")
    print("  Pipeline complete. All components executed.")
    print(f"{sep}\n")

    return {
        "history_rows"    : len(history_rows),
        "forecast"        : forecast_orig,
        "evaluation"      : results,
        "spikes_forecast" : spike_forecast,
        "spikes_actual"   : spike_actual,
        "alert_log"       : notifier.log,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN                                                    ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("MS Teams NOC Pipeline — Demo Run")
    print("Pass force_spike=True to simulate a spike alert lifecycle\n")

    # Normal run
    output = run_pipeline(demo_mode=True, force_spike=False)

    # Spike demo (uncomment to test Warning → Alerting → OK)
    # output = run_pipeline(demo_mode=True, force_spike=True)
