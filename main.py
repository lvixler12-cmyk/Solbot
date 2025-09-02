#!/usr/bin/env python3
# main.py — Solana meme snipe paper-trade bot
#
# Buy paths (checked each cycle, in this order):
#   1) Launch Snipe: brand-new Raydium/Pump/Meteora/Bunk pairs (+ optional RugCheck) + (optional 2x predictor)
#   2) Quantum Scorer: score ≥ threshold  AND  (liq_ok OR vol_ok)  AND  (2x predictor)
#   3) Base Filter: configurable logic across (liq, vol, cap): OR / AND / MAJORITY / HYBRID  (+ optional 2x predictor)
#
# Focus:
#   • Solana SPL meme tokens (base = new coin), quote in {SOL, USDC, USDT}
#   • Excludes majors (SOL/USDT, SOL/BTC, etc.)
#   • PAPER-TRADING ONLY (no real trades)
#
# Quick commands (Telegram):
#   /resume
#   /filtermode HYBRID
#   /mintrue 2
#   /hybridcut 3
#   /setage 8
#   /setcap 5000 2000000
#   /capunknown on
#   /setliq 1500
#   /setvolwin 5
#   /setvol 250
#   /launch on
#   /quantum on 60
#   /2x on 2 0.8        # enable 2x predictor, horizon=2min, alpha=0.8
#   /setmom 60          # momentum window seconds
#   /meme on
#   /norebuy on/off     # NEW: toggle re-buy prevention
#   /status  |  /diag

import os, re, time, json, threading, requests, signal, sys, math, statistics
from typing import List, Dict, Any, Tuple, Optional
from collections import deque, defaultdict

# =========================
# === Telegram Config ====
# =========================
TELEGRAM_BOT_TOKEN = "8413132819:AAGCskfRNS8BsY5MTxqJKsHpFNjYUmop9ms"
TELEGRAM_CHAT_ID   = "5794143622"   # numeric string

def tg_send(text: str, reply_markup: dict = None) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": str(text)}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        r = requests.post(url, json=payload, timeout=20)
        return r.ok
    except Exception as e:
        print("Telegram send error:", e)
        return False

def tg_answer_callback(callback_id: str, text: str = ""):
    if not callback_id: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
        requests.post(url, json={"callback_query_id": callback_id, "text": text}, timeout=10)
    except Exception as e:
        print("answerCallbackQuery error:", e)

# =========================
# === Files (Persistence)
# =========================
SETTINGS_FILE = "settings.json"
STATE_FILE    = "state.json"

DEFAULT_SETTINGS = {
    # ---------- Base Filter conditions (evaluated with a logic mode) ----------
    "min_liq": 500,                    # USD liquidity (very low for new meme tokens)
    "min_vol_usd": 50,                 # USD volume threshold (very low for new tokens)
    "vol_window_min": 5,               # volume window in minutes (short for new detection)
    "cap_range": [1000, 5_000_000],    # FDV band (USD) - wider meme range
    "enforce_cap": False,              # disable cap condition initially to catch more
    "pass_if_cap_unknown": True,       # unknown FDV counts as cap_ok

    # ---------- Filter Logic Mode ----------
    # Modes: "OR", "AND", "MAJORITY", "HYBRID"
    "filter_mode": "OR",               # Use OR mode to be most permissive
    # For MAJORITY/HYBRID fallback: how many of {liq, vol, cap} must be true
    "min_true": 1,                     # Very lenient for new tokens
    # HYBRID: if pair age ≤ hybrid_new_age_min, require liq AND vol; else MAJORITY(min_true)
    "hybrid_new_age_min": 10,          # Longer window for new token detection

    # ---------- Solana MEME focus ----------
    "solana_only": True,                                # only consider chainId == 'solana'
    "meme_only": True,                                  # apply meme candidate checks
    "meme_quote_whitelist": ["SOL", "USDC", "USDT"],    # acceptable quote tokens
    "meme_base_blacklist": ["SOL", "WSOL", "USDT", "USDC", "BTC", "ETH", "WBNB", "WAVAX", "WPLS"],

    # ---------- NO REBUY SYSTEM ----------
    "enable_no_rebuy": True,           # NEW: prevent re-buying sold tokens
    "no_rebuy_after_sell_sec": 3600,   # NEW: cooldown period (1 hour default)

    # ---------- Launch Snipe ----------
    "launch_enabled": True,
    "allowed_dex_ids": ["raydium", "pump", "meteora", "bunk", "jupiter"],  # added jupiter
    "max_pair_age_min": 30,             # pair must be ≤ N minutes old (increased to catch more)
    "launch_liq_min": 300,              # min liq for launch snipe path (very low for new memes)
    "launch_min_buys_m5": 1,            # buys in last 5m for early traction (very low threshold)
    "launch_require_2x": False,         # disable 2x requirement for launch to catch more early

    # ---------- RugCheck Gate ----------
    # Modes: "off" | "lenient" | "strict"
    "rugcheck_mode": "lenient",
    "rugcheck_score_min": 60,          # pass threshold (0..100)
    "rugcheck_timeout_sec": 5,

    # ---------- Social Snipe (X/Twitter) ----------
    "twitter_enabled": False,
    "twitter_bearer": "",
    "twitter_accounts": [],
    "twitter_positive_keywords": ["buy", "sending", "moon", "bullish", "ape", "pump"],
    "twitter_window_min": 10,
    "social_requires_rugcheck": True,
    "social_bypass_filters": True,

    # ---------- Wallet Tracking ----------
    "wallet_tracking_enabled": True,    # NEW: enable wallet tracking
    "min_wallet_roi": 35.0,            # NEW: minimum ROI to track wallets
    "tracked_wallets": [],             # NEW: list of high-performing wallets to track

    # ---------- Quantum Scorer (pre-pump logic) ----------
    "quantum_enabled": True,
    "quantum_threshold": 60,          # score ≥ threshold ⇒ candidate
    
    # ---------- Advanced AI Analysis ----------
    "ai_enabled": True,               # enable advanced AI analysis
    "ai_confidence_threshold": 70,    # minimum AI confidence score (0-100)
    "ai_require_pattern": False,      # require pattern recognition
    "ai_require_volume_spike": False, # require volume spike detection
    "ai_whale_bonus": 15,            # bonus points for whale activity
    "ai_momentum_weight": 0.3,       # weight for momentum in AI score
    # Small-cap preference band (score is best inside/near this range)
    "q_cap_lo": 10_000,               # 10k
    "q_cap_hi": 5_000_000,            # 5m
    # Weights (relative)
    "q_weight_cap": 30,               # favor smaller caps within [q_cap_lo, q_cap_hi]
    "q_weight_age": 15,               # newer pairs score higher
    "q_weight_buys": 20,              # more buys in m5
    "q_weight_buysell": 15,           # buys / (sells+1)
    "q_weight_velocity": 15,          # (vol_window / liquidity)
    "q_weight_venue": 3,              # raydium/pump/meteora/bunk bonus
    "q_weight_social": 2,             # social hint bonus
    # Bounds / normalizers
    "q_max_age_min": 60,              # 0m→best, 60m→worst
    "q_max_buys_m5": 50,              # clamp buys
    "q_max_buysell": 5.0,             # clamp buy:sell ratio
    "q_max_vel": 1.0,                 # clamp vol/liquidity
    # RugCheck behavior in quantum path:
    "q_require_rugcheck_strict": False,  # if True, quantum path requires rugcheck pass even if global is lenient

    # ---------- 2× Predictor (momentum gate) ----------
    "use_two_x_predict": True,            # global toggle
    "two_x_minutes": 2,                   # horizon minutes for doubling
    "two_x_alpha": 0.8,                   # loosen/tighten requirement (0.8 = slightly conservative)
    "momentum_window_sec": 60,            # compare price vs ~N seconds ago
    "momentum_min_dt_sec": 30,            # require at least this much time between anchor & now
    "quantum_require_2x": True,           # require 2x for quantum path
    "base_require_2x": False,             # require 2x for base filter path

    # ---------- Sizing & risk (paper) ----------
    "mode": "amount",          # "amount" (fixed SOL) or "percent" (of balance)
    "per_trade_sol": 2.5,
    "percent": 0.12,
    "take_profit": 0.60,
    "stop_loss": 0.16,
    "trail": 0.05,

    # ---------- Limits & timing ----------
    "max_positions": 6,
    "buy_cooldown_sec": 120,
    "global_cooldown_sec": 15,
    "min_hold_sec": 5,         # minimum hold time is 5 seconds

    # ---------- Loop timing ----------
    "poll_sec": 5,             # very fast polling to catch newborns quickly
    "social_poll_sec": 15,

    # ---------- Network backoff (Dexscreener) ----------
    "min_backoff": 2,
    "max_backoff": 60,

    # ---------- Debug ----------
    "debug_buys": True,                # send reasons when a buy is skipped (rate-limited)
    "debug_max_msgs_per_cycle": 5
}

DEFAULT_STATE = {
    "balance_sol": 75.0,
    "realized_pnl_sol": 0.0,
    "open_positions": {},      # key: pairAddress -> position dict
    "last_buy_time": 0,
    "last_global": 0,
    "telegram_offset": 0,
    "paused": False,
    "backoff": 0,
    "_diag": {},
    "_seen_social_mints": {},  # mint -> last_seen_ts (for quantum social bonus)
    "_sold_mints": {},         # NEW: mint -> sell_timestamp (to prevent re-buying)
    "_tracked_wallet_activity": {},  # NEW: wallet -> activity data
    # momentum anchors (in-memory; not persisted to disk intentionally)
}

def load_json(path: str, fallback: dict) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged = fallback.copy()
                    merged.update(data)
                    return merged
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
    return fallback.copy()

def save_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[WARN] Failed to save {path}: {e}")

settings = load_json(SETTINGS_FILE, DEFAULT_SETTINGS)
state    = load_json(STATE_FILE,    DEFAULT_STATE)

# in-memory momentum anchors (pair_addr -> {"t": ts, "p": price})
_px_anchors: Dict[str, Dict[str, float]] = {}

# Advanced AI Analysis Storage
_price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # Store price history
_volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))   # Store volume history
_whale_activity: Dict[str, List[Dict]] = defaultdict(list)                  # Track large transactions
_pattern_cache: Dict[str, Dict] = {}                                        # Cache pattern analysis

# =========================
# === Helpers ============
# =========================
def now() -> float: return time.time()
def clamp(v, lo, hi): return max(lo, min(hi, v))

# =========================
# === NO REBUY SYSTEM ====
# =========================
def is_recently_sold(mint: str) -> bool:
    """Check if a mint was recently sold and is in cooldown period"""
    if not settings.get("enable_no_rebuy", True):
        return False
    
    if not mint:
        return False
    
    sold_mints = state.get("_sold_mints", {})
    if mint not in sold_mints:
        return False
    
    sell_time = sold_mints[mint]
    cooldown_sec = settings.get("no_rebuy_after_sell_sec", 3600)
    
    return (now() - sell_time) < cooldown_sec

def record_sold_mint(mint: str) -> None:
    """Record that a mint was sold to prevent re-buying"""
    if not mint or not settings.get("enable_no_rebuy", True):
        return
    
    state.setdefault("_sold_mints", {})[mint] = now()
    save_json(STATE_FILE, state)

def cleanup_expired_sold_mints() -> None:
    """Remove expired entries from sold mints tracking"""
    if not settings.get("enable_no_rebuy", True):
        return
    
    cooldown_sec = settings.get("no_rebuy_after_sell_sec", 3600)
    current_time = now()
    sold_mints = state.get("_sold_mints", {})
    
    expired_mints = [mint for mint, sell_time in sold_mints.items() 
                     if (current_time - sell_time) >= cooldown_sec]
    
    for mint in expired_mints:
        del sold_mints[mint]
    
    if expired_mints:
        save_json(STATE_FILE, state)
        print(f"[INFO] Cleaned up {len(expired_mints)} expired sold mints")

# =========================
# === Wallet Tracking ====
# =========================
def track_wallet_activity(wallet: str, roi: float, token_symbol: str) -> None:
    """Track successful wallet trades with ROI >= threshold"""
    min_roi = settings.get("min_wallet_roi", 35.0)
    if roi < min_roi:
        return
    
    activity = state.setdefault("_tracked_wallet_activity", {})
    wallet_data = activity.setdefault(wallet, {
        "total_trades": 0,
        "avg_roi": 0.0,
        "last_activity": 0,
        "tokens": []
    })
    
    # Update wallet stats
    old_avg = wallet_data["avg_roi"]
    old_count = wallet_data["total_trades"]
    
    wallet_data["total_trades"] += 1
    wallet_data["avg_roi"] = ((old_avg * old_count) + roi) / wallet_data["total_trades"]
    wallet_data["last_activity"] = now()
    wallet_data["tokens"].append({
        "symbol": token_symbol,
        "roi": roi,
        "timestamp": now()
    })
    
    # Keep only last 20 tokens
    wallet_data["tokens"] = wallet_data["tokens"][-20:]
    
    # Add to tracked wallets if ROI is good enough and not already tracked
    tracked = settings.get("tracked_wallets", [])
    if wallet not in tracked and wallet_data["avg_roi"] >= min_roi:
        tracked.append(wallet)
        settings["tracked_wallets"] = tracked
        save_json(SETTINGS_FILE, settings)
        tg_send(f"📈 New wallet tracked: {wallet[:8]}... (Avg ROI: {wallet_data['avg_roi']:.1f}%)")
    
    save_json(STATE_FILE, state)

def get_wallet_signals() -> List[Dict[str, Any]]:
    """Get trading signals from tracked high-ROI wallets"""
    if not settings.get("wallet_tracking_enabled", True):
        return []
    
    signals = []
    tracked_wallets = settings.get("tracked_wallets", [])
    
    # This would integrate with Solana transaction monitoring
    # For now, return empty list - would need Solana RPC integration
    return signals

# =========================
# === Advanced AI Analysis
# =========================

class AdvancedAnalyzer:
    """Advanced AI-powered trading analysis with multiple sophisticated indicators"""
    
    @staticmethod
    def update_price_history(pair_addr: str, price: float, volume: float, timestamp: float = None):
        """Update price and volume history for analysis"""
        if timestamp is None:
            timestamp = now()
        
        _price_history[pair_addr].append({
            'price': price,
            'timestamp': timestamp,
            'volume': volume
        })
        _volume_history[pair_addr].append(volume)
    
    @staticmethod
    def calculate_rsi(pair_addr: str, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        history = _price_history[pair_addr]
        if len(history) < period + 1:
            return None
        
        prices = [h['price'] for h in list(history)[-period-1:]]
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        
        if not gains or not losses:
            return 50.0
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi    
    @staticmethod
    def detect_volume_spike(pair_addr: str, spike_threshold: float = 2.5) -> Tuple[bool, float]:
        """Detect unusual volume spikes"""
        volumes = list(_volume_history[pair_addr])
        if len(volumes) < 10:
            return False, 0.0
        
        recent_vol = volumes[-1]
        avg_vol = statistics.mean(volumes[-10:-1]) if len(volumes) > 1 else recent_vol
        
        if avg_vol == 0:
            return False, 0.0
        
        spike_ratio = recent_vol / avg_vol
        return spike_ratio >= spike_threshold, spike_ratio
    
    @staticmethod
    def calculate_vwap(pair_addr: str, periods: int = 20) -> Optional[float]:
        """Calculate Volume Weighted Average Price"""
        history = list(_price_history[pair_addr])
        if len(history) < periods:
            return None
        
        recent_data = history[-periods:]
        total_volume = sum(h['volume'] for h in recent_data)
        
        if total_volume == 0:
            return None
        
        vwap = sum(h['price'] * h['volume'] for h in recent_data) / total_volume
        return vwap
    
    @staticmethod
    def detect_breakout_pattern(pair_addr: str) -> Tuple[bool, str, float]:
        """Detect chart breakout patterns"""
        history = list(_price_history[pair_addr])
        if len(history) < 20:
            return False, "insufficient_data", 0.0
        
        prices = [h['price'] for h in history[-20:]]
        volumes = [h['volume'] for h in history[-20:]]
        
        # Detect ascending triangle pattern
        highs = []
        lows = []
        for i in range(2, len(prices)-2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append(prices[i])
        
        if len(highs) >= 3 and len(lows) >= 2:
            # Check if highs are relatively flat (resistance level)
            high_variance = statistics.variance(highs[-3:]) if len(highs) >= 3 else float('inf')
            high_avg = statistics.mean(highs[-3:])
            
            # Check if lows are ascending (higher lows)
            if len(lows) >= 2 and lows[-1] > lows[-2]:
                current_price = prices[-1]
                if current_price > high_avg * 0.98 and high_variance < (high_avg * 0.02) ** 2:
                    confidence = min(0.9, (current_price / high_avg) * 0.8)
                    return True, "ascending_triangle_breakout", confidence
        
        # Detect volume breakout
        has_volume_spike, spike_ratio = AdvancedAnalyzer.detect_volume_spike(pair_addr, 2.0)
        if has_volume_spike:
            price_change = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
            if price_change > 0.05:  # 5% price increase with volume spike
                confidence = min(0.8, spike_ratio / 5.0)
                return True, "volume_breakout", confidence
        
        return False, "no_pattern", 0.0
    
    @staticmethod
    def calculate_momentum_score(pair_addr: str) -> float:
        """Calculate comprehensive momentum score (0-100)"""
        history = list(_price_history[pair_addr])
        if len(history) < 10:
            return 50.0
        
        scores = []
        
        # Price momentum (30% weight)
        prices = [h['price'] for h in history[-10:]]
        if len(prices) >= 5:
            short_ma = statistics.mean(prices[-3:])
            long_ma = statistics.mean(prices[-10:])
            price_momentum = min(100, max(0, ((short_ma / long_ma - 1) * 500) + 50))
            scores.append((price_momentum, 0.3))
        
        # RSI (20% weight)
        rsi = AdvancedAnalyzer.calculate_rsi(pair_addr)
        if rsi is not None:
            # Convert RSI to momentum score (RSI > 50 = bullish momentum)
            rsi_momentum = min(100, max(0, rsi))
            scores.append((rsi_momentum, 0.2))
        
        # Volume trend (25% weight)
        volumes = list(_volume_history[pair_addr])
        if len(volumes) >= 5:
            recent_vol_avg = statistics.mean(volumes[-3:])
            older_vol_avg = statistics.mean(volumes[-10:-3]) if len(volumes) >= 10 else recent_vol_avg
            if older_vol_avg > 0:
                vol_momentum = min(100, max(0, ((recent_vol_avg / older_vol_avg - 1) * 200) + 50))
                scores.append((vol_momentum, 0.25))
        
        # Pattern strength (25% weight)
        has_pattern, pattern_type, confidence = AdvancedAnalyzer.detect_breakout_pattern(pair_addr)
        if has_pattern:
            pattern_momentum = 70 + (confidence * 30)  # 70-100 based on confidence
        else:
            pattern_momentum = 40  # Neutral
        scores.append((pattern_momentum, 0.25))
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weight for _, weight in scores)
            weighted_sum = sum(score * weight for score, weight in scores)
            return weighted_sum / total_weight
        
        return 50.0
    
    @staticmethod
    def detect_whale_activity(pair_addr: str, txn_data: Dict) -> bool:
        """Detect large whale transactions"""
        if not txn_data:
            return False
        
        # Get transaction counts
        buys = safe_float(txn_data.get("buys", 0))
        sells = safe_float(txn_data.get("sells", 0))
        
        # Look for signs of whale activity:
        # 1. Low transaction count but high volume (large individual transactions)
        # 2. Sudden increase in buy pressure
        
        total_txns = buys + sells
        if total_txns < 5 and total_txns > 0:  # Few but potentially large transactions
            # Check if volume is disproportionately high for transaction count
            recent_volume = list(_volume_history[pair_addr])[-1] if _volume_history[pair_addr] else 0
            if recent_volume > 0 and total_txns > 0:
                avg_txn_size = recent_volume / total_txns
                # If average transaction size is very large, might be whale activity
                if avg_txn_size > 10000:  # $10k+ average transaction
                    return True
        
        # Strong buy pressure with reasonable volume
        if buys > 0 and sells >= 0:
            buy_ratio = buys / (buys + sells + 1)
            if buy_ratio > 0.8 and total_txns >= 3:  # 80%+ buys with decent activity
                return True
        
        return False
    
    @staticmethod
    def calculate_ai_confidence(pair_addr: str, pair_data: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate AI confidence score with breakdown"""
        
        # Update data first
        price = _read_price(pair_data)
        volume = safe_float((pair_data.get("volume") or {}).get("m5", 0))
        AdvancedAnalyzer.update_price_history(pair_addr, price, volume)
        
        scores = {}
        
        # Momentum analysis (25%)
        momentum_score = AdvancedAnalyzer.calculate_momentum_score(pair_addr)
        scores['momentum'] = momentum_score
        
        # Volume analysis (20%)
        has_spike, spike_ratio = AdvancedAnalyzer.detect_volume_spike(pair_addr)
        volume_score = min(100, 50 + (spike_ratio * 10)) if has_spike else 40
        scores['volume'] = volume_score
        
        # Pattern recognition (20%)
        has_pattern, pattern_type, pattern_conf = AdvancedAnalyzer.detect_breakout_pattern(pair_addr)
        pattern_score = (70 + pattern_conf * 30) if has_pattern else 35
        scores['pattern'] = pattern_score
        
        # Whale activity (15%)
        txn_data = (pair_data.get("txns") or {}).get("m5") or {}
        has_whales = AdvancedAnalyzer.detect_whale_activity(pair_addr, txn_data)
        whale_score = 75 if has_whales else 45
        scores['whale'] = whale_score
        
        # Market structure (10%)
        rsi = AdvancedAnalyzer.calculate_rsi(pair_addr)
        if rsi is not None:
            # Optimal RSI for entry: 45-65 (not oversold, not overbought)
            if 45 <= rsi <= 65:
                structure_score = 80
            elif 30 <= rsi <= 75:
                structure_score = 60
            else:
                structure_score = 30
        else:
            structure_score = 50
        scores['structure'] = structure_score
        
        # Price action (10%)
        history = list(_price_history[pair_addr])
        if len(history) >= 5:
            recent_prices = [h['price'] for h in history[-5:]]
            price_trend = (recent_prices[-1] / recent_prices[0] - 1) * 100
            if price_trend > 5:  # Strong uptrend
                action_score = 80
            elif price_trend > 0:  # Mild uptrend
                action_score = 65
            elif price_trend > -5:  # Sideways
                action_score = 50
            else:  # Downtrend
                action_score = 25
        else:
            action_score = 50
        scores['price_action'] = action_score
        
        # Calculate weighted final score
        weights = {
            'momentum': 0.25,
            'volume': 0.20,
            'pattern': 0.20,
            'whale': 0.15,
            'structure': 0.10,
            'price_action': 0.10
        }
        
        final_score = sum(scores[key] * weights[key] for key in scores.keys())
        
        return final_score, scores

def safe_float(x, default=0.0) -> float:
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().replace(",", "")
        if s.lower() in ("nan", "inf", "-inf", "none", ""): return default
        return float(s)
    except: return default

# =========================
# === Dexscreener Fetch ==
# =========================
DEX_URLS = [
    "https://api.dexscreener.com/latest/dex/search?q=SOL",     # Search for SOL pairs
    "https://api.dexscreener.com/latest/dex/search?q=USDC",    # Search for USDC pairs  
    "https://api.dexscreener.com/latest/dex/search?q=pump",    # Search for pump.fun pairs
    "https://api.dexscreener.com/latest/dex/search?q=raydium", # Search for Raydium pairs
]

def fetch_pairs() -> List[Dict[str, Any]]:
    """Fetch pairs via search; try multiple endpoints; exponential backoff on transient errors."""
    all_pairs = []
    
    for url in DEX_URLS:
        try:
            print(f"[INFO] Trying: {url}")
            r = requests.get(url, timeout=15)
            print(f"[INFO] Response: {r.status_code}")
            
            if r.status_code == 200:
                data = r.json() or {}
                pairs = data.get("pairs") or []
                
                if isinstance(pairs, list) and pairs:
                    print(f"[INFO] Got {len(pairs)} pairs from {url}")
                    # Filter for Solana pairs only
                    solana_pairs = [p for p in pairs if str(p.get("chainId", "")).lower() == "solana"]
                    print(f"[INFO] {len(solana_pairs)} Solana pairs")
                    all_pairs.extend(solana_pairs)
                else:
                    print(f"[WARN] No pairs in response from {url}")
                    
            elif r.status_code == 404:
                print(f"[WARN] Endpoint not found: {url}")
                continue
            elif r.status_code in (429, 500, 502, 503, 504):
                print(f"[WARN] Server error {r.status_code}, will backoff")
                break
            else:
                print(f"[WARN] Dexscreener HTTP {r.status_code} on {url}")
                continue
                
        except Exception as e:
            print(f"[WARN] Dexscreener error on {url}: {e}")
            continue
    
    if all_pairs:
        # Remove duplicates by pairAddress
        seen = set()
        unique_pairs = []
        for p in all_pairs:
            addr = p.get("pairAddress")
            if addr and addr not in seen:
                seen.add(addr)
                unique_pairs.append(p)
        
        print(f"[INFO] Total unique Solana pairs: {len(unique_pairs)}")
        state["backoff"] = 0
        save_json(STATE_FILE, state)
        return unique_pairs
    
    # Apply backoff if no data retrieved
    back = state.get("backoff", 0)
    back = settings["min_backoff"] if back == 0 else min(int(back * 2), int(settings["max_backoff"]))
    state["backoff"] = back
    save_json(STATE_FILE, state)
    print(f"[WARN] No pairs fetched, applying {back}s backoff")
    return []

def _pair_id(p: dict) -> str:
    return p.get("pairAddress") or p.get("url") or f"addr:{int(now()*1000)}"

def _pair_symbol(p: dict) -> str:
    base = ((p.get("baseToken") or {}).get("symbol")) or "?"
    quote= ((p.get("quoteToken") or {}).get("symbol")) or "?"
    return f"{base}/{quote}"

def _mint_from_pair(p: dict) -> str:
    return ((p.get("baseToken") or {}).get("address")) or ""

def _read_price(p: dict) -> float:
    """Return a usable price: prefer USD; else priceNative (non-USD), else 0."""
    pu = safe_float(p.get("priceUsd"), 0.0)
    if pu > 0: return pu
    pn = safe_float(p.get("priceNative"), 0.0)
    return pn  # may be non-USD, but consistent for pct PnL

def current_price_map(pairs: List[Dict[str, Any]]) -> Dict[str, float]:
    out = {}
    for p in pairs:
        addr = p.get("pairAddress") or ""
        if addr:
            price = _read_price(p)
            out[addr] = price
    return out

def _vol_window_value(p: dict, minutes: int) -> float:
    """Volume proxy for N minutes: max(m5, h1 * N/60, m1*minutes)."""
    vol = p.get("volume") or {}
    m5 = safe_float(vol.get("m5"), 0.0)
    h1 = safe_float(vol.get("h1"), 0.0)
    m1 = safe_float(vol.get("m1"), 0.0)  # 1-minute volume for very new tokens
    
    # For very new tokens, use 1-minute volume extrapolated
    est_from_h1 = h1 * (minutes / 60.0)
    est_from_m1 = m1 * minutes
    
    if minutes <= 5:
        # For short windows, prefer m5 or m1 extrapolation
        return max(m5, est_from_m1)
    else:
        # For longer windows, use best available estimate
        return max(m5, est_from_h1, est_from_m1)

# =========================
# === RugCheck (optional)
# =========================
def rugcheck_pass(mint: str) -> Tuple[bool, str]:
    mode = settings.get("rugcheck_mode", "off").lower()
    if mode == "off": return True, "rugcheck: off"
    timeout = int(settings.get("rugcheck_timeout_sec", 5))
    thresh = int(settings.get("rugcheck_score_min", 60))
    candidates = [
        f"https://api.rugcheck.xyz/v1/tokens/{mint}",
        f"https://api.rugcheck.xyz/v1/score/{mint}",
    ]
    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                data = r.json() or {}
                score = None
                if isinstance(data, dict):
                    score = data.get("score") or data.get("riskScore") or data.get("overallScore")
                    if isinstance(score, dict):
                        score = score.get("value") or score.get("score")
                score = safe_float(score, -1)
                if score >= 0:
                    return (score >= thresh), f"rugcheck score={score:.0f} (min {thresh})"
        except Exception as e:
            last_err = str(e)
            continue
    if mode == "lenient":
        return True, f"rugcheck: API unavailable ({last_err}) → allowed (lenient)"
    else:
        return False, f"rugcheck: API unavailable ({last_err}) → blocked (strict)"

# =========================
# === Meme / Solana focus
# =========================
def is_solana_pair(p: dict) -> bool:
    return (str(p.get("chainId") or "").lower() == "solana")

def is_meme_candidate(p: dict) -> Tuple[bool, str]:
    """
    Strict meme token filtering for new SPL tokens:
      • must be Solana (if solana_only)
      • quote must be in whitelist (SOL/USDC/USDT) - these are the "major" tokens
      • base must NOT be in majors blacklist - the base should be the NEW meme token
      • base should NOT be any major token (prevent SOL/ETH, SOL/BTC type pairs)
      • prefer tokens with reasonable market caps for memes (10k-5M range)
    """
    if settings.get("solana_only", True) and not is_solana_pair(p):
        return False, "not solana"
    if not settings.get("meme_only", True):
        return True, "meme_only disabled"

    bt = (p.get("baseToken") or {})
    qt = (p.get("quoteToken") or {})
    b_sym = (bt.get("symbol") or "").upper()
    q_sym = (qt.get("symbol") or "").upper()

    # Quote must be a major stable token (SOL, USDC, USDT)
    if q_sym not in settings.get("meme_quote_whitelist", ["SOL","USDC","USDT"]):
        return False, f"bad quote {q_sym}"
    
    # Base must NOT be any major token (this prevents SOL/ETH, SOL/BTC, etc.)
    major_tokens = settings.get("meme_base_blacklist", []) + ["ETH", "BTC", "WETH", "WBTC", "MATIC", "AVAX", "FTM", "ATOM", "DOT", "ADA", "LINK", "UNI", "AAVE"]
    if b_sym in [token.upper() for token in major_tokens]:
        return False, f"blacklisted major base {b_sym}"
    
    # Additional checks for meme characteristics
    # Check if base token has a reasonable name length (meme tokens usually have 3-10 char symbols)
    if len(b_sym) < 2 or len(b_sym) > 12:
        return False, f"unusual symbol length {b_sym}"
    
    # Check market cap is in meme range if available
    fdv = safe_float(p.get("fdv"), -1)
    if fdv <= 0:
        fdv = safe_float(p.get("marketCap"), -1)
    
    if fdv > 0:
        # Very permissive market cap range for meme tokens (allow micro caps)
        if fdv < 100 or fdv > 100_000_000:  # $100 to $100M range
            return False, f"market cap {fdv:,.0f} outside meme range"
    
    return True, "meme candidate ok"

# =========================
# === Base Filter Logic ==
# =========================
def _age_minutes(p: dict) -> float:
    # Try multiple timestamp fields that Dexscreener might use
    ts_ms = p.get("pairCreatedAt") or p.get("createdAt") or p.get("firstSeenAt")
    
    if not ts_ms:
        # If no timestamp, assume it's very old to be safe
        return 1e9
    
    try:
        # Handle both milliseconds and seconds timestamps
        ts_float = float(ts_ms)
        current_ms = now() * 1000
        
        # If timestamp looks like seconds (< year 2030 in seconds), convert to ms
        if ts_float < 1900000000000:  # Less than year 2030 in milliseconds
            ts_float *= 1000
        
        age_ms = max(0.0, current_ms - ts_float)
        age_minutes = age_ms / 60000.0
        
        return age_minutes
    except (ValueError, TypeError):
        return 1e9

def _base_conditions(p: dict, vol_win: int) -> Tuple[bool, bool, bool]:
    """Return (liq_ok, vol_ok, cap_ok) for a pair under current settings."""
    min_liq   = max(0, int(safe_float(settings["min_liq"], 0)))
    min_vol   = max(0, int(safe_float(settings["min_vol_usd"], 0)))
    cap_lo, cap_hi = settings.get("cap_range", [0, 0])
    cap_lo = max(0, int(safe_float(cap_lo, 0)))
    cap_hi = max(cap_lo, int(safe_float(cap_hi, cap_lo)))
    enforce_cap = bool(settings.get("enforce_cap", True))
    pass_if_unknown = bool(settings.get("pass_if_cap_unknown", True))

    # Check if this is a very new token (≤ 10 minutes old)
    age_min = _age_minutes(p)
    is_very_new = age_min <= 10
    
    liq = safe_float((p.get("liquidity") or {}).get("usd"), 0.0)
    
    # More lenient liquidity requirements for very new tokens
    if is_very_new and liq >= (min_liq * 0.3):  # 30% of normal requirement
        liq_ok = True
    else:
        liq_ok = (liq >= min_liq)

    v = _vol_window_value(p, vol_win)
    
    # More lenient volume requirements for very new tokens
    if is_very_new and v >= (min_vol * 0.2):  # 20% of normal requirement
        vol_ok = True
    else:
        vol_ok = (v >= min_vol)

    cap_ok = True  # Default to true for new tokens
    if enforce_cap:
        fdv = safe_float(p.get("fdv"), -1)
        if fdv <= 0:
            fdv = safe_float(p.get("marketCap"), -1)
        if fdv > 0:
            cap_ok = (cap_lo <= fdv <= cap_hi)
        else:
            cap_ok = pass_if_unknown

    return liq_ok, vol_ok, cap_ok# Trading functions with no-rebuy system

def filter_pairs_logic(pairs: List[Dict[str, Any]], for_diag: bool=False) -> List[Dict[str, Any]]:
    """
    Evaluate pairs under configurable logic:
      - OR: any(base) true
      - AND: all(base) true
      - MAJORITY: at least settings['min_true'] base conditions true
      - HYBRID: if age<=hybrid_new_age_min ⇒ require liq AND vol; else MAJORITY(min_true)
    Always applies the simplified meme/solana gate first.
    """
    vol_win = max(5, min(60, int(safe_float(settings.get("vol_window_min", 10), 10))))
    mode = str(settings.get("filter_mode", "OR")).upper()
    min_true = int(settings.get("min_true", 2))
    hybrid_cutoff = float(settings.get("hybrid_new_age_min", 3))

    passed: List[Dict[str, Any]] = []
    pass_liq = pass_vol = pass_cap = 0

    for p in pairs:
        mem_ok, _ = is_meme_candidate(p)
        if not mem_ok:
            continue

        liq_ok, vol_ok, cap_ok = _base_conditions(p, vol_win)
        trues = (1 if liq_ok else 0) + (1 if vol_ok else 0) + (1 if cap_ok else 0)
        age_m = _age_minutes(p)

        allow = False
        if mode == "AND":
            allow = liq_ok and vol_ok and cap_ok
        elif mode == "MAJORITY":
            allow = (trues >= max(1, min_true))
        elif mode == "HYBRID":
            if age_m <= hybrid_cutoff:
                allow = (liq_ok and vol_ok)          # stricter when very new
            else:
                allow = (trues >= max(1, min_true))  # fallback majority
        else:  # OR default
            allow = (liq_ok or vol_ok or cap_ok)

        if allow:
            passed.append(p)
            pass_liq += 1 if liq_ok else 0
            pass_vol += 1 if vol_ok else 0
            pass_cap += 1 if cap_ok else 0

    # Rank by recent activity
    def recent_score(q: dict) -> float:
        tx = (q.get("txns") or {}).get("m5") or {}
        return safe_float(tx.get("buys"), 0) + safe_float(tx.get("sells"), 0) + _vol_window_value(q, vol_win)

    passed.sort(key=recent_score, reverse=True)

    if for_diag:
        state["_diag"] = {
            "total": len(pairs),
            "pass_liq": pass_liq, "pass_vol": pass_vol, "pass_cap": pass_cap,
            "mode": mode, "min_true": min_true, "hybrid_cutoff_min": hybrid_cutoff,
            "params": {
                "min_liq": int(settings["min_liq"]),
                "min_vol_usd": int(settings["min_vol_usd"]),
                "vol_window_min": vol_win,
                "cap": settings.get("cap_range", [0,0]),
                "enforce_cap": bool(settings.get("enforce_cap", True)),
                "pass_if_cap_unknown": bool(settings.get("pass_if_cap_unknown", True)),
            }
        }
        save_json(STATE_FILE, state)

    return passed

# =========================
# === 2× Predictor =======
# =========================
def _momentum_update_and_rate(pair_addr: str, current_price: float) -> Tuple[bool, float, float]:
    """
    Maintain a price anchor ~momentum_window_sec ago and estimate growth rate per minute.
    Returns (ok, rate_per_min, dt_sec). ok=False if not enough time/datum yet.
    """
    if current_price <= 0: return False, 0.0, 0.0
    tnow = now()
    win = float(settings.get("momentum_window_sec", 60))
    min_dt = float(settings.get("momentum_min_dt_sec", 30))
    rec = _px_anchors.get(pair_addr)

    if rec is None:
        _px_anchors[pair_addr] = {"t": tnow, "p": current_price}
        return False, 0.0, 0.0

    dt = tnow - float(rec["t"])
    if dt < min_dt:
        # too soon; keep the old anchor
        return False, 0.0, dt

    if dt < win:
        # not long enough for a stable estimate; do not reset anchor yet
        return False, 0.0, dt

    # compute rate and reset anchor to now
    try:
        r = math.log(max(1e-16, current_price) / max(1e-16, float(rec["p"]))) / (dt / 60.0)  # per-minute log-rate
    except Exception:
        r = 0.0
    _px_anchors[pair_addr] = {"t": tnow, "p": current_price}
    return True, r, dt

def two_x_predict(pair_addr: str, current_price: float) -> bool:
    """
    True iff short-term momentum suggests we can plausibly 2× within target minutes.
    Condition: r_per_min * M >= alpha * ln(2)
    """
    if not settings.get("use_two_x_predict", True):
        return True
    ok, r_per_min, dt = _momentum_update_and_rate(pair_addr, current_price)
    if not ok:
        return False
    M = float(settings.get("two_x_minutes", 2))
    alpha = float(settings.get("two_x_alpha", 0.8))
    return (r_per_min * M) >= (alpha * math.log(2.0))

# =========================
# === Launch Snipe =======
# =========================
def dex_allowed(p: dict) -> bool:
    allow = [s.lower() for s in settings.get("allowed_dex_ids", [])]
    dexid = (p.get("dexId") or "").lower()
    url = (p.get("url") or "").lower()
    return any(k in dexid or k in url for k in allow)

def is_new_pair(p: dict) -> bool:
    max_age_min = int(settings.get("max_pair_age_min", 30))
    
    # Try multiple timestamp fields
    ts_ms = p.get("pairCreatedAt") or p.get("createdAt") or p.get("firstSeenAt")
    
    if not ts_ms:
        # If no timestamp available, be permissive and assume it could be new
        # This helps catch tokens where timestamp data is missing
        return True
    
    try:
        age_min = _age_minutes(p)
        is_new = age_min <= max_age_min
        
        # Debug log for troubleshooting
        if settings.get("debug_buys", False):
            print(f"[DEBUG] {_pair_symbol(p)}: age={age_min:.1f}m, max={max_age_min}m, new={is_new}")
        
        return is_new
    except Exception as e:
        print(f"[WARN] Age calculation error for {_pair_symbol(p)}: {e}")
        return True  # Be permissive on errors

def basic_launch_checks(p: dict) -> Tuple[bool, str]:
    liq = safe_float((p.get("liquidity") or {}).get("usd"), 0.0)
    buys = safe_float(((p.get("txns") or {}).get("m5") or {}).get("buys"), 0.0)
    sells = safe_float(((p.get("txns") or {}).get("m5") or {}).get("sells"), 0.0)
    
    # Very lenient liquidity check 
    min_liq = float(settings.get("launch_liq_min", 300))
    if liq < min_liq:
        return False, f"launch: liq {liq:.0f} < min {min_liq:.0f}"
    
    # Very lenient buying activity check
    min_buys = float(settings.get("launch_min_buys_m5", 1))
    if buys < min_buys:
        return False, f"launch: buys m5 {buys:.0f} < min {min_buys:.0f}"
    
    # Don't be too strict on buy/sell ratio for very new tokens
    total_txns = buys + sells
    if total_txns >= 5 and sells > buys * 2:  # Only reject if way more sells than buys
        return False, f"launch: too many sells {buys:.0f}/{sells:.0f}"
    
    return True, "launch: basics ok"

def _maybe_rugcheck(p: dict) -> Tuple[bool, str]:
    mint = _mint_from_pair(p) or ""
    if settings.get("rugcheck_mode","off") == "off" or not mint:
        return True, "rugcheck: off or no mint"
    return rugcheck_pass(mint)

def should_launch_snipe(p: dict) -> Tuple[bool, str]:
    if not settings.get("launch_enabled", True):
        return False, "launch: disabled"
    
    # meme/solana gate FIRST (most important filter)
    m_ok, why_m = is_meme_candidate(p)
    if not m_ok:
        return False, f"launch: {why_m}"
    
    if not dex_allowed(p):
        return False, "launch: dex not allowed"
    
    # Check if pair is new enough (but be permissive if no timestamp)
    if not is_new_pair(p):
        # Check if we have timestamp data
        ts_ms = p.get("pairCreatedAt") or p.get("createdAt") or p.get("firstSeenAt")
        if ts_ms:  # Only reject if we have timestamp and it's actually old
            return False, "launch: too old"
        # If no timestamp, continue (could be new)
    
    ok, why = basic_launch_checks(p)
    if not ok: return False, why
    
    rc_ok, rc_note = _maybe_rugcheck(p)
    if not rc_ok: return False, rc_note

    # Optional 2x predictor on launch path
    if settings.get("launch_require_2x", True):
        addr = p.get("pairAddress") or ""
        pr = _read_price(p)
        if not two_x_predict(addr, pr):
            return False, "launch: 2x predictor not met yet"

    return True, f"launch ok + {rc_note}"

# =========================
# === Paper Trading ======
# =========================
def can_buy() -> bool:
    if state.get("paused", False): return False
    if len(state["open_positions"]) >= int(settings["max_positions"]): return False
    t = now()
    if t - float(state["last_buy_time"]) < float(settings["buy_cooldown_sec"]): return False
    if t - float(state["last_global"]) < float(settings["global_cooldown_sec"]): return False
    return True

def position_value_change_pct(current_price: float, entry_price: float, highest_price: float) -> Tuple[float, float]:
    if entry_price <= 0: return 0.0, 0.0
    pct_from_entry = (current_price - entry_price) / entry_price
    dd_from_high = (current_price - highest_price) / highest_price if highest_price > 0 else 0.0
    return pct_from_entry, dd_from_high

def _already_holding_symbol(symbol: str) -> bool:
    for pos in state["open_positions"].values():
        if pos.get("symbol") == symbol:
            return True
    return False

def _invest_amount() -> float:
    if settings.get("mode") == "percent":
        return clamp(state["balance_sol"] * float(settings.get("percent", 0.1)), 0.0, state["balance_sol"])
    else:
        return clamp(float(settings.get("per_trade_sol", 1.0)), 0.0, state["balance_sol"])

def _why_not_buy(pair: dict, reason_prefix: str = "") -> str:
    if state.get("paused", False): return "paused"
    if len(state["open_positions"]) >= int(settings["max_positions"]): return "max_positions"
    t = now()
    if t - float(state["last_buy_time"]) < float(settings["buy_cooldown_sec"]): return "buy_cooldown"
    if t - float(state["last_global"]) < float(settings["global_cooldown_sec"]): return "global_cooldown"
    price = _read_price(pair)
    if price <= 0: return "price_missing"
    invest = _invest_amount()
    if invest <= 0: return "no_balance"
    sym = _pair_symbol(pair)
    if _already_holding_symbol(sym): return "already_holding_symbol"
    
    # NEW: Check if this mint was recently sold
    mint = _mint_from_pair(pair)
    if mint and is_recently_sold(mint):
        cooldown_sec = settings.get("no_rebuy_after_sell_sec", 3600)
        return f"recently_sold_cooldown_{cooldown_sec}s"
    
    return reason_prefix or "unknown"

def try_buy(pair: Dict[str, Any], reason: str = "filter") -> bool:
    if not can_buy():
        if settings.get("debug_buys"): tg_send(f"SKIP (gate): {_why_not_buy(pair)}")
        return False

    pair_addr = _pair_id(pair)
    symbol = _pair_symbol(pair)
    mint = _mint_from_pair(pair)
    
    # NEW: Check if recently sold (prevent re-buying)
    if mint and is_recently_sold(mint):
        cooldown_sec = settings.get("no_rebuy_after_sell_sec", 3600)
        if settings.get("debug_buys"): 
            tg_send(f"SKIP ({symbol}): recently sold, cooldown {cooldown_sec//60}min remaining")
        return False
    
    if pair_addr in state["open_positions"] or _already_holding_symbol(symbol):
        if settings.get("debug_buys"): tg_send(f"SKIP ({symbol}): already holding")
        return False

    price = _read_price(pair)
    if price <= 0:
        if settings.get("debug_buys"): tg_send(f"SKIP ({symbol}): price missing")
        return False

    invest_sol = _invest_amount()
    if invest_sol <= 0.0:
        if settings.get("debug_buys"): tg_send(f"SKIP ({symbol}): invest=0 (balance {state['balance_sol']:.4f})")
        return False

    pos = {"symbol": symbol, "entry_price": price, "highest_price": price,
           "invest_sol": invest_sol, "opened_ts": now(), "mint": mint}  # NEW: store mint
    state["open_positions"][pair_addr] = pos
    state["balance_sol"] -= invest_sol
    state["last_buy_time"] = now()
    state["last_global"]   = now()
    save_json(STATE_FILE, state)

    tg_send(f"🟢 Bought {symbol} (paper) — {reason}\nEntry: {price:.10f}\nIn: {invest_sol:.4f} SOL\nOpen positions: {len(state['open_positions'])}/{settings['max_positions']}")
    return True

def try_sell(pair_addr: str, pos: dict, reason: str, current_price: float, pct: float) -> None:
    invest = float(pos.get("invest_sol", 0.0))
    pnl_sol = invest * pct
    state["realized_pnl_sol"] += pnl_sol
    state["balance_sol"] += (invest + pnl_sol)
    
    # NEW: Record sold mint to prevent re-buying
    mint = pos.get("mint")
    if mint:
        record_sold_mint(mint)
    
    if pair_addr in state["open_positions"]:
        del state["open_positions"][pair_addr]
    save_json(STATE_FILE, state)
    
    # Track wallet activity if this was a good trade
    if pct > 0.35:  # 35%+ ROI
        symbol = pos.get("symbol", "")
        # In a real implementation, you'd get the wallet from transaction data
        # For now, we'll just track the performance
        track_wallet_activity("sample_wallet", pct * 100, symbol)
    
    tg_send(f"🔴 Sold {pos.get('symbol','?')} (paper) — {reason}\nExit: {current_price:.10f}\nPnL: {pnl_sol:+.4f} SOL ({pct*100:+.2f}%)\nBalance: {state['balance_sol']:.4f} SOL | Realized: {state['realized_pnl_sol']:+.4f} SOL")

def manage_positions(prices_by_addr: Dict[str, float]) -> None:
    tp = float(settings["take_profit"])
    sl = float(settings["stop_loss"])
    trail = float(settings["trail"])
    min_hold = float(settings["min_hold_sec"])
    to_close = []

    for addr, pos in list(state["open_positions"].items()):
        cur = safe_float(prices_by_addr.get(addr), pos.get("entry_price", 0.0))
        if cur > pos["highest_price"]:
            pos["highest_price"] = cur

        pct, dd_from_high = position_value_change_pct(cur, pos["entry_price"], pos["highest_price"])
        held_ok = (now() - float(pos.get("opened_ts", now()))) >= min_hold

        reason = None
        
        # Traditional exit conditions
        if pct >= tp and held_ok: 
            reason = "take-profit"
        elif pct <= -sl and held_ok: 
            reason = "stop-loss"
        elif dd_from_high <= -trail and held_ok: 
            reason = "trailing stop"
        
        # AI-powered exit signals (if enabled and we have enough data)
        if not reason and settings.get("ai_enabled", True) and held_ok:
            try:
                # Get current pair data for this address
                pairs = fetch_pairs()
                current_pair = None
                for p in pairs:
                    if p.get("pairAddress") == addr:
                        current_pair = p
                        break
                
                if current_pair:
                    ai_score, ai_breakdown = AdvancedAnalyzer.calculate_ai_confidence(addr, current_pair)
                    
                    # AI Exit conditions:
                    # 1. Very low AI confidence (< 30) suggests exit
                    # 2. RSI overbought (> 80) with declining momentum
                    # 3. Volume drying up with pattern breakdown
                    
                    if ai_score < 30:
                        reason = f"AI-exit (confidence {ai_score:.0f}%)"
                    elif ai_breakdown.get('structure', 50) > 80 and ai_breakdown.get('momentum', 50) < 40:
                        reason = "AI-exit (overbought + weak momentum)"
                    elif ai_breakdown.get('volume', 50) < 25 and pct > 0.1:  # Volume dying, take some profit
                        reason = "AI-exit (volume decline)"
                        
            except Exception as e:
                # Don't let AI analysis errors affect normal operation
                pass

        if reason: 
            to_close.append((addr, pos, reason, cur, pct))
        else: 
            state["open_positions"][addr] = pos

    for addr, pos, reason, cur, pct in to_close:
        try_sell(addr, pos, reason, cur, pct)# Enhanced Telegram Commands with No-Rebuy and Wallet Tracking

HELP_TEXT = (
"/start – intro & status\n"
"/help – show commands\n"
"/status – show bot & filters\n"
"/positions – list open positions\n"
"/setcap <min> <max> – set cap band (USD)\n"
"/setliq <usd> – set min liquidity USD\n"
"/setvol <usd> – set min volume USD on the current window\n"
"/setvolwin <minutes> – set volume window (5..60)\n"
"/setminhold <seconds> – set minimum hold time\n"
"/capunknown on|off – allow tokens with unknown cap to satisfy cap condition\n"
"/nocap – ignore cap condition   |   /yescap – enforce cap condition\n"
"/filtermode OR|AND|MAJORITY|HYBRID – select base filter logic\n"
"/mintrue <1|2|3> – how many base conditions must be true (MAJORITY/HYBRID)\n"
"/hybridcut <minutes> – new-pair cutoff for HYBRID\n"
"/meme on|off – enable/disable Solana meme-only filter\n"
"/launch on|off – enable/disable launch snipe\n"
"/dexallow list|add <id>|rm <id> – manage allowed dex ids\n"
"/setage <minutes> – max pair age for launch snipe\n"
"/rugcheck off|lenient|strict [score] – set rugcheck mode & min score\n"
"/tw on|off – enable/disable Twitter monitor\n"
"/twadd <@handle> | /twrm <@handle> | /twlist – manage watched accounts\n"
"/twpos add <kw> | rm <kw> | list – manage positive keywords\n"
"/quantum on|off [threshold] – enable smart scorer & set threshold\n"
"/qstatus – show scorer config & weights\n"
"/qparams – dump current quantum params (JSON)\n"
"/qset <key> <value> – set a quantum param\n"
"/ai on|off [threshold] – enable advanced AI analysis & set confidence threshold\n"
"/aistatus – show AI analysis configuration\n"
"/aitest <symbol> – test AI analysis on specific pair\n"
"/aiconfig pattern|volume on/off – require pattern/volume for AI path\n"
"/aiconfig threshold <0-100> – set AI confidence threshold\n"
"/2x on|off [minutes] [alpha] – toggle 2× predictor, optional minutes & alpha\n"
"/setmom <seconds> – set momentum window in seconds (default 60)\n"
"/norebuy on|off – toggle re-buy prevention after selling\n"
"/rebuytime <seconds> – set no-rebuy cooldown period\n"
"/clearsold – clear all sold tokens tracking\n"
"/walltrack on|off – enable/disable wallet tracking\n"
"/wallroi <percentage> – set minimum ROI to track wallets\n"
"/walllist – show tracked high-ROI wallets\n"
"/diag – show base filter diagnostics\n"
"/pause – pause NEW buys   |   /resume – resume NEW buys\n"
"/forceresume – force resume and reset all cooldowns\n"
"/ultramode – ultra-aggressive mode for low cap coins with rugcheck\n"
"/panel – show control buttons\n"
"/closeall – close all open positions at current market\n"
"/ping – check bot responsiveness\n"
"/testapi – test Dexscreener API endpoints\n"
"/testpairs – debug: show sample pairs being processed\n"
"/debugfilters – detailed filter analysis and troubleshooting\n"
"/mememode – quick setup for optimal meme token sniping\n"
)

def panel_markup(paused: bool) -> dict:
    if paused:
        buttons = [[{"text": "▶️ Resume", "callback_data": "RESUME"}],
                   [{"text": "📊 Status", "callback_data": "STATUS"}]]
    else:
        buttons = [[{"text": "⏸️ Pause", "callback_data": "PAUSE"}],
                   [{"text": "📊 Status", "callback_data": "STATUS"}]]
    return {"inline_keyboard": buttons}

def get_updates(offset: int) -> Tuple[int, List[dict]]:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        r = requests.get(url, params={"timeout": 25, "offset": offset}, timeout=30)
        if not r.ok: return offset, []
        data = r.json() or {}
        if not data.get("ok"): return offset, []
        updates = data.get("result") or []
        new_offset = offset
        for u in updates:
            uid = int(u.get("update_id", 0))
            new_offset = max(new_offset, uid + 1)
        return new_offset, updates
    except Exception as e:
        print("TG getUpdates error:", e)
        return offset, []

def _status_text() -> str:
    cap_lo, cap_hi = settings.get("cap_range", [0, 0])
    meme_status = "ON" if settings.get("meme_only", True) else "OFF"
    sol_only = "ON" if settings.get("solana_only", True) else "OFF"
    two_x = "ON" if settings.get("use_two_x_predict", True) else "OFF"
    norebuy_status = "ON" if settings.get("enable_no_rebuy", True) else "OFF"
    norebuy_time = int(settings.get("no_rebuy_after_sell_sec", 3600)) // 60
    wallet_track = "ON" if settings.get("wallet_tracking_enabled", True) else "OFF"
    min_roi = settings.get("min_wallet_roi", 35.0)
    tracked_count = len(settings.get("tracked_wallets", []))
    sold_count = len(state.get("_sold_mints", {}))
    
    return (
        f"{'⏸️ PAUSED' if state.get('paused') else '▶️ RUNNING'} (paper)\n"
        f"Balance: {state['balance_sol']:.4f} SOL | Realized PnL: {state['realized_pnl_sol']:+.4f} SOL\n"
        f"Hold≥{int(settings['min_hold_sec'])}s | BuyCD {int(settings['buy_cooldown_sec'])}s | GlobalCD {int(settings['global_cooldown_sec'])}s\n"
        f"🚫 No-Rebuy: {norebuy_status} ({norebuy_time}min cooldown, {sold_count} blocked)\n"
        f"📈 Wallet Track: {wallet_track} (ROI≥{min_roi}%, tracking {tracked_count} wallets)\n"
        f"Base Filter [{settings.get('filter_mode','OR')}]: "
        f"liq≥{int(settings['min_liq']):,} / vol≥{int(settings['min_vol_usd']):,} on {int(settings['vol_window_min'])}m / "
        f"cap∈[{int(cap_lo):,},{int(cap_hi):,}] (cap_enforced={bool(settings.get('enforce_cap'))}, allow_unknown={bool(settings.get('pass_if_cap_unknown'))}, min_true={int(settings.get('min_true',2))})\n"
        f"Meme focus: {meme_status} (solana_only={sol_only}, quotes={','.join(settings.get('meme_quote_whitelist',[]))})\n"
        f"LaunchSnipe: {('ON' if settings.get('launch_enabled') else 'OFF')} "
        f"(dex={','.join(settings.get('allowed_dex_ids',[]))}, age≤{int(settings.get('max_pair_age_min',8))}m, 2x_required={bool(settings.get('launch_require_2x',True))})\n"
        f"Quantum: {('ON' if settings.get('quantum_enabled') else 'OFF')} (thr={int(settings.get('quantum_threshold',60))}, gate=(liq OR vol), 2x_required={bool(settings.get('quantum_require_2x',True))})\n"
        f"🤖 AI Analysis: {('ON' if settings.get('ai_enabled') else 'OFF')} (thr={int(settings.get('ai_confidence_threshold',70))}%, pattern_req={bool(settings.get('ai_require_pattern'))}, vol_req={bool(settings.get('ai_require_volume_spike'))})\n"
        f"2× Predictor: {two_x} (M={float(settings.get('two_x_minutes',2))}m, alpha={float(settings.get('two_x_alpha',0.8))}, window={int(settings.get('momentum_window_sec',60))}s)\n"
        f"RugCheck: {settings.get('rugcheck_mode','off').upper()} (min {int(settings['rugcheck_score_min'])})\n"
        f"Twitter: {('ON' if twitter_enabled() else 'OFF')} (tracked: {len(settings.get('twitter_accounts',[]))})\n"
        f"Sizing: {('%.2f SOL' % settings['per_trade_sol']) if settings['mode']=='amount' else ('%.0f%%' % (settings['percent']*100))}; "
        f"TP {settings['take_profit']*100:.0f}%, SL {settings['stop_loss']*100:.0f}%, Trail {settings['trail']*100:.0f}%\n"
        f"MaxPos {settings['max_positions']}"
    )

def handle_text(msg: str) -> None:
    parts = (msg or "").strip().split()
    if not parts: return
    cmd = parts[0].lower()

    if cmd in ("/start", "/help"):
        tg_send("✅ Bot online.\n" + _status_text() + "\n\n" + HELP_TEXT)

    elif cmd == "/status":
        tg_send(_status_text())

    elif cmd == "/positions":
        if not state["open_positions"]:
            tg_send("No open positions."); return
        lines = []
        for addr, p in state["open_positions"].items():
            lines.append(f"{p.get('symbol','?')}  in:{float(p.get('invest_sol',0)):.4f} SOL  entry:{float(p.get('entry_price',0)):.10f}")
        tg_send("Open positions:\n" + "\n".join(lines))

    # NEW NO-REBUY COMMANDS
    elif cmd == "/norebuy" and len(parts) >= 2:
        val = parts[1].lower()
        if val in ("on","true","1"):
            settings["enable_no_rebuy"] = True
            save_json(SETTINGS_FILE, settings)
            tg_send(f"✅ No-rebuy enabled ({int(settings.get('no_rebuy_after_sell_sec', 3600))//60}min cooldown)")
        elif val in ("off","false","0"):
            settings["enable_no_rebuy"] = False
            save_json(SETTINGS_FILE, settings)
            tg_send("✅ No-rebuy disabled (can re-buy sold tokens)")
        else:
            tg_send("❌ Usage: /norebuy on|off")

    elif cmd == "/rebuytime" and len(parts) >= 2:
        try:
            seconds = max(60, int(parts[1]))  # Minimum 1 minute
            settings["no_rebuy_after_sell_sec"] = seconds
            save_json(SETTINGS_FILE, settings)
            tg_send(f"✅ No-rebuy cooldown set to {seconds}s ({seconds//60}min)")
        except Exception:
            tg_send("❌ Example: /rebuytime 3600 (for 1 hour)")

    elif cmd == "/clearsold":
        state["_sold_mints"] = {}
        save_json(STATE_FILE, state)
        tg_send("✅ Cleared all sold tokens tracking. Can now re-buy previously sold tokens.")

    # WALLET TRACKING COMMANDS
    elif cmd == "/walltrack" and len(parts) >= 2:
        val = parts[1].lower()
        if val in ("on","true","1"):
            settings["wallet_tracking_enabled"] = True
            save_json(SETTINGS_FILE, settings)
            tg_send(f"✅ Wallet tracking enabled (ROI≥{settings.get('min_wallet_roi', 35)}%)")
        elif val in ("off","false","0"):
            settings["wallet_tracking_enabled"] = False
            save_json(SETTINGS_FILE, settings)
            tg_send("✅ Wallet tracking disabled")
        else:
            tg_send("❌ Usage: /walltrack on|off")

    elif cmd == "/wallroi" and len(parts) >= 2:
        try:
            roi = max(10, min(100, float(parts[1])))
            settings["min_wallet_roi"] = roi
            save_json(SETTINGS_FILE, settings)
            tg_send(f"✅ Minimum wallet ROI set to {roi}%")
        except Exception:
            tg_send("❌ Example: /wallroi 35")

    elif cmd == "/walllist":
        tracked = settings.get("tracked_wallets", [])
        activity = state.get("_tracked_wallet_activity", {})
        if not tracked:
            tg_send("No wallets being tracked.")
            return
        
        lines = []
        for wallet in tracked[:10]:  # Show top 10
            data = activity.get(wallet, {})
            avg_roi = data.get("avg_roi", 0)
            trades = data.get("total_trades", 0)
            lines.append(f"{wallet[:8]}... | Avg ROI: {avg_roi:.1f}% | Trades: {trades}")
        
        tg_send(f"📈 Tracked Wallets ({len(tracked)}):\n" + "\n".join(lines))

    elif cmd == "/setcap" and len(parts) >= 3:
        try:
            lo = int(parts[1].replace(",", "")); hi = int(parts[2].replace(",", ""))
            if lo < 0 or hi < 0 or hi < lo: tg_send("❌ Invalid range. Example: /setcap 10000 5000000")
            else:
                settings["cap_range"] = [lo, hi]; save_json(SETTINGS_FILE, settings)
                tg_send(f"✅ cap range set to [{lo:,},{hi:,}]")
        except Exception:
            tg_send("❌ Could not parse numbers. Example: /setcap 10000 5000000")

    elif cmd == "/ultramode":
        # IMPROVED Ultra-aggressive mode for low cap coins with mandatory rugcheck
        settings.update({
            "min_liq": 100,                    # Ultra low liquidity for micro caps
            "min_vol_usd": 15,                 # Ultra low volume
            "vol_window_min": 5,
            "cap_range": [500, 200_000],       # NEW: Focus on $500-$200K market cap
            "enforce_cap": True,               # NEW: Enforce cap for ultra mode
            "pass_if_cap_unknown": False,      # NEW: Require known cap
            "filter_mode": "OR",
            "launch_enabled": True,
            "max_pair_age_min": 60,            # 1 hour max age
            "launch_liq_min": 75,              # Ultra low launch liq
            "launch_min_buys_m5": 0,           # No minimum buys required
            "launch_require_2x": False,
            "ai_enabled": True,
            "ai_confidence_threshold": 35,     # Very low AI threshold
            "quantum_enabled": True,
            "quantum_threshold": 25,           # Very low quantum threshold
            "quantum_require_2x": False,
            "rugcheck_mode": "strict",         # NEW: MANDATORY rugcheck for ultra mode
            "rugcheck_score_min": 75,          # NEW: Higher rugcheck requirement
            "buy_cooldown_sec": 20,            # Very short cooldowns
            "global_cooldown_sec": 3,
            "enable_no_rebuy": True,           # NEW: Ensure no-rebuy is enabled
            "no_rebuy_after_sell_sec": 1800,   # NEW: 30min cooldown for ultra mode
        })
        state["paused"] = False
        state["last_buy_time"] = 0
        state["last_global"] = 0
        save_json(SETTINGS_FILE, settings)
        save_json(STATE_FILE, state)
        tg_send("⚡ ULTRA MODE ACTIVATED!\n"
               "🎯 MICRO-CAP FOCUS: $500-$200K market cap\n"
               "🛡️ MANDATORY RUGCHECK: Score ≥75 required\n"
               "🚫 NO-REBUY: 30min cooldown after selling\n"
               "• Min liq: $100 | Min vol: $15\n"
               "• Enforced cap range | Known caps only\n" 
               "• AI threshold: 35% | Quantum: 25%\n"
               "• Ultra-short cooldowns (20s/3s)\n"
               "⚠️ SAFER but aggressive for proven micro-caps!")

    elif cmd == "/setliq" and len(parts) >= 2:
        try:
            v = int(parts[1].replace(",", "")); settings["min_liq"] = max(0, v)
            save_json(SETTINGS_FILE, settings); tg_send(f"✅ min_liq set to {v:,}")
        except Exception: tg_send("❌ Example: /setliq 2000")

    elif cmd == "/setvol" and len(parts) >= 2:
        try:
            v = int(parts[1].replace(",", "")); settings["min_vol_usd"] = max(0, v)
            save_json(SETTINGS_FILE, settings); tg_send(f"✅ min_vol_usd set to {v:,}")
        except Exception: tg_send("❌ Example: /setvol 300")

    elif cmd == "/pause":
        state["paused"] = True; save_json(STATE_FILE, state)
        tg_send("⏸️ Paused (no new buys). Use /resume to continue.")

    elif cmd == "/resume":
        state["paused"] = False; save_json(STATE_FILE, state)
        tg_send("▶️ Resumed (new buys allowed).")
    
    elif cmd == "/forceresume":
        # Force resume and reset cooldowns
        state["paused"] = False
        state["last_buy_time"] = 0
        state["last_global"] = 0
        save_json(STATE_FILE, state)
        tg_send("🚀 FORCE RESUMED! All cooldowns reset, ready to buy immediately.")

    elif cmd == "/closeall":
        pairs = fetch_pairs(); price_map = current_price_map(pairs)
        for addr, pos in list(state["open_positions"].items()):
            cur = price_map.get(addr, pos.get("entry_price", 0.0))
            pct, _ = position_value_change_pct(cur, pos.get("entry_price", 0.0), pos.get("highest_price", 0.0))
            try_sell(addr, pos, "manual close", cur, pct)
        tg_send("✅ All positions closed.")

    elif cmd == "/ping":
        tg_send("pong 🟩")

def handle_callback(u: dict):
    cb = u.get("callback_query") or {}
    data = cb.get("data") or ""
    cb_id = cb.get("id") or ""
    chat = (cb.get("message") or {}).get("chat") or {}
    from_chat = str(chat.get("id") or "")
    if TELEGRAM_CHAT_ID and from_chat and str(from_chat) != str(TELEGRAM_CHAT_ID):
        tg_answer_callback(cb_id, "Unauthorized"); return
    if data == "PAUSE":
        state["paused"] = True; save_json(STATE_FILE, state)
        tg_answer_callback(cb_id, "Paused")
        tg_send("⏸️ Paused (no new buys).", reply_markup=panel_markup(True))
    elif data == "RESUME":
        state["paused"] = False; save_json(STATE_FILE, state)
        tg_answer_callback(cb_id, "Resumed")
        tg_send("▶️ Resumed (new buys allowed).", reply_markup=panel_markup(False))
    elif data == "STATUS":
        tg_answer_callback(cb_id)
        tg_send(_status_text(), reply_markup=panel_markup(bool(state.get("paused"))))
    else:
        tg_answer_callback(cb_id, "Unknown")

# =========================
# === Trading Loops ======
# =========================
def trading_loop():
    while True:
        try:
            # Clean up expired sold mints periodically
            cleanup_expired_sold_mints()
            
            if state.get("backoff", 0) > 0:
                time.sleep(int(state["backoff"]))

            pairs = fetch_pairs()
            if not pairs:
                tg_send("🚨 No pairs fetched from Dexscreener. Retrying...")
                time.sleep(int(settings["poll_sec"]))
                continue
            
            print(f"[INFO] Successfully fetched {len(pairs)} pairs")

            # Debug: Show what we got
            total_pairs = len(pairs)
            meme_candidates = 0
            new_pairs = 0
            for p in pairs:
                is_meme, _ = is_meme_candidate(p)
                if is_meme:
                    meme_candidates += 1
                if is_new_pair(p):
                    new_pairs += 1
            
            if settings.get("debug_buys", False):
                tg_send(f"🔍 Scanning: {total_pairs} pairs, {meme_candidates} meme candidates, {new_pairs} new pairs")

            # Keep a price map for momentum + PnL management
            price_by_addr = current_price_map(pairs)

            # 1) Launch snipe — fastest path (meme gate is inside should_launch_snipe)
            debug_msgs = 0
            launch_candidates = 0
            launch_rejected = {}
            
            for p in pairs:
                ok, why = should_launch_snipe(p)
                if ok:
                    launch_candidates += 1
                    if not state.get("paused", False) and can_buy():
                        if not try_buy(p, reason=why) and settings.get("debug_buys") and debug_msgs < int(settings.get("debug_max_msgs_per_cycle", 5)):
                            tg_send(f"SKIP launch: {_why_not_buy(p, why)}")
                            debug_msgs += 1
                else:
                    # Track why launches are being rejected
                    rejection_reason = why.split(":")[0] if ":" in why else why
                    launch_rejected[rejection_reason] = launch_rejected.get(rejection_reason, 0) + 1
            
            if settings.get("debug_buys", False) and launch_candidates == 0 and launch_rejected:
                top_rejections = sorted(launch_rejected.items(), key=lambda x: x[1], reverse=True)[:3]
                rejection_summary = ", ".join([f"{reason}({count})" for reason, count in top_rejections])
                tg_send(f"DEBUG: No launch candidates. Top rejections: {rejection_summary}")

            # 2) Quantum smart scorer — must satisfy (liq OR vol) and (2x predictor)
            if settings.get("quantum_enabled", True) and not state.get("paused", False):
                thr = float(settings.get("quantum_threshold", 60))
                q_debug = 0
                vol_win = max(5, min(60, int(safe_float(settings.get("vol_window_min", 10), 10))))
                # Score meme candidates only, try high-scorers first
                scored = []
                for p in pairs:
                    mem_ok, _ = is_meme_candidate(p)
                    if not mem_ok:
                        continue
                    s, br = quantum_score(p)
                    scored.append((s, p, br))
                scored.sort(key=lambda t: t[0], reverse=True)

                for s, p, br in scored:
                    if s < thr: break

                    # Gate: liq_ok OR vol_ok (cap not required for quantum)
                    liq_ok, vol_ok, _cap_ok = _base_conditions(p, vol_win)
                    if not (liq_ok or vol_ok):
                        if settings.get("debug_buys") and q_debug < int(settings.get("debug_max_msgs_per_cycle", 5)):
                            tg_send(f"SKIP quantum ({_pair_symbol(p)}): need (liq OR vol)")
                            q_debug += 1
                        continue

                    # 2× predictor
                    if settings.get("quantum_require_2x", True):
                        addr = p.get("pairAddress") or ""
                        pr = price_by_addr.get(addr, _read_price(p))
                        if not two_x_predict(addr, pr):
                            if settings.get("debug_buys") and q_debug < int(settings.get("debug_max_msgs_per_cycle", 5)):
                                tg_send(f"SKIP quantum ({_pair_symbol(p)}): 2x predictor not met")
                                q_debug += 1
                            continue

                    reason = (f"quantum {s:.1f}% "
                              f"cap:{br['cap']:.0f} age:{br['age']:.0f} buys:{br['buys']:.0f} "
                              f"bsr:{br['bsr']:.0f} vel:{br['vel']:.0f}")
                    if not try_buy(p, reason=reason) and settings.get("debug_buys") and q_debug < int(settings.get("debug_max_msgs_per_cycle", 5)):
                        tg_send(f"SKIP quantum: {_why_not_buy(p)}")
                        q_debug += 1
                    if not can_buy():
                        break

            # 3) AI-Powered Analysis Path
            if settings.get("ai_enabled", True) and not state.get("paused", False):
                ai_threshold = float(settings.get("ai_confidence_threshold", 70))
                ai_debug = 0
                ai_candidates = []
                
                for p in pairs:
                    mem_ok, _ = is_meme_candidate(p)
                    if not mem_ok:
                        continue
                    
                    pair_addr = p.get("pairAddress") or ""
                    if not pair_addr:
                        continue
                    
                    ai_score, ai_breakdown = AdvancedAnalyzer.calculate_ai_confidence(pair_addr, p)
                    
                    if ai_score >= ai_threshold:
                        ai_candidates.append((ai_score, p, ai_breakdown))
                
                ai_candidates.sort(key=lambda x: x[0], reverse=True)
                
                for ai_score, p, breakdown in ai_candidates:
                    if not can_buy():
                        break
                    
                    reason = (f"AI {ai_score:.1f}% "
                             f"mom:{breakdown['momentum']:.0f} vol:{breakdown['volume']:.0f} "
                             f"pat:{breakdown['pattern']:.0f} whale:{breakdown['whale']:.0f}")
                    
                    if not try_buy(p, reason=reason) and settings.get("debug_buys") and ai_debug < int(settings.get("debug_max_msgs_per_cycle", 5)):
                        tg_send(f"SKIP AI: {_why_not_buy(p)}")
                        ai_debug += 1

            # 4) Base Filter (configurable) + meme gate already inside
            filtered = filter_pairs_logic(pairs)

            # manage existing positions
            if state["open_positions"]:
                manage_positions(price_by_addr)

            # attempt buys from Base Filter (optional 2x requirement)
            buys = 0
            debug_msgs = 0
            if not state.get("paused", False):
                for p in filtered:
                    if not can_buy():
                        if settings.get("debug_buys") and debug_msgs < int(settings.get("debug_max_msgs_per_cycle", 5)):
                            tg_send(f"SKIP (gate): {_why_not_buy(p)}")
                            debug_msgs += 1
                        break

                    if settings.get("base_require_2x", False):
                        addr = p.get("pairAddress") or ""
                        pr = price_by_addr.get(addr, _read_price(p))
                        if not two_x_predict(addr, pr):
                            if settings.get("debug_buys") and debug_msgs < int(settings.get("debug_max_msgs_per_cycle", 5)):
                                tg_send(f"SKIP base ({_pair_symbol(p)}): 2x predictor not met")
                                debug_msgs += 1
                            continue

                    if try_buy(p, reason=f"Base {settings.get('filter_mode','OR')}"):
                        buys += 1
                    elif settings.get("debug_buys") and debug_msgs < int(settings.get("debug_max_msgs_per_cycle", 5)):
                        tg_send(f"SKIP ({_pair_symbol(p)}): {_why_not_buy(p)}")
                        debug_msgs += 1

            if buys == 0 and not filtered and launch_candidates == 0:
                cap_lo, cap_hi = settings.get("cap_range", [0, 0])
                tg_send(
                    "DEBUG: 0 candidates found. Check settings or /debugfilters.\n"
                    f"(Mode={settings.get('filter_mode','OR')}, min_true={int(settings.get('min_true',2))}, "
                    f"Thresholds: liq≥${int(settings['min_liq']):,} / "
                    f"vol≥${int(settings['min_vol_usd']):,} on {int(settings['vol_window_min'])}m / "
                    f"cap∈[${int(cap_lo):,},${int(cap_hi):,}])"
                )

        except Exception as e:
            print("trade loop error:", e)
            tg_send(f"🚨 Trading loop error: {str(e)[:100]}")

        time.sleep(int(settings["poll_sec"]))

def telegram_loop():
    # Auto-resume if paused on startup
    if state.get("paused", False):
        state["paused"] = False
        save_json(STATE_FILE, state)
        tg_send("🔄 Auto-resumed on startup")
    
    tg_send("✅ Bot online with NO-REBUY system!\n" + _status_text() + "\nType /panel for buttons or /help for commands.")
    while True:
        try:
            state["telegram_offset"], updates = get_updates(int(state.get("telegram_offset", 0)))
            save_json(STATE_FILE, state)
            for u in updates:
                if "message" in u:
                    msg = ((u.get("message") or {}).get("text")) or ""
                    chat_id = str(((u.get("message") or {}).get("chat") or {}).get("id") or "")
                    if not msg: continue
                    if TELEGRAM_CHAT_ID and chat_id and str(chat_id) != str(TELEGRAM_CHAT_ID): continue
                    handle_text(msg)
                elif "callback_query" in u:
                    handle_callback(u)
        except Exception as e:
            print("telegram_loop error:", e)
        time.sleep(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    t1 = threading.Thread(target=telegram_loop, daemon=True)
    t2 = threading.Thread(target=trading_loop, daemon=True)
    t1.start(); t2.start()

    while True:
        time.sleep(3600)

# Missing helper functions
def twitter_enabled() -> bool:
    return bool(settings.get("twitter_enabled")) and bool(settings.get("twitter_bearer")) and len(settings.get("twitter_accounts", [])) > 0

def quantum_score(p: dict) -> Tuple[float, Dict[str,float]]:
    """Returns (score_0_100, breakdown dict)."""
    # Extract features
    fdv = safe_float(p.get("fdv"), -1)
    if fdv <= 0:
        fdv = safe_float(p.get("marketCap"), -1)
    liq  = safe_float((p.get("liquidity") or {}).get("usd"), 0.0)
    tx_m5 = (p.get("txns") or {}).get("m5") or {}
    buys_m5  = safe_float(tx_m5.get("buys"), 0.0)
    sells_m5 = safe_float(tx_m5.get("sells"), 0.0)
    vol_win = _vol_window_value(p, max(5, int(settings.get("vol_window_min", 10))))

    # Params
    cap_lo = float(settings.get("q_cap_lo", 10_000))
    cap_hi = float(settings.get("q_cap_hi", 5_000_000))
    w_cap  = float(settings.get("q_weight_cap", 30))
    w_age  = float(settings.get("q_weight_age", 15))
    w_buys = float(settings.get("q_weight_buys", 20))
    w_bs   = float(settings.get("q_weight_buysell", 15))
    w_vel  = float(settings.get("q_weight_velocity", 15))
    w_ven  = float(settings.get("q_weight_venue", 3))
    w_soc  = float(settings.get("q_weight_social", 2))

    max_age   = float(settings.get("q_max_age_min", 60))
    max_buys  = float(settings.get("q_max_buys_m5", 50))
    max_bsr   = float(settings.get("q_max_buysell", 5.0))
    max_vel   = float(settings.get("q_max_vel", 1.0))

    # 1) Small-cap preference
    cap_score = 0.0
    if fdv > 0:
        if fdv < cap_lo:
            cap_score = 1.0
        elif fdv <= cap_hi:
            cap_score = 1.0 - 0.7 * ((fdv - cap_lo) / max(1.0, cap_hi - cap_lo))
        else:
            over = fdv - cap_hi
            cap_score = max(0.0, 0.3 - 0.3 * (over / (10*cap_hi)))

    # 2) Age (newer is better)
    age = _age_minutes(p)
    age_score = 1.0 - _normalize(age, max_age)
    age_score = max(0.0, min(1.0, age_score))

    # 3) Buys m5
    buys_score = _normalize(buys_m5, max_buys)

    # 4) Buy/Sell ratio
    bsr = buys_m5 / (sells_m5 + 1.0)
    bsr_score = _normalize(bsr, max_bsr)

    # 5) Velocity: (vol_window / liq)
    vel = (vol_win / max(1.0, liq))
    vel_score = _normalize(vel, max_vel)

    # 6) Venue bonus
    venue_score = 1.0 if is_allowed_venue(p) else 0.0

    # 7) Social bonus
    soc_score = 0.0
    mint = _mint_from_pair(p)
    if mint and recent_social_hint(mint):
        soc_score = 1.0

    score_raw = (
        w_cap  * cap_score   +
        w_age  * age_score   +
        w_buys * buys_score  +
        w_bs   * bsr_score   +
        w_vel  * vel_score   +
        w_ven  * venue_score +
        w_soc  * soc_score
    )
    max_possible = (w_cap + w_age + w_buys + w_bs + w_vel + w_ven + w_soc)
    score_pct = 0.0 if max_possible <= 0 else (100.0 * score_raw / max_possible)

    breakdown = {
        "cap": round(cap_score*100, 1),
        "age": round(age_score*100, 1),
        "buys": round(buys_score*100, 1),
        "bsr": round(bsr_score*100, 1),
        "vel": round(vel_score*100, 1),
        "venue": round(venue_score*100, 1),
        "social": round(soc_score*100, 1)
    }
    return score_pct, breakdown

def _normalize(x: float, hi: float) -> float:
    if hi <= 0: return 0.0
    v = x / hi
    return max(0.0, min(1.0, v))

def is_allowed_venue(p: dict) -> bool:
    return dex_allowed(p)

def recent_social_hint(mint: str, within_min: int = 20) -> bool:
    ts = state.get("_seen_social_mints", {}).get(mint)
    if not ts: return False
    return (now() - ts) <= within_min * 60

def _graceful_exit(*_):
    try: tg_send("🟨 Bot shutting down.")
    except: pass
    sys.exit(0)