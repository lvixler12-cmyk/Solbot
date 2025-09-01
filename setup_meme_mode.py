#!/usr/bin/env python3
"""
Quick setup script to configure the bot for optimal meme token sniping
Run this to reset all settings to aggressive meme-catching mode
"""

import json
import os

SETTINGS_FILE = "settings.json"

# Ultra-aggressive settings for new meme token detection
MEME_SETTINGS = {
    # Base Filter - Very permissive
    "min_liq": 200,                    # Very low liquidity requirement
    "min_vol_usd": 25,                 # Very low volume requirement  
    "vol_window_min": 5,               # Short window for new tokens
    "cap_range": [500, 20_000_000],    # Wide market cap range
    "enforce_cap": False,              # Don't enforce cap to catch everything
    "pass_if_cap_unknown": True,       # Allow unknown caps
    
    # Filter Logic - Most permissive
    "filter_mode": "OR",               # Any condition passes
    "min_true": 1,                     # Only need 1 condition
    "hybrid_new_age_min": 15,          # Longer new token window
    
    # Meme Focus
    "solana_only": True,
    "meme_only": True,
    "meme_quote_whitelist": ["SOL", "USDC", "USDT"],
    "meme_base_blacklist": ["SOL", "WSOL", "USDT", "USDC", "BTC", "ETH", "WBNB", "WAVAX", "WPLS"],
    
    # Launch Snipe - Very aggressive
    "launch_enabled": True,
    "allowed_dex_ids": ["raydium", "pump", "meteora", "bunk", "jupiter"],
    "max_pair_age_min": 45,            # Catch pairs up to 45 minutes old
    "launch_liq_min": 150,             # Very low liquidity requirement
    "launch_min_buys_m5": 1,           # Just need 1 buy to show interest
    "launch_require_2x": False,        # Don't require 2x prediction for launch
    
    # RugCheck - Lenient for more opportunities
    "rugcheck_mode": "lenient",
    "rugcheck_score_min": 50,          # Lower threshold
    "rugcheck_timeout_sec": 3,         # Faster timeout
    
    # AI Analysis - Moderate threshold
    "ai_enabled": True,
    "ai_confidence_threshold": 60,     # Lower threshold for more opportunities
    "ai_require_pattern": False,       # Don't require patterns
    "ai_require_volume_spike": False,  # Don't require volume spikes
    
    # Quantum - Lower threshold
    "quantum_enabled": True,
    "quantum_threshold": 50,           # Lower threshold
    "quantum_require_2x": False,       # Don't require 2x for quantum
    
    # 2x Predictor - Disabled for more opportunities
    "use_two_x_predict": False,        # Disable to catch more early
    "two_x_minutes": 2,
    "two_x_alpha": 0.6,                # Lower alpha when enabled
    "momentum_window_sec": 45,         # Shorter window
    "momentum_min_dt_sec": 20,         # Shorter minimum time
    "base_require_2x": False,          # Don't require for base filter
    
    # Risk Management - Moderate
    "mode": "amount",
    "per_trade_sol": 1.0,              # Smaller position size for more opportunities
    "percent": 0.08,
    "take_profit": 0.50,               # 50% take profit
    "stop_loss": 0.20,                 # 20% stop loss
    "trail": 0.08,                     # 8% trailing stop
    
    # Limits - Allow more positions
    "max_positions": 8,                # More concurrent positions
    "buy_cooldown_sec": 60,            # Shorter cooldown
    "global_cooldown_sec": 8,          # Shorter global cooldown
    "min_hold_sec": 3,                 # Shorter minimum hold
    
    # Timing - Faster polling
    "poll_sec": 3,                     # Very fast polling
    "social_poll_sec": 10,
    
    # Debug
    "debug_buys": True,
    "debug_max_msgs_per_cycle": 8,     # More debug messages
}

def setup_meme_mode():
    """Apply ultra-aggressive meme sniping settings"""
    
    # Load existing settings if any
    existing = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                existing = json.load(f)
        except:
            pass
    
    # Update with meme settings
    existing.update(MEME_SETTINGS)
    
    # Save settings
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(existing, f, indent=2)
        print("✅ MEME MODE ACTIVATED!")
        print("🎯 Ultra-aggressive settings applied:")
        print(f"   • Min liquidity: ${MEME_SETTINGS['min_liq']}")
        print(f"   • Min volume: ${MEME_SETTINGS['min_vol_usd']} ({MEME_SETTINGS['vol_window_min']}m)")
        print(f"   • Market cap: ${MEME_SETTINGS['cap_range'][0]:,} - ${MEME_SETTINGS['cap_range'][1]:,}")
        print(f"   • Max age: {MEME_SETTINGS['max_pair_age_min']} minutes")
        print(f"   • Filter mode: {MEME_SETTINGS['filter_mode']} (most permissive)")
        print(f"   • AI threshold: {MEME_SETTINGS['ai_confidence_threshold']}%")
        print(f"   • Position size: {MEME_SETTINGS['per_trade_sol']} SOL")
        print(f"   • Polling: every {MEME_SETTINGS['poll_sec']} seconds")
        print("\n🚀 Ready to catch new meme tokens!")
        print("   Start the bot and use /debugfilters to see results")
        
    except Exception as e:
        print(f"❌ Error saving settings: {e}")

if __name__ == "__main__":
    setup_meme_mode()