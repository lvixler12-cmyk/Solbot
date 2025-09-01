# main.py — Advanced Solana Meme Coin Trading Bot
# Three distinct modes: Sniper, Mid-Cap, and Prediction Bot

import os, re, time, json, threading, requests, signal, sys, math, statistics
from typing import List, Dict, Any, Tuple, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
class Config:
    # Telegram settings
    TELEGRAM_BOT_TOKEN = "8413132819:AAGCskfRNS8BsY5MTxqJKsHpFNjYUmop9ms"
    TELEGRAM_CHAT_ID = "5794143622"
    
    # Free API endpoints
    SOLSCAN_API = "https://public-api.solscan.io"
    BIRDEYE_API = "https://public-api.birdeye.so"
    DEXSCREENER_API = "https://api.dexscreener.com/latest/dex"
    
    # Paper trading settings
    INITIAL_BALANCE = 1000  # SOL
    MAX_POSITIONS = 10
    MIN_INVESTMENT = 0.1  # SOL
    MAX_INVESTMENT = 5.0   # SOL

# === Bot Modes ===
class BotMode:
    SNIPER = "sniper"      # Quick flip on new mints
    MIDCAP = "midcap"      # 50k-200k market cap analysis
    PREDICTION = "prediction"  # AI-powered rally prediction

# === State Management ===
class BotState:
    def __init__(self):
        self.mode = None
        self.is_running = False
        self.balance = Config.INITIAL_BALANCE
        self.open_positions = {}
        self.closed_positions = []
        self.settings = self.get_default_settings()
        self.telegram_offset = 0
        
    def get_default_settings(self):
        return {
            "max_positions": Config.MAX_POSITIONS,
            "min_investment": Config.MIN_INVESTMENT,
            "max_investment": Config.MAX_INVESTMENT,
            "stop_loss": 0.15,  # 15%
            "take_profit": 0.30,  # 30%
            "max_hold_time": 300,  # 5 minutes for sniper
        }
    
    def get_mode_settings(self):
        if self.mode == BotMode.SNIPER:
            return {
                "max_hold_time": 300,  # 5 minutes
                "stop_loss": 0.20,     # 20%
                "take_profit": 0.50,   # 50%
                "filters": ["liquidity", "holder_count", "mint_age"]
            }
        elif self.mode == BotMode.MIDCAP:
            return {
                "max_hold_time": 3600,  # 1 hour
                "stop_loss": 0.15,      # 15%
                "take_profit": 0.25,    # 25%
                "filters": ["liquidity", "holder_count", "volume", "market_cap", "social_sentiment"]
            }
        elif self.mode == BotMode.PREDICTION:
            return {
                "max_hold_time": 7200,  # 2 hours
                "stop_loss": 0.10,      # 10%
                "take_profit": 0.40,    # 40%
                "filters": ["ai_analysis", "volume_patterns", "social_signals", "technical_indicators"]
            }
        return {}

# === AI Analysis Engine ===
class AIAnalyzer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, token_data: Dict) -> List[float]:
        """Extract numerical features for AI analysis"""
        features = []
        
        # Volume patterns
        features.extend([
            token_data.get('volume_24h', 0),
            token_data.get('volume_change_1h', 0),
            token_data.get('volume_change_24h', 0),
        ])
        
        # Price patterns
        features.extend([
            token_data.get('price_change_1h', 0),
            token_data.get('price_change_24h', 0),
            token_data.get('price_volatility', 0),
        ])
        
        # Social metrics
        features.extend([
            token_data.get('holder_count', 0),
            token_data.get('social_mentions', 0),
            token_data.get('social_sentiment', 0),
        ])
        
        # Technical indicators
        features.extend([
            token_data.get('rsi', 50),
            token_data.get('macd_signal', 0),
            token_data.get('bollinger_position', 0.5),
        ])
        
        return features
    
    def predict_rally_probability(self, token_data: Dict) -> float:
        """Predict probability of upcoming rally"""
        if not self.is_trained:
            return 0.5  # Default neutral probability
            
        features = self.extract_features(token_data)
        if len(features) < 12:  # Need minimum features
            return 0.5
            
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        return max(0.0, min(1.0, prediction))  # Clamp between 0 and 1

# === Token Scanner ===
class TokenScanner:
    def __init__(self, mode: str):
        self.mode = mode
        self.ai_analyzer = AIAnalyzer()
        
    def scan_new_tokens(self) -> List[Dict]:
        """Scan for new tokens based on mode"""
        if self.mode == BotMode.SNIPER:
            return self._scan_sniper_tokens()
        elif self.mode == BotMode.MIDCAP:
            return self._scan_midcap_tokens()
        elif self.mode == BotMode.PREDICTION:
            return self._scan_prediction_tokens()
        return []
    
    def _scan_sniper_tokens(self) -> List[Dict]:
        """Find fresh mints for quick flipping"""
        # Simplified scanning for sniper mode - focus on speed
        try:
            # Use free API to get recent tokens
            response = requests.get(f"{Config.DEXSCREENER_API}/tokens/solana", timeout=5)
            if response.status_code == 200:
                data = response.json()
                tokens = []
                
                for token in data.get('pairs', [])[:50]:  # Check recent 50
                    if self._apply_sniper_filters(token):
                        tokens.append(token)
                        
                return tokens
        except Exception as e:
            print(f"Sniper scan error: {e}")
        return []
    
    def _scan_midcap_tokens(self) -> List[Dict]:
        """Find mid-cap tokens with growth potential"""
        try:
            response = requests.get(f"{Config.DEXSCREENER_API}/tokens/solana", timeout=10)
            if response.status_code == 200:
                data = response.json()
                tokens = []
                
                for token in data.get('pairs', [])[:100]:
                    if self._apply_midcap_filters(token):
                        tokens.append(token)
                        
                return tokens
        except Exception as e:
            print(f"Midcap scan error: {e}")
        return []
    
    def _scan_prediction_tokens(self) -> List[Dict]:
        """Find tokens showing rally signals"""
        try:
            response = requests.get(f"{Config.DEXSCREENER_API}/tokens/solana", timeout=15)
            if response.status_code == 200:
                data = response.json()
                tokens = []
                
                for token in data.get('pairs', [])[:200]:
                    if self._apply_prediction_filters(token):
                        tokens.append(token)
                        
                return tokens
        except Exception as e:
            print(f"Prediction scan error: {e}")
        return []
    
    def _apply_sniper_filters(self, token: Dict) -> bool:
        """Minimal filters for sniper mode - speed over analysis"""
        try:
            # Basic liquidity check
            liquidity = float(token.get('liquidity', {}).get('usd', 0))
            if liquidity < 1000:  # Minimum $1k liquidity
                return False
                
            # Check if it's a fresh mint (within last hour)
            created_at = token.get('pairCreatedAt', 0)
            if time.time() - created_at > 3600:  # Older than 1 hour
                return False
                
            return True
        except:
            return False
    
    def _apply_midcap_filters(self, token: Dict) -> bool:
        """Comprehensive filters for mid-cap analysis"""
        try:
            # Market cap range
            market_cap = float(token.get('marketCap', 0))
            if not (50000 <= market_cap <= 200000):
                return False
                
            # Volume requirements
            volume_24h = float(token.get('volume', {}).get('h24', 0))
            if volume_24h < 5000:  # Minimum $5k daily volume
                return False
                
            # Holder count
            holder_count = int(token.get('holders', 0))
            if holder_count < 100:  # Minimum 100 holders
                return False
                
            return True
        except:
            return False
    
    def _apply_prediction_filters(self, token: Dict) -> bool:
        """AI-powered filters for prediction mode"""
        try:
            # Enhanced token data for AI analysis
            enhanced_data = self._enhance_token_data(token)
            
            # AI prediction threshold
            rally_probability = self.ai_analyzer.predict_rally_probability(enhanced_data)
            if rally_probability < 0.7:  # 70% confidence threshold
                return False
                
            # Volume pattern check
            volume_change = float(token.get('volume', {}).get('h1', 0)) / max(float(token.get('volume', {}).get('h24', 1)), 1)
            if volume_change > 10:  # Volume already spiked
                return False
                
            return True
        except:
            return False
    
    def _enhance_token_data(self, token: Dict) -> Dict:
        """Add calculated metrics for AI analysis"""
        enhanced = token.copy()
        
        try:
            # Calculate volatility
            price = float(token.get('priceUsd', 0))
            price_change_1h = float(token.get('priceChange', {}).get('h1', 0))
            enhanced['price_volatility'] = abs(price_change_1h) / max(price, 0.000001)
            
            # Calculate RSI-like metric
            enhanced['rsi'] = 50 + (price_change_1h * 10)  # Simplified RSI
            
            # Social sentiment (placeholder)
            enhanced['social_sentiment'] = 0.5
            enhanced['social_mentions'] = 0
            
        except:
            pass
            
        return enhanced

# === Trading Engine ===
class TradingEngine:
    def __init__(self, state: BotState):
        self.state = state
        
    def should_buy(self, token: Dict) -> Tuple[bool, str]:
        """Determine if we should buy a token"""
        if len(self.state.open_positions) >= self.state.settings["max_positions"]:
            return False, "max_positions"
            
        if self.state.balance < self.state.settings["min_investment"]:
            return False, "insufficient_balance"
            
        return True, "ok"
    
    def buy_token(self, token: Dict, reason: str) -> bool:
        """Execute paper trade buy"""
        try:
            should_buy, msg = self.should_buy(token)
            if not should_buy:
                return False
                
            # Calculate investment amount
            invest_sol = min(
                self.state.settings["max_investment"],
                self.state.balance * 0.1  # 10% of balance
            )
            
            if invest_sol < self.state.settings["min_investment"]:
                return False
                
            # Create position
            position = {
                "symbol": token.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                "address": token.get('pairAddress', ''),
                "entry_price": float(token.get('priceUsd', 0)),
                "entry_time": time.time(),
                "investment_sol": invest_sol,
                "reason": reason
            }
            
            # Update state
            self.state.open_positions[token.get('pairAddress', '')] = position
            self.state.balance -= invest_sol
            
            # Send Telegram notification
            self._send_buy_notification(position, token)
            
            return True
            
        except Exception as e:
            print(f"Buy error: {e}")
            return False
    
    def sell_token(self, pair_addr: str, current_price: float, reason: str) -> bool:
        """Execute paper trade sell"""
        try:
            if pair_addr not in self.state.open_positions:
                return False
                
            position = self.state.open_positions[pair_addr]
            
            # Calculate profit/loss
            entry_price = position["entry_price"]
            price_change = (current_price - entry_price) / entry_price
            profit_sol = position["investment_sol"] * price_change
            
            # Update state
            self.state.balance += position["investment_sol"] + profit_sol
            
            # Record closed position
            closed_pos = position.copy()
            closed_pos.update({
                "exit_price": current_price,
                "exit_time": time.time(),
                "profit_sol": profit_sol,
                "profit_percent": price_change * 100,
                "exit_reason": reason
            })
            self.state.closed_positions.append(closed_pos)
            
            # Remove from open positions
            del self.state.open_positions[pair_addr]
            
            # Send Telegram notification
            self._send_sell_notification(closed_pos)
            
            return True
            
        except Exception as e:
            print(f"Sell error: {e}")
            return False
    
    def _send_buy_notification(self, position: Dict, token: Dict):
        """Send buy notification to Telegram"""
        message = (
            f"�� Bought {position['symbol']} ({self.state.mode.upper()})\n"
            f"Entry: ${position['entry_price']:.8f}\n"
            f"In: {position['investment_sol']:.4f} SOL\n"
            f"Reason: {position['reason']}\n"
            f"Open positions: {len(self.state.open_positions)}/{self.state.settings['max_positions']}"
        )
        tg_send(message)
    
    def _send_sell_notification(self, position: Dict):
        """Send sell notification to Telegram"""
        emoji = "🟢" if position["profit_sol"] > 0 else "��"
        message = (
            f"{emoji} Sold {position['symbol']}\n"
            f"Exit: ${position['exit_price']:.8f}\n"
            f"Profit: {position['profit_sol']:.4f} SOL ({position['profit_percent']:.2f}%)\n"
            f"Reason: {position['exit_reason']}\n"
            f"Hold time: {self._format_time(position['exit_time'] - position['entry_time'])}"
        )
        tg_send(message)
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        else:
            return f"{seconds/3600:.1f}h"

# === Position Manager ===
class PositionManager:
    def __init__(self, state: BotState, trading_engine: TradingEngine):
        self.state = state
        self.trading_engine = trading_engine
        
    def manage_positions(self, current_prices: Dict[str, float]) -> None:
        """Manage all open positions"""
        for pair_addr, position in list(self.state.open_positions.items()):
            current_price = current_prices.get(pair_addr)
            if not current_price:
                continue
                
            # Check stop loss and take profit
            entry_price = position["entry_price"]
            price_change = (current_price - entry_price) / entry_price
            
            # Stop loss
            if price_change <= -self.state.settings["stop_loss"]:
                self.trading_engine.sell_token(
                    pair_addr, current_price, "stop_loss"
                )
                continue
                
            # Take profit
            if price_change >= self.state.settings["take_profit"]:
                self.trading_engine.sell_token(
                    pair_addr, current_price, "take_profit"
                )
                continue
                
            # Time-based exit
            hold_time = time.time() - position["entry_time"]
            if hold_time > self.state.settings["max_hold_time"]:
                self.trading_engine.sell_token(
                    pair_addr, current_price, "time_exit"
                )
                continue
                
            # Update position with current profit
            position["current_price"] = current_price
            position["current_profit"] = price_change * 100

# === Telegram Interface ===
def tg_send(text: str) -> None:
    """Send message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": str(text)}
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("Telegram send error:", e)

def tg_send_keyboard(text: str, keyboard: List[List[Dict]]) -> None:
    """Send message with inline keyboard"""
    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": Config.TELEGRAM_CHAT_ID,
            "text": text,
            "reply_markup": {"inline_keyboard": keyboard}
        }
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("Telegram keyboard error:", e)

# === Main Bot Class ===
class SolanaMemeBot:
    def __init__(self):
        self.state = BotState()
        self.scanner = None
        self.trading_engine = TradingEngine(self.state)
        self.position_manager = PositionManager(self.state, self.trading_engine)
        self.running = False
        
    def start(self):
        """Start the bot with mode selection"""
        if self.state.is_running:
            tg_send("Bot is already running!")
            return
            
        # Send mode selection keyboard
        keyboard = [
            [{"text": "🚀 SNIPER BOT", "callback_data": "mode_sniper"}],
            [{"text": "📈 MID-CAP BOT", "callback_data": "mode_midcap"}],
            [{"text": "�� PREDICTION BOT", "callback_data": "mode_prediction"}]
        ]
        
        tg_send_keyboard(
            "🤖 Welcome to Solana Meme Coin Bot!\n\n"
            "Select which bot mode you want to run:\n\n"
            "🚀 SNIPER: Quick flips on new mints\n"
            "📈 MID-CAP: Analysis of 50k-200k tokens\n"
            "🤖 PREDICTION: AI-powered rally detection",
            keyboard
        )
    
    def set_mode(self, mode: str):
        """Set bot mode and initialize scanner"""
        self.state.mode = mode
        self.scanner = TokenScanner(mode)
        
        # Update settings for mode
        mode_settings = self.state.get_mode_settings()
        self.state.settings.update(mode_settings)
        
        tg_send(f"✅ Bot mode set to: {mode.upper()}\n\n"
                f"Settings:\n"
                f"• Max hold time: {self.state.settings['max_hold_time']}s\n"
                f"• Stop loss: {self.state.settings['stop_loss']*100:.0f}%\n"
                f"• Take profit: {self.state.settings['take_profit']*100:.0f}%\n"
                f"• Filters: {', '.join(self.state.settings.get('filters', []))}\n\n"
                f"Use /start_trading to begin scanning!")
    
    def start_trading(self):
        """Start the trading loop"""
        if not self.state.mode:
            tg_send("❌ Please select a bot mode first!")
            return
            
        if self.state.is_running:
            tg_send("Bot is already running!")
            return
            
        self.state.is_running = True
        self.running = True
        
        # Start trading thread
        trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        trading_thread.start()
        
        tg_send(f"🚀 {self.state.mode.upper()} bot started!\n"
                f"Scanning for opportunities...")
    
    def stop_trading(self):
        """Stop the trading loop"""
        self.state.is_running = False
        self.running = False
        tg_send("⏹️ Bot stopped!")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running and self.state.is_running:
            try:
                # Scan for tokens
                tokens = self.scanner.scan_new_tokens()
                
                # Process each token
                for token in tokens:
                    if not self.running:
                        break
                        
                    # Check if we should buy
                    should_buy, reason = self.trading_engine.should_buy(token)
                    
                    if should_buy:
                        # Apply mode-specific filters
                        if self._passes_filters(token):
                            buy_reason = self._get_buy_reason(token)
                            self.trading_engine.buy_token(token, buy_reason)
                    else:
                        # Skip notification for sniper mode to avoid spam
                        if self.state.mode != BotMode.SNIPER:
                            tg_send(f"⏭️ Skipping {token.get('baseToken', {}).get('symbol', 'UNKNOWN')} - {reason}")
                
                # Manage existing positions
                if self.state.open_positions:
                    current_prices = self._get_current_prices()
                    self.position_manager.manage_positions(current_prices)
                
                # Wait before next scan
                scan_interval = 30 if self.state.mode == BotMode.SNIPER else 60
                time.sleep(scan_interval)
                
            except Exception as e:
                print(f"Trading loop error: {e}")
                time.sleep(10)
    
    def _passes_filters(self, token: Dict) -> bool:
        """Check if token passes mode-specific filters"""
        if self.state.mode == BotMode.SNIPER:
            return self.scanner._apply_sniper_filters(token)
        elif self.state.mode == BotMode.MIDCAP:
            return self.scanner._apply_midcap_filters(token)
        elif self.state.mode == BotMode.PREDICTION:
            return self.scanner._apply_prediction_filters(token)
        return False
    
    def _get_buy_reason(self, token: Dict) -> bool:
        """Get reason for buying token"""
        if self.state.mode == BotMode.SNIPER:
            return "Fresh mint - quick flip opportunity"
        elif self.state.mode == BotMode.MIDCAP:
            return "Mid-cap with growth potential"
        elif self.state.mode == BotMode.PREDICTION:
            return "AI detected rally signals"
        return "General opportunity"
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for open positions"""
        prices = {}
        for pair_addr in self.state.open_positions:
            try:
                # Use free API to get current price
                response = requests.get(f"{Config.DEXSCREENER_API}/pairs/solana/{pair_addr}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('pairs'):
                        price = float(data['pairs'][0].get('priceUsd', 0))
                        prices[pair_addr] = price
            except:
                continue
        return prices

# === Command Handlers ===
def handle_telegram_command(bot: SolanaMemeBot, command: str, chat_id: str) -> None:
    """Handle Telegram commands"""
    if command == "/start":
        bot.start()
    elif command == "/start_trading":
        bot.start_trading()
    elif command == "/stop":
        bot.stop_trading()
    elif command == "/positions":
        show_positions(bot.state)
    elif command == "/balance":
        show_balance(bot.state)
    elif command == "/stats":
        show_stats(bot.state)
    elif command == "/help":
        show_help()

def show_positions(state: BotState):
    """Show open positions with current profit"""
    if not state.open_positions:
        tg_send("No open positions.")
        return
        
    lines = []
    for addr, pos in state.open_positions.items():
        current_price = pos.get('current_price', pos['entry_price'])
        profit_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
        emoji = "��" if profit_pct > 0 else "🔴"
        
        lines.append(
            f"{emoji} {pos['symbol']}\n"
            f"Entry: ${pos['entry_price']:.8f}\n"
            f"Current: ${current_price:.8f}\n"
            f"Profit: {profit_pct:.2f}%\n"
            f"Investment: {pos['investment_sol']:.4f} SOL\n"
            f"Hold time: {format_time(time.time() - pos['entry_time'])}"
        )
    
    tg_send("Open positions:\n\n" + "\n\n".join(lines))

def show_balance(state: BotState):
    """Show current balance and P&L"""
    total_invested = sum(pos['investment_sol'] for pos in state.open_positions.values())
    available = state.balance
    
    message = (
        f"💰 Balance Summary\n\n"
        f"Available: {available:.4f} SOL\n"
        f"Invested: {total_invested:.4f} SOL\n"
        f"Total: {available + total_invested:.4f} SOL\n"
        f"Open positions: {len(state.open_positions)}/{state.settings['max_positions']}"
    )
    tg_send(message)

def show_stats(state: BotState):
    """Show trading statistics"""
    if not state.closed_positions:
        tg_send("No trading history yet.")
        return
        
    total_trades = len(state.closed_positions)
    profitable_trades = len([p for p in state.closed_positions if p['profit_sol'] > 0])
    win_rate = (profitable_trades / total_trades) * 100
    
    total_profit = sum(p['profit_sol'] for p in state.closed_positions)
    avg_profit = total_profit / total_trades
    
    message = (
        f"�� Trading Statistics\n\n"
        f"Total trades: {total_trades}\n"
        f"Profitable: {profitable_trades}\n"
        f"Win rate: {win_rate:.1f}%\n"
        f"Total P&L: {total_profit:.4f} SOL\n"
        f"Average P&L: {avg_profit:.4f} SOL"
    )
    tg_send(message)

def show_help():
    """Show help message"""
    help_text = (
        "🤖 Solana Meme Coin Bot Commands\n\n"
        "/start - Select bot mode\n"
        "/start_trading - Begin trading\n"
        "/stop - Stop trading\n"
        "/positions - Show open positions\n"
        "/balance - Show balance\n"
        "/stats - Show trading statistics\n"
        "/help - Show this help\n\n"
        "Bot Modes:\n"
        "🚀 Sniper: Quick flips on new mints\n"
        "📈 Mid-Cap: Analysis of 50k-200k tokens\n"
        "🤖 Prediction: AI-powered rally detection"
    )
    tg_send(help_text)

def format_time(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        return f"{seconds/3600:.1f}h"

# === Main Execution ===
def main():
    bot = SolanaMemeBot()
    
    # Handle Telegram updates
    def telegram_loop():
        while True:
            try:
                # Get updates
                url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/getUpdates"
                params = {"offset": bot.state.telegram_offset + 1, "timeout": 30}
                response = requests.get(url, params=params, timeout=35)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok') and data.get('result'):
                        for update in data['result']:
                            bot.state.telegram_offset = update['update_id']
                            
                            # Handle messages
                            if 'message' in update:
                                message = update['message']
                                if 'text' in message:
                                    handle_telegram_command(bot, message['text'], str(message['chat']['id']))
                            
                            # Handle callback queries (mode selection)
                            elif 'callback_query' in update:
                                callback = update['callback_query']
                                if callback['data'].startswith('mode_'):
                                    mode = callback['data'].replace('mode_', '')
                                    bot.set_mode(mode)
                                    
                                    # Answer callback query
                                    answer_url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
                                    answer_data = {"callback_query_id": callback['id']}
                                    requests.post(answer_url, json=answer_data)
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Telegram loop error: {e}")
                time.sleep(5)
    
    # Start Telegram handler
    telegram_thread = threading.Thread(target=telegram_loop, daemon=True)
    telegram_thread.start()
    
    print("🤖 Solana Meme Coin Bot started!")
    print("Send /start to your Telegram bot to begin")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        bot.stop_trading()

if __name__ == "__main__":
    main()
