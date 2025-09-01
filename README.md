# Solana Meme Snipe Paper-Trade Bot

A sophisticated paper-trading bot designed to automatically identify and trade Solana meme tokens using multiple strategies and filters. The bot focuses on catching early momentum in new meme token launches while managing risk through configurable parameters.

## 🎯 Features

### Three-Tier Buy Strategy
1. **Launch Snipe**: Catches brand-new pairs on Raydium/Pump/Meteora/Bunk with optional RugCheck and 2x momentum predictor
2. **Quantum Scorer**: Advanced scoring system that evaluates tokens based on multiple factors (cap, age, buys, velocity, etc.)
3. **Base Filter**: Configurable logic across liquidity, volume, and market cap with multiple modes (OR/AND/MAJORITY/HYBRID)

### Risk Management
- **Paper Trading Only**: No real money at risk
- **Position Limits**: Maximum 6 concurrent positions
- **Take Profit/Stop Loss**: Configurable exit strategies
- **Trailing Stops**: Dynamic position management
- **Cooldown Periods**: Prevents overtrading

### Advanced Features
- **2x Momentum Predictor**: Uses price momentum to predict doubling potential
- **RugCheck Integration**: Optional security scoring via rugcheck.xyz API
- **Social Media Monitoring**: Twitter integration for sentiment-based trading
- **Telegram Control**: Full bot control via Telegram commands and buttons

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Telegram Bot Token
- (Optional) Twitter API Bearer Token

### Installation
```bash
git clone <repository>
cd solana-meme-bot
pip install -r requirements.txt
```

### Configuration
1. **Telegram Setup**:
   - Create a bot via @BotFather
   - Get your bot token and chat ID
   - Update `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `main.py`

2. **Optional Twitter Setup**:
   - Get Twitter API Bearer Token
   - Add accounts to monitor via `/twadd @handle`

### Running the Bot
```bash
python main.py
```

## 📊 Key Commands

### Basic Control
- `/start` - Initialize bot and show status
- `/status` - Show current bot configuration and performance
- `/pause` - Pause new buys
- `/resume` - Resume trading
- `/positions` - List open positions
- `/closeall` - Close all positions

### Filter Configuration
- `/setcap 10000 5000000` - Set market cap range (USD)
- `/setliq 1500` - Set minimum liquidity (USD)
- `/setvol 250` - Set minimum volume (USD)
- `/setvolwin 10` - Set volume window (minutes)
- `/filtermode HYBRID` - Set filter logic mode

### Advanced Features
- `/quantum on 60` - Enable quantum scorer with threshold
- `/2x on 2 0.8` - Enable 2x predictor (2min horizon, 0.8 alpha)
- `/launch on` - Enable launch snipe
- `/rugcheck lenient 60` - Set rugcheck mode and score threshold

### Diagnostics
- `/diag` - Show filter diagnostics
- `/qparams` - Show quantum scorer parameters
- `/panel` - Show control buttons

## 🔧 Configuration Modes

### Filter Logic Modes
- **OR**: Any condition (liquidity OR volume OR cap) passes
- **AND**: All conditions must pass
- **MAJORITY**: At least N conditions must pass (configurable)
- **HYBRID**: Stricter for new pairs, majority for older ones

### Quantum Scorer Weights
The quantum scorer evaluates tokens based on:
- **Cap Score** (30%): Prefers smaller market caps
- **Age Score** (15%): Newer pairs score higher
- **Buy Activity** (20%): More buys in last 5 minutes
- **Buy/Sell Ratio** (15%): Favor buying pressure
- **Velocity** (15%): Volume/liquidity ratio
- **Venue Bonus** (3%): Preferred DEX bonus
- **Social Bonus** (2%): Recent social mentions

## 📈 Trading Strategy

### Entry Conditions
1. **Meme Token Focus**: Only Solana pairs with SOL/USDC/USDT quotes
2. **Excluded Tokens**: Major tokens (SOL, USDT, USDC, BTC, ETH, etc.)
3. **Minimum Requirements**: Configurable liquidity, volume, and market cap thresholds

### Exit Strategy
- **Take Profit**: 60% gain (configurable)
- **Stop Loss**: 16% loss (configurable)
- **Trailing Stop**: 5% from highest price (configurable)
- **Minimum Hold**: 5 seconds (configurable)

### Position Sizing
- **Fixed Amount**: 2.5 SOL per trade (default)
- **Percentage**: 12% of balance (alternative mode)
- **Maximum Positions**: 6 concurrent trades

## 🔍 Data Sources

- **DexScreener API**: Real-time pair data and metrics
- **RugCheck API**: Security scoring for token contracts
- **Twitter API**: Social sentiment monitoring
- **Telegram Bot API**: User interface and control

## ⚙️ Advanced Configuration

### Momentum Predictor
The 2x predictor uses exponential growth modeling:
- **Horizon**: Time window for doubling prediction (default: 2 minutes)
- **Alpha**: Conservative factor (default: 0.8)
- **Window**: Price comparison window (default: 60 seconds)

### Social Monitoring
- **Positive Keywords**: "buy", "sending", "moon", "bullish", "ape", "pump"
- **Account Tracking**: Monitor specific Twitter accounts
- **Mint/Ticker Extraction**: Parse Solana addresses and ticker symbols

### Risk Controls
- **Buy Cooldown**: 120 seconds between buys
- **Global Cooldown**: 15 seconds between any actions
- **Network Backoff**: Exponential backoff on API failures
- **Debug Mode**: Detailed logging of skipped trades

## 📁 File Structure

```
├── main.py              # Main bot code
├── settings.json        # Persistent configuration
├── state.json          # Trading state and positions
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🛡️ Safety Features

- **Paper Trading Only**: No real transactions
- **Rate Limiting**: Prevents API abuse
- **Error Handling**: Graceful degradation on failures
- **State Persistence**: Survives restarts
- **Authorization**: Telegram chat ID verification

## 📊 Performance Tracking

The bot tracks:
- **Balance**: Current SOL balance
- **Realized PnL**: Closed position profits/losses
- **Open Positions**: Active trades with entry prices
- **Trade History**: All completed trades with reasons

## 🔧 Customization

### Adding New Filters
1. Modify `is_meme_candidate()` for token filtering
2. Add new conditions to `_base_conditions()`
3. Update filter logic in `filter_pairs_logic()`

### Adding New Data Sources
1. Create new fetch function
2. Integrate into main trading loop
3. Add configuration options to `DEFAULT_SETTINGS`

### Modifying Scoring
1. Adjust weights in quantum scorer
2. Add new scoring factors
3. Update normalization parameters

## ⚠️ Disclaimer

This bot is for educational and research purposes only. It performs paper trading and does not execute real transactions. Cryptocurrency trading involves substantial risk of loss. Use at your own risk.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
