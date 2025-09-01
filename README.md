Solana Meme Snipe Paper-Trade Bot (Dexscreener)

Prereqs
- Python 3.10+

Setup
1) (Optional) Create a virtualenv
   python3 -m venv .venv
   source .venv/bin/activate

2) Install dependencies
   pip install -U pip
   pip install -r requirements.txt

3) Configure Telegram (optional but recommended)
   export TELEGRAM_BOT_TOKEN=123456:ABC...
   export TELEGRAM_CHAT_ID=5794143622

4) Run
   python3 main.py

Notes
- If TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are not set, the Telegram loop is disabled.
- The bot is paper-trading only; it never places real trades.
- Settings persist in settings.json; runtime state in state.json.
# Solbot
Botaarroooo
