#!/usr/bin/env python3
"""
Simple startup script for Solbot with enhanced no-rebuy system
"""

import subprocess
import sys
import time

def main():
    print("🚀 Starting Solbot with No-Rebuy System...")
    print("✅ Features enabled:")
    print("   • No-rebuy prevention (tracks sold tokens)")
    print("   • Wallet tracking with ROI filtering")
    print("   • Improved ultra mode for micro-caps")
    print("   • Enhanced error handling")
    print()
    
    try:
        # Run the main bot
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"🚨 Bot crashed: {e}")
        print("Restarting in 5 seconds...")
        time.sleep(5)
        main()  # Restart

if __name__ == "__main__":
    main()