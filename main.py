"""BTC race bot: oversold-dip mean reversion on cbBTC 1-minute bars.

Strategy (long only, one asset, CBBTC vs USDC, $1,000 paper money):
  - Wait for a sharp, oversold dip: the latest close is at least 2% below the
    highest close of the last 60 bars AND the 14-bar RSI is below 30.
  - Then buy with all available cash.
  - Exit the whole position when the close rises 1.0% above the actual fill
    price (take profit), falls 0.6% below the highest close since entry
    (trailing stop), or 360 bars (6 hours) have passed since the buy filled
    without either trigger (max hold: a bounce that never comes should not
    be ridden into the close-out).
  - Otherwise stay in cash. No shorts, no leverage, no martingale.

Why this shape: a full round trip costs about 30 bps (4 bps fee + 2 bps
slippage + ~10 bps price impact per leg at full size), so the bot only
trades on a move large enough to pay for itself, and it trades rarely
(a handful of times per day at most). The 2% dip + RSI<30 filter keeps it
out of slow grinds; the trailing stop keeps a winning bounce from being
given back.

Determinism: no clock, no network, no randomness, no environment reads.
All decisions come from the bars the harness sends. Integer math is used
for the entry/exit thresholds; the RSI uses plain floats over a fixed
window, which is deterministic in CPython.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- strategy parameters (few, round, chosen on the published dev window) ---
LOOKBACK_BARS = 60        # bars for the trailing high
DIP_BPS = 200             # buy only 2.00% below the trailing high
RSI_BARS = 14             # bars for the RSI
RSI_MAX = 30              # buy only while RSI is below 30 (oversold)
TAKE_PROFIT_BPS = 100     # sell 1.00% above the fill price
TRAIL_BPS = 60            # sell 0.60% below the peak close since entry
MAX_HOLD_BARS = 360       # sell 6 hours after the buy filled, whatever the price
MAX_SLIPPAGE_BPS = 200    # headroom for a 1-minute gap; modelled cost is ~12 bps
MIN_NOTIONAL = 1_000_000  # 1 USDC in quote units


class DipBot:
    """Keeps the bar history and the open trade's fill price and peak."""

    def __init__(self):
        self.closes = []        # CBBTC closes (ints at price_scale), in order
        self.entry_price = 0    # fill price of the open buy (0 when flat)
        self.peak = 0           # highest close since the open buy
        self.entry_bar = -1     # seq of the batch that decided the open buy

    def rsi(self):
        """RSI over the last RSI_BARS one-minute changes (simple means)."""
        n = RSI_BARS
        if len(self.closes) < n + 1:
            return 50.0
        gains = 0.0
        losses = 0.0
        tail = self.closes[-n - 1:]
        for i in range(1, len(tail)):
            d = tail[i] - tail[i - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
        avg_gain = gains / n
        avg_loss = losses / n
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    def on_batch(self, state, batch):
        # Fold in the fills of the previous batch first: a filled buy sets
        # the entry price we measure the take-profit from.
        for fill in batch.get("fills", []):
            if fill.get("symbol") == SYMBOL and fill.get("status") == "filled":
                if fill.get("side") == "buy":
                    self.entry_price = int(fill["price"])
                    self.peak = self.entry_price
                    self.entry_bar = state.seq
                elif fill.get("side") == "sell":
                    self.entry_price = 0
                    self.peak = 0
                    self.entry_bar = -1

        close = state.close(SYMBOL)
        if close is None:
            return []
        # Only trust this close when the batch actually carried a fresh bar
        # for CBBTC; otherwise the shim hands back the previous bar's close,
        # and re-appending it would double-count one minute.
        if not any(b.get("symbol") == SYMBOL for b in batch.get("bars", [])):
            return []
        self.closes.append(close)

        # Never trade during warm-up: those orders are rejected anyway.
        if state.warming_up:
            return []

        holding = state.positions.get(SYMBOL)
        qty = holding.qty if holding is not None else 0

        # If we hold a position but lost the fill price (should not happen),
        # rebuild it from the position's cost so the exits still work.
        if qty > 0 and self.entry_price == 0 and holding is not None:
            sym = state.symbols[SYMBOL]
            self.entry_price = (
                holding.cost
                * 10 ** (sym["qty_scale"] + sym["price_scale"])
                // (qty * 10 ** state.hello["capital"]["quote_scale"])
            )
            self.peak = self.entry_price

        if qty > 0:
            if close > self.peak:
                self.peak = close
            # Take profit: 1.00% above the fill price.
            if close * 10_000 >= self.entry_price * (10_000 + TAKE_PROFIT_BPS):
                return self._sell_all(state, qty)
            # Trailing stop: 0.60% below the peak close since entry.
            if close * 10_000 <= self.peak * (10_000 - TRAIL_BPS):
                return self._sell_all(state, qty)
            # Max hold: 6 hours after the buy filled, exit whatever the price.
            if self.entry_bar >= 0 and state.seq - self.entry_bar >= MAX_HOLD_BARS:
                return self._sell_all(state, qty)
            return []

        # Flat: look for an oversold dip.
        if len(self.closes) < LOOKBACK_BARS or len(self.closes) < RSI_BARS + 1:
            return []
        high = max(self.closes[-LOOKBACK_BARS:])
        if close * 10_000 > high * (10_000 - DIP_BPS):
            return []
        if self.rsi() >= RSI_MAX:
            return []
        if state.cash < MIN_NOTIONAL:
            return []
        return [
            {
                "side": "buy",
                "symbol": SYMBOL,
                "notional_quote_units": state.cash,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }
        ]

    def _sell_all(self, state, qty):
        return [
            {
                "side": "sell",
                "symbol": SYMBOL,
                "qty": qty,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }
        ]


def main():
    bot = DipBot()
    makefun_bars.run(bot.on_batch)


if __name__ == "__main__":
    main()
