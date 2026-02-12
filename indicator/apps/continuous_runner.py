#!/usr/bin/env python3
"""
Continuous Market Analysis Runner

Live market analysis with rolling windows and state machine.
Streams data continuously, computes signals, and outputs trade decisions.

Usage:
    python continuous_runner.py                    # Default: BTCUSDT
    python continuous_runner.py ETHUSDT            # Specific symbol
    python continuous_runner.py BTCUSDT --quiet    # Minimal output
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from indicator.continuous import (
    AnalyzerConfig,
    ContinuousAnalyzer,
    MarketRegime,
    MarketState,
    StateTransition,
    TradeSignal,
)
from indicator.display.colors import Colors

# =============================================================================
# TRADE TRACKER - Simulated P&L from trade signals
# =============================================================================


@dataclass
class SimulatedTrade:
    """A single simulated trade from entry to exit."""

    trade_num: int
    direction: str  # "long" or "short"
    signal_type: str  # "breakout", "squeeze", "trend", "reversal"
    entry_price: float
    entry_time: float  # unix timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    fee_percent: float = 0.04  # 0.02% per side (entry + exit = 0.04% total)

    # Exit info (filled when closed)
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    exit_reason: str = ""  # "stop_loss", "take_profit", "signal_reversal", "open"

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def pnl_percent(self) -> Optional[float]:
        """P&L as percentage of entry price, after trading fees."""
        if self.exit_price is None:
            return None

        # Calculate raw P&L
        if self.direction == "long":
            raw_pnl = (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            raw_pnl = (self.entry_price - self.exit_price) / self.entry_price * 100

        # Subtract round-trip trading fees (0.02% entry + 0.02% exit = 0.04%)
        return raw_pnl - self.fee_percent

    @property
    def is_winner(self) -> Optional[bool]:
        pnl = self.pnl_percent
        if pnl is None:
            return None
        return pnl > 0

    def unrealized_pnl_percent(self, current_price: float) -> float:
        """Unrealized P&L for open trades, including fees already paid on entry."""
        # Calculate raw P&L
        if self.direction == "long":
            raw_pnl = (current_price - self.entry_price) / self.entry_price * 100
        else:
            raw_pnl = (self.entry_price - current_price) / self.entry_price * 100

        # Subtract fees: entry fee already paid + potential exit fee
        return raw_pnl - self.fee_percent


class TradeTracker:
    """
    Tracks simulated trades from TradeSignal events.

    Simulates what would happen if you took every signal:
    - Opens a trade on each TradeSignal
    - Closes on stop loss, take profit, or opposing signal
    - Tracks cumulative P&L, win rate, etc.
    """

    def __init__(self):
        self._trades: List[SimulatedTrade] = []
        self._trade_count = 0

    @property
    def trades(self) -> List[SimulatedTrade]:
        return self._trades

    @property
    def total_trades(self) -> int:
        return self._trade_count

    @property
    def closed_trades(self) -> List[SimulatedTrade]:
        return [t for t in self._trades if not t.is_open]

    @property
    def open_trade(self) -> Optional[SimulatedTrade]:
        for t in reversed(self._trades):
            if t.is_open:
                return t
        return None

    def handle_signal(self, signal: TradeSignal) -> None:
        """Process a new trade signal."""
        now = time.time()

        # Check if we have an open trade
        open_trade = self.open_trade
        if open_trade is not None:
            # Same direction signal: update the existing trade instead of closing/reopening
            # This prevents inflating "signal reversals" when the state machine is just
            # reconfirming the same trend (e.g., TREND_LONG → TREND_LONG).
            if open_trade.direction == signal.direction:
                # Update stop/target if the new signal has them
                if signal.stop_loss is not None:
                    open_trade.stop_loss = signal.stop_loss
                if signal.take_profit is not None:
                    open_trade.take_profit = signal.take_profit
                # Update confidence to the higher of the two
                open_trade.confidence = max(open_trade.confidence, signal.confidence)
                return

            # Opposite direction signal: close and reverse
            open_trade.exit_price = signal.entry_price
            open_trade.exit_time = now
            open_trade.exit_reason = "signal_reversal"

        # Open new trade
        self._trade_count += 1
        trade = SimulatedTrade(
            trade_num=self._trade_count,
            direction=signal.direction,
            signal_type=signal.signal_type,
            entry_price=signal.entry_price,
            entry_time=now,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
        )
        self._trades.append(trade)

    def check_exits(self, current_price: float) -> Optional[SimulatedTrade]:
        """Check if any open trade hit stop loss or take profit."""
        open_trade = self.open_trade
        if open_trade is None:
            return None

        now = time.time()

        # Check stop loss
        if open_trade.stop_loss is not None:
            if open_trade.direction == "long" and current_price <= open_trade.stop_loss:
                open_trade.exit_price = open_trade.stop_loss
                open_trade.exit_time = now
                open_trade.exit_reason = "stop_loss"
                return open_trade
            elif open_trade.direction == "short" and current_price >= open_trade.stop_loss:
                open_trade.exit_price = open_trade.stop_loss
                open_trade.exit_time = now
                open_trade.exit_reason = "stop_loss"
                return open_trade

        # Check take profit
        if open_trade.take_profit is not None:
            if open_trade.direction == "long" and current_price >= open_trade.take_profit:
                open_trade.exit_price = open_trade.take_profit
                open_trade.exit_time = now
                open_trade.exit_reason = "take_profit"
                return open_trade
            elif open_trade.direction == "short" and current_price <= open_trade.take_profit:
                open_trade.exit_price = open_trade.take_profit
                open_trade.exit_time = now
                open_trade.exit_reason = "take_profit"
                return open_trade

        return None

    def get_stats(self, current_price: float) -> dict:
        """Get trade performance statistics."""
        closed = self.closed_trades
        open_trade = self.open_trade

        winners = [t for t in closed if t.is_winner]
        losers = [t for t in closed if t.is_winner is False]

        # Compounded realized P&L: multiply (1 + r) for each trade
        equity = 1.0
        for t in closed:
            pnl = t.pnl_percent
            if pnl is not None:
                equity *= 1 + pnl / 100
        compounded_pnl = (equity - 1) * 100

        win_pnls = [t.pnl_percent for t in winners if t.pnl_percent is not None]
        loss_pnls = [t.pnl_percent for t in losers if t.pnl_percent is not None]

        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0

        # Unrealized P&L on current equity
        unrealized_pct = open_trade.unrealized_pnl_percent(current_price) if open_trade else 0.0
        net_equity = equity * (1 + unrealized_pct / 100)
        net_pnl = (net_equity - 1) * 100

        # By exit reason
        stop_losses = len([t for t in closed if t.exit_reason == "stop_loss"])
        take_profits = len([t for t in closed if t.exit_reason == "take_profit"])
        reversals = len([t for t in closed if t.exit_reason == "signal_reversal"])

        return {
            "total_signals": self._trade_count,
            "closed": len(closed),
            "open": 1 if open_trade else 0,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(closed) * 100 if closed else 0.0,
            "total_pnl": compounded_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "unrealized_pnl": unrealized_pct,
            "net_pnl": net_pnl,
            "stop_losses": stop_losses,
            "take_profits": take_profits,
            "reversals": reversals,
            "open_trade": open_trade,
        }


class ContinuousDisplay:
    """Terminal display for continuous analysis."""

    def __init__(self, quiet: bool = False, show_metrics: bool = False):
        self.quiet = quiet
        self.show_metrics = show_metrics
        self._last_state: MarketRegime = MarketRegime.NO_TRADE
        self._trade_count = 0
        self._start_time = time.time()
        self._metrics_interval = 0  # Counter for metrics display
        self.trade_tracker = TradeTracker()

    def print_header(self, symbol: str) -> None:
        """Print startup header."""
        print()
        print(
            f"{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}"
        )
        print(
            f"{Colors.BOLD}{Colors.CYAN}║         CONTINUOUS MARKET ANALYSIS                            ║{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.CYAN}║         {symbol:^50}   ║{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.CYAN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}"
        )
        print()
        print(
            f"{Colors.DIM}Architecture: aggTrades(ws) → Rolling Windows → Engines → State Machine{Colors.RESET}"
        )
        print(f"{Colors.DIM}Windows: 15s | 60s | 180s | 900s | 3600s{Colors.RESET}")
        print()

    def print_status(self, analyzer: ContinuousAnalyzer) -> None:
        """Print current status line."""
        if self.quiet:
            return

        status = analyzer.get_status()
        signals = analyzer.get_signals_summary()

        # Time running
        elapsed = time.time() - self._start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        # State with color
        state = status["current_state"]
        state_color = self._state_color(MarketRegime(state))

        # Price
        price = status["latest_price"]
        price_str = f"${price:,.2f}" if price else "---"

        # Score
        score = status["unified_score"]
        score_str = f"{score:+.2f}" if score is not None else "---"
        score_color = (
            Colors.GREEN
            if score and score > 0.3
            else Colors.RED if score and score < -0.3 else Colors.RESET
        )

        # Confidence
        conf = status["confidence"]
        conf_str = f"{conf:.0f}%" if conf is not None else "---"

        # Data quality
        quality = status["data_quality"]
        quality_bar = self._quality_bar(quality) if quality else "---"

        # Delta ratio from volume signal
        delta = signals["volume"]["delta_ratio"]
        delta_str = f"{delta:+.1%}" if delta is not None else "---"
        delta_color = (
            Colors.GREEN
            if delta and delta > 0.1
            else Colors.RED if delta and delta < -0.1 else Colors.RESET
        )

        # Check stop loss / take profit exits
        if price:
            closed = self.trade_tracker.check_exits(price)
            if closed:
                pnl = closed.pnl_percent or 0.0
                pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
                reason = closed.exit_reason.replace("_", " ").upper()
                print(
                    f"\n  {Colors.BOLD}{pnl_color}TRADE #{closed.trade_num} CLOSED ({reason}): "
                    f"{pnl:+.2f}%{Colors.RESET}"
                )

        # Trade tracker summary
        trade_str = ""
        if self.trade_tracker.total_trades > 0:
            stats = self.trade_tracker.get_stats(price or 0)
            net = stats["net_pnl"]
            net_color = Colors.GREEN if net >= 0 else Colors.RED
            trade_str = (
                f" │ Trades: {stats['total_signals']} "
                f"│ P&L: {net_color}{net:+.2f}%{Colors.RESET}"
            )

        # Clear line and print
        print(
            f"\r{Colors.DIM}[{elapsed_str}]{Colors.RESET} "
            f"{state_color}{state:20}{Colors.RESET} "
            f"│ {price_str:>10} "
            f"│ Score: {score_color}{score_str:>6}{Colors.RESET} "
            f"│ Δ: {delta_color}{delta_str:>6}{Colors.RESET} "
            f"│ Conf: {conf_str:>4} "
            f"│ {quality_bar}{trade_str}",
            end="",
            flush=True,
        )

    def print_state_change(self, transition: StateTransition) -> None:
        """Print state transition."""
        from_color = self._state_color(transition.from_state)
        to_color = self._state_color(transition.to_state)

        timestamp = datetime.fromtimestamp(transition.timestamp_ms / 1000).strftime("%H:%M:%S")

        print()  # New line after status
        print(f"\n{Colors.BOLD}[{timestamp}] STATE CHANGE:{Colors.RESET}")
        print(
            f"  {from_color}{transition.from_state.value}{Colors.RESET} → {to_color}{transition.to_state.value}{Colors.RESET}"
        )
        print(
            f"  {Colors.DIM}Trigger: {transition.trigger} | Confidence: {transition.confidence:.1f}%{Colors.RESET}"
        )

    def print_trade_signal(self, signal: TradeSignal) -> None:
        """Print trade signal with emphasis."""
        self._trade_count += 1
        self.trade_tracker.handle_signal(signal)

        timestamp = datetime.fromtimestamp(signal.timestamp_ms / 1000).strftime("%H:%M:%S")

        # Direction color
        if signal.direction == "long":
            dir_color = Colors.GREEN
            arrow = "▲"
        else:
            dir_color = Colors.RED
            arrow = "▼"

        print()
        print(f"\n{Colors.BOLD}{dir_color}{'═' * 60}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{dir_color}  {arrow} TRADE SIGNAL #{self._trade_count}: {signal.direction.upper()} {signal.signal_type.upper()}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{dir_color}{'═' * 60}{Colors.RESET}")
        print(f"  {Colors.BOLD}Entry:{Colors.RESET}  ${signal.entry_price:,.2f}")
        if signal.stop_loss:
            print(f"  {Colors.BOLD}Stop:{Colors.RESET}   ${signal.stop_loss:,.2f}")
        if signal.take_profit:
            print(f"  {Colors.BOLD}Target:{Colors.RESET} ${signal.take_profit:,.2f}")
        print(f"  {Colors.BOLD}Confidence:{Colors.RESET} {signal.confidence:.1f}%")
        print(f"  {Colors.BOLD}Risk:{Colors.RESET} {signal.risk_level}")

        # Confirmations
        confirms = []
        if signal.volume_confirms:
            confirms.append(f"{Colors.GREEN}✓ Volume{Colors.RESET}")
        else:
            confirms.append(f"{Colors.DIM}✗ Volume{Colors.RESET}")
        if signal.book_confirms:
            confirms.append(f"{Colors.GREEN}✓ Book{Colors.RESET}")
        else:
            confirms.append(f"{Colors.DIM}✗ Book{Colors.RESET}")
        if signal.oi_confirms:
            confirms.append(f"{Colors.GREEN}✓ OI{Colors.RESET}")
        else:
            confirms.append(f"{Colors.DIM}✗ OI{Colors.RESET}")

        print(f"  {Colors.BOLD}Confirmations:{Colors.RESET} {' | '.join(confirms)}")

        if signal.warnings:
            print(f"  {Colors.YELLOW}⚠ Warnings:{Colors.RESET}")
            for warning in signal.warnings:
                print(f"    - {warning}")

        print(f"{dir_color}{'═' * 60}{Colors.RESET}\n")

    def print_market_state(self, state: MarketState) -> None:
        """Print detailed market state (verbose mode)."""
        if self.quiet:
            return

        # Only print occasionally to avoid spam
        # This would typically be called less frequently

    def _state_color(self, state: MarketRegime) -> str:
        """Get color for state."""
        bullish_states = [
            MarketRegime.SQUEEZE_SETUP_LONG,
            MarketRegime.TREND_CONTINUATION_LONG,
            MarketRegime.BREAKOUT_LONG,
            MarketRegime.REVERSAL_LONG,
            MarketRegime.ACCUMULATION,
        ]
        bearish_states = [
            MarketRegime.SQUEEZE_SETUP_SHORT,
            MarketRegime.TREND_CONTINUATION_SHORT,
            MarketRegime.BREAKOUT_SHORT,
            MarketRegime.REVERSAL_SHORT,
            MarketRegime.DISTRIBUTION,
        ]
        caution_states = [
            MarketRegime.EXHAUSTION_LONG,
            MarketRegime.EXHAUSTION_SHORT,
            MarketRegime.COMPRESSION,
        ]

        if state in bullish_states:
            return Colors.GREEN
        elif state in bearish_states:
            return Colors.RED
        elif state in caution_states:
            return Colors.YELLOW
        elif state == MarketRegime.CHOPPY:
            return Colors.MAGENTA
        else:
            return Colors.DIM

    def _quality_bar(self, quality: float) -> str:
        """Create data quality bar."""
        filled = int(quality * 5)
        bar = "█" * filled + "░" * (5 - filled)

        if quality >= 0.8:
            color = Colors.GREEN
        elif quality >= 0.5:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        return f"{color}{bar}{Colors.RESET}"

    def print_metrics(self, analyzer: ContinuousAnalyzer) -> None:
        """Print latency metrics periodically."""
        if self.quiet or not self.show_metrics:
            return

        self._metrics_interval += 1
        if self._metrics_interval < 10:  # Print every 10 seconds
            return
        self._metrics_interval = 0

        metrics = analyzer.get_metrics_summary()
        latencies = metrics.get("latencies_ms", {})
        ingestion = metrics.get("ingestion", {})

        print(f"\n\n{Colors.CYAN}{'─' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}METRICS SNAPSHOT{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")

        # Ingestion rates
        print(f"\n{Colors.BOLD}Ingestion:{Colors.RESET}")
        print(f"  Trades/sec: {ingestion.get('trades_per_second', '0')}")
        print(f"  Orderbook updates/sec: {ingestion.get('orderbook_updates_per_second', '0')}")
        print(f"  Total trades: {ingestion.get('total_trades', 0):,}")

        # Latencies
        print(f"\n{Colors.BOLD}Signal Latencies (ms):{Colors.RESET}")
        signal_lat = latencies.get("signal_compute", {})
        print(
            f"  Overall:  mean={signal_lat.get('mean', '0')} p95={signal_lat.get('p95', '0')} max={signal_lat.get('max', '0')}"
        )

        for name in ["volume_engine", "book_engine", "unified_score"]:
            lat = latencies.get(name, {})
            if lat.get("mean"):
                display_name = name.replace("_", " ").title()
                print(f"  {display_name:12}: mean={lat.get('mean', '0')} p95={lat.get('p95', '0')}")

        # State machine stats
        sm = metrics.get("state_machine", {})
        print(f"\n{Colors.BOLD}State Machine:{Colors.RESET}")
        print(f"  Transitions: {sm.get('transitions', 0)}")
        print(f"  Trade signals: {sm.get('trade_signals', 0)}")

        print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}\n")


class DeepDiveDisplay:
    """
    Deep-dive display mode that shows full analysis sections
    like in analyze.py, but continuously updated.
    """

    def __init__(self, refresh_interval: int = 5):
        self.refresh_interval = refresh_interval  # seconds between refreshes
        self._last_refresh = 0
        self._iteration = 0
        self.trade_tracker = TradeTracker()

    def should_refresh(self) -> bool:
        """Check if it's time to refresh the deep dive display."""
        import time

        now = time.time()
        if now - self._last_refresh >= self.refresh_interval:
            self._last_refresh = now
            return True
        return False

    def print_deep_dive(self, analyzer: ContinuousAnalyzer) -> None:
        """Print full deep-dive analysis sections."""
        if not self.should_refresh():
            return

        # Check if analyzer is warmed up
        if not analyzer.is_warmed_up:
            warmup = analyzer.warmup_progress
            elapsed = warmup.get("elapsed_seconds", 0)
            required = warmup.get("required_seconds", 60)
            trades = warmup.get("trade_count", 0)
            required_trades = warmup.get("required_trades", 100)

            print(f"\n{Colors.YELLOW}⏳ WARMING UP - Collecting initial data...{Colors.RESET}")
            print(f"   Time: {elapsed}/{required}s")
            print(f"   Trades: {trades}/{required_trades}")
            print(
                f"   {Colors.DIM}Please wait for sufficient data before analysis begins{Colors.RESET}\n"
            )
            return

        self._iteration += 1

        # Import display functions
        from indicator.display import (
            print_funding_deep_dive,
            print_header,
            print_oi_deep_dive,
            print_orderbook_deep_dive,
            print_unified_score,
            print_volume_deep_dive,
            print_volume_engine_deep_dive,
        )

        # Get current state
        status = analyzer.get_status()
        current_price = status["latest_price"]
        state = status["current_state"]
        unified_score = status["unified_score"]

        # Get full results
        results = analyzer.get_all_full_results()
        volume_analysis = results["volume_analysis"]
        volume_engine = results["volume_engine"]
        oi_analysis = results["oi"]
        funding_analysis = results["funding"]
        orderbook_analysis = results["orderbook"]
        unified_score_obj = results["unified_score"]

        # Calculate price change
        price_change_pct = 0.0
        price_history = analyzer.get_recent_price_history(30)
        if len(price_history) >= 2:
            price_change_pct = (price_history[-1] - price_history[0]) / price_history[0] * 100

        # Clear screen for clean refresh
        print("\033[2J\033[H")  # ANSI clear screen

        # Print header
        print_header(analyzer.symbol, current_price, price_change_pct, f"LIVE (#{self._iteration})")

        # Print deep dive sections if data available
        if volume_analysis:
            print_volume_deep_dive(volume_analysis)

        if volume_engine:
            print_volume_engine_deep_dive(volume_engine)

        if oi_analysis:
            print_oi_deep_dive(oi_analysis)

        if funding_analysis:
            oi_change_for_display = (
                oi_analysis.rate_of_change.oi_change_percent if oi_analysis else None
            )
            print_funding_deep_dive(funding_analysis, oi_change_for_display)

        if orderbook_analysis:
            print_orderbook_deep_dive(orderbook_analysis)

        if unified_score_obj:
            print_unified_score(unified_score_obj)

        # Print ATR expansion (timing gate)
        atr_signals = results.get("atr_signals")
        if atr_signals:
            from continuous.atr_expansion_adapter import format_atr_signals

            print()
            print(f"{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
            print(
                f"{Colors.BOLD}{Colors.CYAN}│  ATR EXPANSION - Volatility Timing{' ' * 46}│{Colors.RESET}"
            )
            print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")
            print()
            formatted = format_atr_signals(atr_signals)
            for line in formatted.split("\n"):
                print(f"  {line}")
            print()

        # Print current state and confidence
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}│  CURRENT STATE{' ' * 63}│{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")
        print(f"  {Colors.BOLD}Regime:{Colors.RESET} {state}")
        print(
            f"  {Colors.BOLD}Score:{Colors.RESET} {unified_score:+.2f}"
            if unified_score is not None
            else "  Score: ---"
        )
        print(
            f"  {Colors.BOLD}Confidence:{Colors.RESET} {status['confidence']:.0f}%"
            if status["confidence"] is not None
            else "  Confidence: ---"
        )
        print(f"  {Colors.DIM}Last updated: {datetime.now().strftime('%H:%M:%S')}{Colors.RESET}")
        print()

        # Check stop loss / take profit exits
        if current_price:
            self.trade_tracker.check_exits(current_price)

        # Print trade performance section
        self._print_trade_performance(current_price or 0)

    def _print_trade_performance(self, current_price: float) -> None:
        """Print trade performance summary section."""
        stats = self.trade_tracker.get_stats(current_price)

        print(f"{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.CYAN}│  TRADE PERFORMANCE (Simulated){' ' * 47}│{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")

        if stats["total_signals"] == 0:
            print(
                f"  {Colors.DIM}No trade signals generated yet. Waiting for setups...{Colors.RESET}"
            )
            print()
            return

        # Overview row
        net_pnl = stats["net_pnl"]
        net_color = Colors.GREEN if net_pnl >= 0 else Colors.RED
        total_pnl = stats["total_pnl"]
        total_color = Colors.GREEN if total_pnl >= 0 else Colors.RED

        print()
        print(
            f"  {Colors.BOLD}Total Signals:{Colors.RESET} {stats['total_signals']}  "
            f"│  {Colors.BOLD}Closed:{Colors.RESET} {stats['closed']}  "
            f"│  {Colors.BOLD}Open:{Colors.RESET} {stats['open']}"
        )
        print()

        # Win/Loss breakdown
        if stats["closed"] > 0:
            win_rate = stats["win_rate"]
            wr_color = Colors.GREEN if win_rate >= 50 else Colors.RED

            # Win rate bar
            bar_width = 30
            filled = int(win_rate / 100 * bar_width)
            win_bar = (
                f"{Colors.GREEN}{'█' * filled}{Colors.RESET}"
                f"{Colors.RED}{'█' * (bar_width - filled)}{Colors.RESET}"
            )

            print(
                f"  {Colors.BOLD}Win Rate:{Colors.RESET}  {wr_color}{win_rate:.0f}%{Colors.RESET} "
                f"({stats['winners']}W / {stats['losers']}L)"
            )
            print(f"             {win_bar}")
            print()

            # P&L details
            print(
                f"  {Colors.BOLD}Realized P&L:{Colors.RESET}   {total_color}{total_pnl:+.2f}%{Colors.RESET}"
            )
            if stats["avg_win"] > 0:
                print(
                    f"  {Colors.BOLD}Avg Winner:{Colors.RESET}     {Colors.GREEN}{stats['avg_win']:+.2f}%{Colors.RESET}"
                )
            if stats["avg_loss"] < 0:
                print(
                    f"  {Colors.BOLD}Avg Loser:{Colors.RESET}      {Colors.RED}{stats['avg_loss']:+.2f}%{Colors.RESET}"
                )

            # Profit factor
            if stats["avg_loss"] != 0:
                pf = abs(stats["avg_win"] / stats["avg_loss"]) if stats["avg_loss"] != 0 else 0
                pf_color = Colors.GREEN if pf >= 1.0 else Colors.RED
                print(
                    f"  {Colors.BOLD}Profit Factor:{Colors.RESET}  {pf_color}{pf:.2f}{Colors.RESET}"
                )

            # Exit reasons
            print()
            print(f"  {Colors.BOLD}Exit Reasons:{Colors.RESET}")
            if stats["take_profits"] > 0:
                print(f"    {Colors.GREEN}Target Hit:{Colors.RESET}     {stats['take_profits']}")
            if stats["stop_losses"] > 0:
                print(f"    {Colors.RED}Stop Loss:{Colors.RESET}      {stats['stop_losses']}")
            if stats["reversals"] > 0:
                print(f"    {Colors.YELLOW}Signal Flip:{Colors.RESET}    {stats['reversals']}")

        # Open trade details
        open_trade = stats["open_trade"]
        if open_trade is not None:
            unrealized = stats["unrealized_pnl"]
            ur_color = Colors.GREEN if unrealized >= 0 else Colors.RED
            dir_arrow = "▲" if open_trade.direction == "long" else "▼"
            dir_color = Colors.GREEN if open_trade.direction == "long" else Colors.RED

            print()
            print(f"  {Colors.DIM}{'─' * 40}{Colors.RESET}")
            print(
                f"  {Colors.BOLD}Open Position:{Colors.RESET} "
                f"{dir_color}{dir_arrow} {open_trade.direction.upper()}{Colors.RESET} "
                f"({open_trade.signal_type})"
            )
            print(f"    Entry:      ${open_trade.entry_price:,.2f}")
            print(f"    Current:    ${current_price:,.2f}")
            print(f"    Unrealized: {ur_color}{unrealized:+.2f}%{Colors.RESET}")
            if open_trade.stop_loss:
                sl_dist = abs(current_price - open_trade.stop_loss) / current_price * 100
                print(f"    Stop:       ${open_trade.stop_loss:,.2f} ({sl_dist:.2f}% away)")
            if open_trade.take_profit:
                tp_dist = abs(open_trade.take_profit - current_price) / current_price * 100
                print(f"    Target:     ${open_trade.take_profit:,.2f} ({tp_dist:.2f}% away)")

        # Net P&L summary
        print()
        print(f"  {Colors.BOLD}{'─' * 40}{Colors.RESET}")
        print(
            f"  {Colors.BOLD}Net P&L (realized + unrealized): "
            f"{net_color}{net_pnl:+.2f}%{Colors.RESET}"
        )

        # Recent trades list (last 5)
        closed_trades = self.trade_tracker.closed_trades
        if closed_trades:
            recent = closed_trades[-5:]
            print()
            print(f"  {Colors.BOLD}Recent Trades:{Colors.RESET}")
            for t in recent:
                pnl = t.pnl_percent or 0
                pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
                icon = "✓" if pnl >= 0 else "✗"
                dir_sym = "▲" if t.direction == "long" else "▼"
                reason = t.exit_reason.replace("_", " ")
                entry_time = datetime.fromtimestamp(t.entry_time).strftime("%H:%M:%S")
                print(
                    f"    {pnl_color}{icon}{Colors.RESET} #{t.trade_num} "
                    f"{dir_sym} {t.direction:5} "
                    f"${t.entry_price:>10,.2f} → ${t.exit_price:>10,.2f} "
                    f"{pnl_color}{pnl:+.2f}%{Colors.RESET} "
                    f"{Colors.DIM}({reason}){Colors.RESET}"
                )

        print()


async def run_continuous(
    symbol: str,
    quiet: bool = False,
    show_metrics: bool = False,
    deep_dive: bool = False,
    refresh_interval: int = 5,
):
    """Run continuous analysis."""
    if deep_dive:
        display = DeepDiveDisplay(refresh_interval=refresh_interval)
    else:
        display = ContinuousDisplay(quiet=quiet, show_metrics=show_metrics)
        display.print_header(symbol)

    # Create analyzer with config
    config = AnalyzerConfig(
        signal_interval_ms=1000,  # Compute signals every second
        min_confidence=45.0,
        min_volume_ratio=0.6,
        state_cooldown_ms=3000,
        primary_window_seconds=60,
    )

    analyzer = ContinuousAnalyzer(symbol, config=config)

    # Register callbacks
    if not deep_dive:
        analyzer.on_state_change(display.print_state_change)
        analyzer.on_trade_signal(display.print_trade_signal)
    else:
        # Deep-dive mode: track trades silently for the performance section
        analyzer.on_trade_signal(display.trade_tracker.handle_signal)

    print(f"{Colors.DIM}Connecting to streams...{Colors.RESET}")

    try:
        async with analyzer:
            print(f"{Colors.GREEN}Connected! Streaming data...{Colors.RESET}\n")

            # Main display loop
            while True:
                if deep_dive:
                    display.print_deep_dive(analyzer)
                else:
                    display.print_status(analyzer)
                    display.print_metrics(analyzer)
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        # Print final metrics summary on shutdown
        if show_metrics:
            print(f"\n\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
            print(f"{Colors.BOLD}FINAL METRICS SUMMARY{Colors.RESET}")
            metrics = analyzer.get_metrics_summary()
            latencies = metrics.get("latencies_ms", {})
            signal_lat = latencies.get("signal_compute", {})
            print(
                f"Signal compute: mean={signal_lat.get('mean', '0')}ms p99={signal_lat.get('p99', '0')}ms"
            )
            print(
                f"Total trades processed: {metrics.get('ingestion', {}).get('total_trades', 0):,}"
            )
            print(f"State transitions: {metrics.get('state_machine', {}).get('transitions', 0)}")
            print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Continuous market analysis with rolling windows")
    parser.add_argument(
        "symbol", nargs="?", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Minimal output (only trade signals)"
    )
    parser.add_argument(
        "--metrics", "-m", action="store_true", help="Show latency metrics every 10 seconds"
    )
    parser.add_argument(
        "--deep-dive",
        "-d",
        action="store_true",
        help="Deep-dive mode: show full analysis sections like analyze.py",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Deep-dive refresh interval in seconds (min: 2, default: 5)",
    )

    args = parser.parse_args()

    # Normalize symbol
    symbol = args.symbol.upper().replace("/", "").replace("-", "")

    # Enforce minimum refresh interval
    refresh_interval = max(2, args.refresh)
    if args.refresh < 2:
        print(
            f"{Colors.YELLOW}Warning: Minimum refresh interval is 2s, using 2s instead of {args.refresh}s{Colors.RESET}"
        )

    try:
        asyncio.run(
            run_continuous(
                symbol,
                quiet=args.quiet,
                show_metrics=args.metrics,
                deep_dive=args.deep_dive,
                refresh_interval=refresh_interval,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
