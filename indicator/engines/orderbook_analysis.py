"""
Advanced Order Book Analysis Module
"Where price is FORCED to go"

This is your SHARPEST weapon.

What to IGNORE:
❌ Single walls (meaningless)
❌ Static snapshots (deceiving)
❌ Pretty heatmaps without behavior tracking

What ACTUALLY MATTERS:
✓ Absorption - Smart money defending levels
✓ Liquidity Imbalance - Directional pressure
✓ Spoof Detection - Behavioral patterns
✓ Liquidity Ladders - Where price is forced to go
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .signals import Signal

if TYPE_CHECKING:
    from .indicator_config import IndicatorConfig

from .indicator_config import DEFAULT_CONFIG


class AbsorptionSide(Enum):
    """Which side is absorbing."""

    BID_ABSORPTION = "bid_absorption"  # Bids absorbing sells (bullish)
    ASK_ABSORPTION = "ask_absorption"  # Asks absorbing buys (bearish)
    NONE = "none"


class ImbalanceDirection(Enum):
    """Direction of liquidity imbalance."""

    BID_HEAVY = "bid_heavy"  # More bids → bullish pressure
    ASK_HEAVY = "ask_heavy"  # More asks → bearish pressure
    BALANCED = "balanced"  # No clear imbalance


class SpoofType(Enum):
    """Type of spoof detected."""

    BID_SPOOF = "bid_spoof"  # Fake bid wall pulled before hit
    ASK_SPOOF = "ask_spoof"  # Fake ask wall pulled before hit
    NONE = "none"


class LiquidityZone(Enum):
    """Type of liquidity zone."""

    THICK = "thick"  # Heavy liquidity - price repels
    THIN = "thin"  # Light liquidity - price attracts
    NORMAL = "normal"


@dataclass
class OrderbookSnapshot:
    """A single orderbook snapshot with analysis."""

    timestamp: int
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]
    mid_price: float
    spread: float
    spread_percent: float

    # Depth at various levels
    bid_depth_10: float  # Top 10 levels
    ask_depth_10: float
    bid_depth_20: float  # Top 20 levels
    ask_depth_20: float

    # Value-weighted depth
    bid_value_10: float
    ask_value_10: float


@dataclass
class AbsorptionResult:
    """Result of absorption detection."""

    detected: bool
    side: AbsorptionSide
    strength: float  # 0-100, how strong the absorption
    volume_absorbed: float  # Estimated volume absorbed
    price_impact: float  # How much price moved despite volume
    efficiency: float  # volume / price_move (higher = more absorption)
    description: str
    action: str


@dataclass
class ImbalanceResult:
    """Result of liquidity imbalance analysis."""

    direction: ImbalanceDirection
    ratio: float  # bid/ask ratio
    is_actionable: bool  # Has volume confirmation?
    is_bait: bool  # Imbalance alone (no volume)?
    strength: float  # 0-100
    description: str
    action: str


@dataclass
class SpoofResult:
    """Result of spoof detection."""

    detected: bool
    spoof_type: SpoofType
    confidence: float
    wall_size: float
    wall_price: float
    times_pulled: int  # How many times this wall was pulled
    trapped_traders: bool  # Did OI increase after pull?
    description: str
    action: str


@dataclass
class LiquidityLadderResult:
    """Result of liquidity ladder analysis."""

    # Above current price
    thick_zones_above: List[Tuple[float, float]]  # (price, liquidity)
    thin_zones_above: List[Tuple[float, float]]

    # Below current price
    thick_zones_below: List[Tuple[float, float]]
    thin_zones_below: List[Tuple[float, float]]

    # Path of least resistance
    path_of_least_resistance: str  # 'up', 'down', 'unclear'
    nearest_thick_above: Optional[float]
    nearest_thick_below: Optional[float]
    nearest_thin_above: Optional[float]
    nearest_thin_below: Optional[float]

    description: str


@dataclass
class OrderbookAnalysisSummary:
    """Complete orderbook analysis summary."""

    snapshot: OrderbookSnapshot
    absorption: AbsorptionResult
    imbalance: ImbalanceResult
    spoof: SpoofResult
    liquidity_ladder: LiquidityLadderResult

    overall_signal: Signal
    confidence: float
    where_price_forced: str  # Direction price is forced to go
    summary: str


class AdvancedOrderbookAnalyzer:
    """
    Professional orderbook analysis focusing on BEHAVIOR, not static snapshots.

    Key principles:
    - Track changes over time, not single snapshots
    - Absorption = Smart money defending
    - Imbalance without volume = BAIT
    - Spoofs are behavioral patterns
    """

    def __init__(
        self,
        imbalance_threshold: Optional[float] = None,
        absorption_efficiency: Optional[float] = None,
        config: Optional["IndicatorConfig"] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        cfg = self.config.orderbook
        self.imbalance_threshold = (
            imbalance_threshold
            if imbalance_threshold is not None
            else cfg.imbalance_ratio_threshold
        )
        self.absorption_efficiency = (
            absorption_efficiency
            if absorption_efficiency is not None
            else cfg.absorption_efficiency_max
        )
        self.strong_imbalance_ratio = cfg.strong_imbalance_ratio

        # Track previous path for hysteresis (prevent oscillation)
        self._previous_path: Optional[str] = None
        self.thick_liquidity_multiplier = cfg.thick_liquidity_multiplier
        self.thin_liquidity_multiplier = cfg.thin_liquidity_multiplier

        # Historical tracking for spoof detection
        self.wall_history: Dict[str, List[Dict]] = {}  # price -> [appearances]

    def create_snapshot(
        self, bids: List[List[float]], asks: List[List[float]], timestamp: Optional[int] = None
    ) -> OrderbookSnapshot:
        """Create an analyzed orderbook snapshot."""
        if not bids or not asks:
            raise ValueError("Empty orderbook")

        ts = timestamp or int(time.time() * 1000)

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100

        # Calculate depths
        bid_depth_10 = sum(b[1] for b in bids[:10])
        ask_depth_10 = sum(a[1] for a in asks[:10])
        bid_depth_20 = sum(b[1] for b in bids[:20])
        ask_depth_20 = sum(a[1] for a in asks[:20])

        # Value-weighted depth
        bid_value_10 = sum(b[0] * b[1] for b in bids[:10])
        ask_value_10 = sum(a[0] * a[1] for a in asks[:10])

        return OrderbookSnapshot(
            timestamp=ts,
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            spread=spread,
            spread_percent=spread_pct,
            bid_depth_10=bid_depth_10,
            ask_depth_10=ask_depth_10,
            bid_depth_20=bid_depth_20,
            ask_depth_20=ask_depth_20,
            bid_value_10=bid_value_10,
            ask_value_10=ask_value_10,
        )

    def detect_absorption(
        self,
        current_snapshot: OrderbookSnapshot,
        recent_volume: float,
        price_change_percent: float,
        oi_change_percent: Optional[float] = None,
        avg_volume: Optional[float] = None,
    ) -> AbsorptionResult:
        """
        Detect absorption - Large limit orders absorbing aggressive market orders.

        Signs of absorption:
        - Market buys/sells hitting the book
        - Price NOT moving despite volume
        - Volume increasing
        - OI flat or dropping (positions closing, not opening)

        Absorption = Smart money defending a level

        Args:
            avg_volume: Average volume over recent period. If None, uses recent_volume (disables spike detection)
        """
        # Use provided average or fallback to recent_volume (no spike detection)
        if avg_volume is None or avg_volume == 0:
            avg_volume = recent_volume
            volume_ratio = 1.0
        else:
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Calculate efficiency: how much price moved per unit of volume
        # Lower efficiency = more absorption
        if volume_ratio == 0:
            efficiency = 1.0
        else:
            efficiency = abs(price_change_percent) / volume_ratio if volume_ratio > 0 else 1.0

        cfg = self.config.orderbook

        # Check for absorption conditions
        high_volume = volume_ratio > cfg.absorption_volume_ratio  # Volume spike
        low_price_impact = (
            abs(price_change_percent) < cfg.absorption_price_impact_pct
        )  # Price didn't move much

        # OI confirmation (if available)
        oi_confirms = False
        if oi_change_percent is not None:
            oi_confirms = (
                oi_change_percent <= cfg.absorption_oi_confirm_max
            )  # OI flat or down = closing positions

        # Determine absorption side based on imbalance and price direction
        imbalance = (
            current_snapshot.bid_depth_10 / current_snapshot.ask_depth_10
            if current_snapshot.ask_depth_10 > 0
            else 1
        )

        is_absorption = high_volume and low_price_impact and efficiency < self.absorption_efficiency

        if is_absorption:
            if imbalance > cfg.absorption_bid_imbalance_ratio and price_change_percent <= 0:
                # Strong bids but price flat/down = bids absorbing sells
                side = AbsorptionSide.BID_ABSORPTION
                strength = min(90, 50 + (imbalance - 1) * 20 + (1 - efficiency) * 30)
                desc = "BID ABSORPTION: Large bids absorbing sell pressure"
                action = "Smart money defending - Look for long entries on dips"
            elif imbalance < cfg.absorption_ask_imbalance_ratio and price_change_percent >= 0:
                # Strong asks but price flat/up = asks absorbing buys
                side = AbsorptionSide.ASK_ABSORPTION
                strength = min(90, 50 + (1 / imbalance - 1) * 20 + (1 - efficiency) * 30)
                desc = "ASK ABSORPTION: Large asks absorbing buy pressure"
                action = "Smart money defending - Look for short entries on rallies"
            else:
                side = AbsorptionSide.NONE
                strength = 40
                desc = "Volume spike with low impact but unclear side"
                action = "Watch for clearer signal"
        else:
            side = AbsorptionSide.NONE
            strength = 30
            desc = "No absorption detected"
            action = "Normal market conditions"

        if oi_confirms and side != AbsorptionSide.NONE:
            desc += " | OI confirms (positions closing)"
            strength = min(95, strength + 10)

        return AbsorptionResult(
            detected=is_absorption and side != AbsorptionSide.NONE,
            side=side,
            strength=strength,
            volume_absorbed=recent_volume,
            price_impact=abs(price_change_percent),
            efficiency=efficiency,
            description=desc,
            action=action,
        )

    def analyze_imbalance(
        self, snapshot: OrderbookSnapshot, has_volume_confirmation: bool = False
    ) -> ImbalanceResult:
        """
        Analyze liquidity imbalance.

        Rules:
        - 2×–3× imbalance → directional pressure
        - Imbalance + volume → ACTIONABLE
        - Imbalance alone → BAIT (likely spoof/manipulation)
        """
        bid_depth = snapshot.bid_depth_10
        ask_depth = snapshot.ask_depth_10

        if ask_depth == 0:
            ratio = 99.0
        else:
            ratio = bid_depth / ask_depth

        # Also check value-weighted
        if snapshot.ask_value_10 > 0:
            value_ratio = snapshot.bid_value_10 / snapshot.ask_value_10
        else:
            value_ratio = ratio

        # Use average of depth and value ratios
        combined_ratio = (ratio + value_ratio) / 2

        # Determine direction
        if combined_ratio >= self.strong_imbalance_ratio:
            direction = ImbalanceDirection.BID_HEAVY
            strength = min(85, 50 + (combined_ratio - 1) * 10)
            desc = f"STRONG bid imbalance ({combined_ratio:.1f}x) - Heavy buying pressure"
        elif combined_ratio >= self.imbalance_threshold:
            direction = ImbalanceDirection.BID_HEAVY
            strength = 60 + (combined_ratio - 2) * 10
            desc = f"Moderate bid imbalance ({combined_ratio:.1f}x) - Buyers present"
        elif combined_ratio <= 1 / self.strong_imbalance_ratio:
            direction = ImbalanceDirection.ASK_HEAVY
            strength = min(85, 50 + (1 / combined_ratio - 1) * 10)
            desc = f"STRONG ask imbalance ({1/combined_ratio:.1f}x) - Heavy selling pressure"
        elif combined_ratio <= 1 / self.imbalance_threshold:
            direction = ImbalanceDirection.ASK_HEAVY
            strength = 60 + (1 / combined_ratio - 2) * 10
            desc = f"Moderate ask imbalance ({1/combined_ratio:.1f}x) - Sellers present"
        else:
            direction = ImbalanceDirection.BALANCED
            strength = 40
            desc = f"Balanced orderbook ({combined_ratio:.1f}x ratio)"

        # Critical: Is it actionable or bait?
        is_actionable = has_volume_confirmation and direction != ImbalanceDirection.BALANCED
        is_bait = direction != ImbalanceDirection.BALANCED and not has_volume_confirmation

        if is_actionable:
            if direction == ImbalanceDirection.BID_HEAVY:
                action = "ACTIONABLE: Bid imbalance with volume - Bullish bias"
            else:
                action = "ACTIONABLE: Ask imbalance with volume - Bearish bias"
        elif is_bait:
            action = "⚠ BAIT WARNING: Imbalance without volume - Likely manipulation"
            strength = max(30, strength - 20)  # Reduce confidence
        else:
            action = "No actionable imbalance"

        return ImbalanceResult(
            direction=direction,
            ratio=combined_ratio,
            is_actionable=is_actionable,
            is_bait=is_bait,
            strength=strength,
            description=desc,
            action=action,
        )

    def detect_spoof(
        self,
        current_snapshot: OrderbookSnapshot,
        previous_snapshots: Optional[List[OrderbookSnapshot]] = None,
        oi_increased: bool = False,
        wall_size_threshold: Optional[float] = None,
    ) -> SpoofResult:
        """
        Detect spoofing - Behavioral patterns, not visual.

        Spoof confirmation requires:
        1. Wall appears (large order)
        2. Price approaches wall
        3. Wall pulls (disappears before hit)
        4. Price snaps in opposite direction

        Extra confirmation:
        - Repeated behavior (same pattern multiple times)
        - No volume absorption when wall present
        - OI increases (traders got trapped)
        """
        cfg = self.config.orderbook

        if not previous_snapshots or len(previous_snapshots) < 3:
            return SpoofResult(
                detected=False,
                spoof_type=SpoofType.NONE,
                confidence=0,
                wall_size=0,
                wall_price=0,
                times_pulled=0,
                trapped_traders=False,
                description="Insufficient history for spoof detection",
                action="Need more snapshots to track wall behavior",
            )

        # Calculate average order size
        all_sizes = [b[1] for b in current_snapshot.bids[:20]] + [
            a[1] for a in current_snapshot.asks[:20]
        ]
        avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 1
        wall_multiplier = (
            wall_size_threshold
            if wall_size_threshold is not None
            else cfg.spoof_wall_size_multiplier
        )
        wall_threshold = avg_size * wall_multiplier

        # Track walls over time
        spoof_detected = False
        spoof_type = SpoofType.NONE
        confidence = 0
        wall_size = 0
        wall_price = 0
        times_pulled = 0

        # Check for walls that appeared and disappeared
        lookback = min(cfg.spoof_lookback_snapshots, len(previous_snapshots))
        for i, prev_snap in enumerate(previous_snapshots[-lookback:]):
            # Check bid walls
            for prev_bid in prev_snap.bids[:10]:
                if prev_bid[1] >= wall_threshold:
                    # Large wall existed - check if it's still there
                    still_exists = any(
                        abs(b[0] - prev_bid[0]) < cfg.spoof_price_tolerance_ratio * prev_bid[0]
                        and b[1] >= prev_bid[1] * cfg.spoof_wall_persist_ratio
                        for b in current_snapshot.bids[:10]
                    )

                    if not still_exists:
                        # Wall was pulled - check if price approached
                        price_approached = prev_snap.mid_price < prev_bid[0] * (
                            1 + cfg.spoof_price_approach_pct / 100
                        )

                        # CRITICAL: Check if price snapped in OPPOSITE direction after pull
                        # Bid wall pulled → price should move DOWN (opposite of fake support)
                        # Require minimum move in bps to filter out noise from 250ms snapshots
                        price_move_bps = (
                            (prev_snap.mid_price - current_snapshot.mid_price)
                            / prev_snap.mid_price
                            * 10000
                        )
                        price_snapped_opposite = price_move_bps > cfg.spoof_price_snap_min_bps

                        if price_approached and price_snapped_opposite:
                            spoof_detected = True
                            spoof_type = SpoofType.BID_SPOOF
                            wall_size = prev_bid[1]
                            wall_price = prev_bid[0]
                            times_pulled += 1

            # Check ask walls
            for prev_ask in prev_snap.asks[:10]:
                if prev_ask[1] >= wall_threshold:
                    still_exists = any(
                        abs(a[0] - prev_ask[0]) < cfg.spoof_price_tolerance_ratio * prev_ask[0]
                        and a[1] >= prev_ask[1] * cfg.spoof_wall_persist_ratio
                        for a in current_snapshot.asks[:10]
                    )

                    if not still_exists:
                        # Wall was pulled - check if price approached
                        price_approached = prev_snap.mid_price > prev_ask[0] * (
                            1 - cfg.spoof_price_approach_pct / 100
                        )

                        # CRITICAL: Check if price snapped in OPPOSITE direction after pull
                        # Ask wall pulled → price should move UP (opposite of fake resistance)
                        # Require minimum move in bps to filter out noise from 250ms snapshots
                        price_move_bps = (
                            (current_snapshot.mid_price - prev_snap.mid_price)
                            / prev_snap.mid_price
                            * 10000
                        )
                        price_snapped_opposite = price_move_bps > cfg.spoof_price_snap_min_bps

                        if price_approached and price_snapped_opposite:
                            spoof_detected = True
                            spoof_type = SpoofType.ASK_SPOOF
                            wall_size = prev_ask[1]
                            wall_price = prev_ask[0]
                            times_pulled += 1

        # Calculate confidence
        if spoof_detected:
            confidence = 50
            if times_pulled >= 2:
                confidence += 15  # Repeated behavior
            if oi_increased:
                confidence += 20  # Trapped traders
            confidence = min(90, confidence)

            # Enforce minimum confidence threshold
            if confidence < cfg.spoof_min_confidence:
                spoof_detected = False
                confidence = 0

        if spoof_detected:
            if spoof_type == SpoofType.BID_SPOOF:
                desc = f"BID SPOOF detected: {wall_size:,.0f} wall at ${wall_price:,.2f} pulled"
                action = "Fake support - Price likely to drop. Look for shorts."
            else:
                desc = f"ASK SPOOF detected: {wall_size:,.0f} wall at ${wall_price:,.2f} pulled"
                action = "Fake resistance - Price likely to rise. Look for longs."

            if oi_increased:
                desc += " | Traders trapped (OI up)"
        else:
            desc = "No spoof patterns detected"
            action = "Normal wall behavior"

        return SpoofResult(
            detected=spoof_detected,
            spoof_type=spoof_type,
            confidence=confidence,
            wall_size=wall_size,
            wall_price=wall_price,
            times_pulled=times_pulled,
            trapped_traders=oi_increased and spoof_detected,
            description=desc,
            action=action,
        )

    def analyze_liquidity_ladder(
        self,
        snapshot: OrderbookSnapshot,
        levels_to_analyze: int = 50,  # Increased from 20 to match new orderbook depth
    ) -> LiquidityLadderResult:
        """
        Analyze liquidity ladders - Where is price FORCED to go?

        Price moves:
        - TOWARD thin zones (less resistance)
        - AWAY FROM thick zones (more resistance)

        This shows the path of least resistance.

        Uses notional value weighting and distance-in-bps to resist outliers.
        """
        cfg = self.config.orderbook
        mid_price = snapshot.mid_price

        # Calculate notional-weighted liquidity with distance weighting
        # Outlier-resistant: use median instead of mean
        bid_notionals = []
        ask_notionals = []

        for bid in snapshot.bids[:levels_to_analyze]:
            price, qty = bid[0], bid[1]
            notional = price * qty
            distance_bps = abs(price - mid_price) / mid_price * 10000  # distance in basis points

            # Weight by inverse distance (closer levels matter more)
            # Use 1/(1 + distance_bps/100) to avoid division by zero and smooth falloff
            weight = 1.0 / (1.0 + distance_bps / 100)
            weighted_notional = notional * weight
            bid_notionals.append(weighted_notional)

        for ask in snapshot.asks[:levels_to_analyze]:
            price, qty = ask[0], ask[1]
            notional = price * qty
            distance_bps = abs(price - mid_price) / mid_price * 10000

            weight = 1.0 / (1.0 + distance_bps / 100)
            weighted_notional = notional * weight
            ask_notionals.append(weighted_notional)

        # Use MEDIAN for outlier resistance
        all_notionals = bid_notionals + ask_notionals
        if not all_notionals:
            # Return neutral result if no data
            return LiquidityLadderResult(
                thick_zones_above=[],
                thin_zones_above=[],
                thick_zones_below=[],
                thin_zones_below=[],
                path_of_least_resistance="unclear",
                nearest_thick_above=None,
                nearest_thick_below=None,
                nearest_thin_above=None,
                nearest_thin_below=None,
                description="Insufficient orderbook data",
            )

        median_liquidity = sorted(all_notionals)[len(all_notionals) // 2]
        thick_threshold = median_liquidity * self.thick_liquidity_multiplier
        thin_threshold = median_liquidity * self.thin_liquidity_multiplier

        # Analyze zones above (asks) with notional values
        thick_above = []
        thin_above = []
        for i, ask in enumerate(snapshot.asks[:levels_to_analyze]):
            if i < len(ask_notionals):
                weighted_notional = ask_notionals[i]
                if weighted_notional >= thick_threshold:
                    thick_above.append((ask[0], ask[1]))
                elif weighted_notional <= thin_threshold:
                    thin_above.append((ask[0], ask[1]))

        # Analyze zones below (bids) with notional values
        thick_below = []
        thin_below = []
        for i, bid in enumerate(snapshot.bids[:levels_to_analyze]):
            if i < len(bid_notionals):
                weighted_notional = bid_notionals[i]
                if weighted_notional >= thick_threshold:
                    thick_below.append((bid[0], bid[1]))
                elif weighted_notional <= thin_threshold:
                    thin_below.append((bid[0], bid[1]))

        # Find nearest zones
        nearest_thick_above = thick_above[0][0] if thick_above else None
        nearest_thick_below = thick_below[0][0] if thick_below else None
        nearest_thin_above = thin_above[0][0] if thin_above else None
        nearest_thin_below = thin_below[0][0] if thin_below else None

        # Determine path of least resistance using CUMULATIVE notional, not distance
        # Sum the weighted notional resistance in each direction
        # Use configurable depth (default 15) to capture more of the book
        levels_to_sum = min(cfg.path_resistance_levels, len(ask_notionals), len(bid_notionals))
        total_resistance_up = sum(ask_notionals[:levels_to_sum]) if ask_notionals else 1.0
        total_resistance_down = sum(bid_notionals[:levels_to_sum]) if bid_notionals else 1.0

        # Lower resistance = easier path
        resistance_ratio = (
            total_resistance_up / total_resistance_down if total_resistance_down > 0 else 1.0
        )

        # Apply hysteresis: use stricter threshold if flipping from previous direction
        # This prevents oscillation in choppy markets
        threshold_up = cfg.path_resistance_threshold  # 0.7
        threshold_down = 1.0 / cfg.path_resistance_threshold  # 1.43

        if self._previous_path == "down":
            # Make it harder to flip from down to up (require 15% more evidence)
            threshold_up = threshold_up * cfg.path_hysteresis_factor  # 0.7 * 0.85 = 0.595
        elif self._previous_path == "up":
            # Make it harder to flip from up to down (require 15% more evidence)
            threshold_down = threshold_down / cfg.path_hysteresis_factor  # 1.43 / 0.85 = 1.68

        # Use ratio-based scoring with hysteresis
        if resistance_ratio < threshold_up:
            path = "up"
            desc = f"PATH OF LEAST RESISTANCE: UP - {(1-resistance_ratio)*100:.0f}% less resistance above"
        elif resistance_ratio > threshold_down:
            path = "down"
            desc = f"PATH OF LEAST RESISTANCE: DOWN - {(resistance_ratio-1)*100:.0f}% less resistance below"
        else:
            path = "unclear"
            desc = "Path unclear - Balanced resistance"

        # Update previous path for next calculation
        self._previous_path = path

        # Add zone details
        if nearest_thick_above:
            desc += f" | Resistance at ${nearest_thick_above:,.2f}"
        if nearest_thick_below:
            desc += f" | Support at ${nearest_thick_below:,.2f}"

        return LiquidityLadderResult(
            thick_zones_above=thick_above,
            thin_zones_above=thin_above,
            thick_zones_below=thick_below,
            thin_zones_below=thin_below,
            path_of_least_resistance=path,
            nearest_thick_above=nearest_thick_above,
            nearest_thick_below=nearest_thick_below,
            nearest_thin_above=nearest_thin_above,
            nearest_thin_below=nearest_thin_below,
            description=desc,
        )

    def full_analysis(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        recent_volume: float = 0,
        price_change_percent: float = 0,
        oi_change_percent: Optional[float] = None,
        previous_snapshots: Optional[List[OrderbookSnapshot]] = None,
        timestamp: Optional[int] = None,
        avg_volume: Optional[float] = None,
    ) -> OrderbookAnalysisSummary:
        """
        Complete orderbook analysis answering: "Where is price FORCED to go?"

        Combines:
        - Absorption detection
        - Liquidity imbalance
        - Spoof detection
        - Liquidity ladders
        """
        # Create snapshot
        snapshot = self.create_snapshot(bids, asks, timestamp)

        cfg = self.config.orderbook

        # Check if volume confirms imbalance
        # Use actual volume ratio (not price change) to determine if there's real activity
        if avg_volume is not None and avg_volume > 0:
            volume_ratio = recent_volume / avg_volume
            has_volume = volume_ratio > cfg.volume_confirmation_ratio_threshold
        else:
            # Fallback: if no avg_volume baseline, be conservative
            # Don't confirm imbalances without a proper baseline to compare against
            # This prevents false positives when we can't measure relative volume
            has_volume = False

        # Run analyses
        absorption = self.detect_absorption(
            snapshot, recent_volume, price_change_percent, oi_change_percent, avg_volume
        )

        imbalance = self.analyze_imbalance(snapshot, has_volume)

        oi_increased = (
            oi_change_percent is not None and oi_change_percent > cfg.spoof_oi_increase_threshold
        )
        spoof = self.detect_spoof(snapshot, previous_snapshots, oi_increased)

        ladder = self.analyze_liquidity_ladder(snapshot)

        # Determine overall signal
        signals: List[Tuple[Signal, float]] = []

        if absorption.detected:
            if absorption.side == AbsorptionSide.BID_ABSORPTION:
                signals.append((Signal.BULLISH, absorption.strength))
            else:
                signals.append((Signal.BEARISH, absorption.strength))

        if imbalance.is_actionable:
            if imbalance.direction == ImbalanceDirection.BID_HEAVY:
                signals.append((Signal.BULLISH, imbalance.strength))
            elif imbalance.direction == ImbalanceDirection.ASK_HEAVY:
                signals.append((Signal.BEARISH, imbalance.strength))

        if spoof.detected:
            # Spoof is contrarian - fake wall means opposite direction
            if spoof.spoof_type == SpoofType.BID_SPOOF:
                signals.append((Signal.BEARISH, spoof.confidence))
            else:
                signals.append((Signal.BULLISH, spoof.confidence))

        if ladder.path_of_least_resistance == "up":
            signals.append((Signal.BULLISH, 55))
        elif ladder.path_of_least_resistance == "down":
            signals.append((Signal.BEARISH, 55))

        # Aggregate signals
        if not signals:
            overall_signal = Signal.NEUTRAL
            confidence = 40
        else:
            bullish_weight = sum(s for sig, s in signals if sig == Signal.BULLISH)
            bearish_weight = sum(s for sig, s in signals if sig == Signal.BEARISH)

            if bullish_weight > bearish_weight * cfg.directional_signal_ratio:
                overall_signal = Signal.BULLISH
                confidence = min(
                    85, bullish_weight / len([s for s in signals if s[0] == Signal.BULLISH])
                )
            elif bearish_weight > bullish_weight * cfg.directional_signal_ratio:
                overall_signal = Signal.BEARISH
                confidence = min(
                    85, bearish_weight / len([s for s in signals if s[0] == Signal.BEARISH])
                )
            else:
                overall_signal = Signal.NEUTRAL
                confidence = 50

        # Check for trap conditions
        if imbalance.is_bait and spoof.detected:
            overall_signal = Signal.TRAP
            confidence = max(spoof.confidence, 60)

        # Determine where price is forced
        if ladder.path_of_least_resistance != "unclear":
            where_forced = ladder.path_of_least_resistance.upper()
        elif overall_signal == Signal.BULLISH:
            where_forced = "UP (based on flow)"
        elif overall_signal == Signal.BEARISH:
            where_forced = "DOWN (based on flow)"
        else:
            where_forced = "UNCLEAR"

        # Generate summary
        summary_parts = []
        if absorption.detected:
            summary_parts.append(f"Absorption: {absorption.side.value}")
        if imbalance.is_actionable:
            summary_parts.append(f"Imbalance: {imbalance.direction.value}")
        if spoof.detected:
            summary_parts.append(f"Spoof: {spoof.spoof_type.value}")
        summary_parts.append(f"Path: {ladder.path_of_least_resistance}")

        summary = " | ".join(summary_parts) if summary_parts else "Normal orderbook conditions"

        return OrderbookAnalysisSummary(
            snapshot=snapshot,
            absorption=absorption,
            imbalance=imbalance,
            spoof=spoof,
            liquidity_ladder=ladder,
            overall_signal=overall_signal,
            confidence=confidence,
            where_price_forced=where_forced,
            summary=summary,
        )


# Convenience functions
def get_path_of_least_resistance(
    bids: List[List[float]], asks: List[List[float]], config: Optional["IndicatorConfig"] = None
) -> str:
    """Quick check: Where is price forced to go?"""
    analyzer = AdvancedOrderbookAnalyzer(config=config)
    snapshot = analyzer.create_snapshot(bids, asks)
    ladder = analyzer.analyze_liquidity_ladder(snapshot)
    return ladder.path_of_least_resistance


def is_imbalance_actionable(
    bids: List[List[float]],
    asks: List[List[float]],
    has_volume: bool,
    config: Optional["IndicatorConfig"] = None,
) -> Tuple[bool, str]:
    """Quick check: Is this imbalance actionable or bait?"""
    analyzer = AdvancedOrderbookAnalyzer(config=config)
    snapshot = analyzer.create_snapshot(bids, asks)
    result = analyzer.analyze_imbalance(snapshot, has_volume)
    return result.is_actionable, result.action
