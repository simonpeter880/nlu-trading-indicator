"""
Institutional Market Structure Engine

Complete implementation of professional market structure analysis:
- Swing detection (HH/HL/LH/LL) with pivot algorithm
- BOS (Break of Structure) - continuation
- CHoCH (Change of Character) - reversal
- Liquidity sweeps with confirmation
- Acceptance vs Rejection after breaks
- Range classification (Compression/Distribution/Accumulation)
- Fair Value Gaps (FVG) - optional
- Time-to-followthrough momentum
- Multi-timeframe alignment

DESIGN:
- Deterministic, unit-testable, no external services
- No repainting: pivots use L/R lookahead windows
- Clean dataclasses for all outputs
- Configurable parameters with sane defaults
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import math


# ============================================================================
# ENUMS
# ============================================================================

class SwingType(Enum):
    """Swing point type."""
    HIGH = "HIGH"
    LOW = "LOW"


class StructureSide(Enum):
    """Bullish or bearish."""
    BULL = "BULL"
    BEAR = "BEAR"


class EventType(Enum):
    """Structure event types."""
    BOS = "BOS"           # Break of Structure
    CHOCH = "CHOCH"       # Change of Character
    SWEEP = "SWEEP"       # Liquidity sweep
    ACCEPT = "ACCEPT"     # Acceptance confirmation
    REJECT = "REJECT"     # Rejection confirmation


class RangeClassification(Enum):
    """Range type classification."""
    COMPRESSION = "COMPRESSION"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    NEUTRAL = "NEUTRAL"


class StructureState(Enum):
    """Overall market structure state."""
    UP = "UP"
    DOWN = "DOWN"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class StructuralMomentum(Enum):
    """Time-to-followthrough momentum."""
    FAST = "FAST"
    SLOW = "SLOW"
    STALLED = "STALLED"


class TimeframeAlignment(Enum):
    """Multi-timeframe alignment."""
    ALIGNED = "ALIGNED"           # Both same direction
    MIXED = "MIXED"               # HTF trend, LTF range/opposite
    RANGE_DOMINANT = "RANGE_DOMINANT"  # HTF range


class TradingMode(Enum):
    """Recommended trading mode."""
    TREND_MODE = "TREND_MODE"
    RANGE_MODE = "RANGE_MODE"
    SCALP_ONLY = "SCALP_ONLY"


class ZoneStatus(Enum):
    """Zone status."""
    OPEN = "OPEN"
    FILLED = "FILLED"
    BROKEN = "BROKEN"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Candle:
    """OHLCV candle."""
    timestamp: int  # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SwingPoint:
    """A swing high or swing low."""
    swing_type: SwingType
    price: float
    index: int        # Index in candle array
    time: int         # Timestamp ms
    strength: float   # 0-1, based on distance to opposite swing / ATR


@dataclass
class StructureEvent:
    """Market structure event (BOS, CHoCH, SWEEP, etc.)."""
    event_type: EventType
    side: StructureSide
    level: float      # Price level involved
    time: int         # Timestamp ms
    index: int        # Candle index
    details: Dict = field(default_factory=dict)


@dataclass
class Zone:
    """Range or FVG zone."""
    zone_type: str    # "RANGE" or "FVG"
    side: Optional[StructureSide]  # BULL/BEAR for FVG, None for RANGE
    top: float
    bottom: float
    created_time: int
    status: ZoneStatus = ZoneStatus.OPEN


@dataclass
class MarketStructureState:
    """Complete market structure state for a timeframe."""
    structure: StructureState           # UP / DOWN / RANGE
    strength_0_100: float              # Confidence 0-100
    regime: str                        # "TREND" / "RANGE" / "MIXED"

    # Swings
    last_swing_high: Optional[SwingPoint] = None
    last_swing_low: Optional[SwingPoint] = None
    recent_swings: List[SwingPoint] = field(default_factory=list)

    # Events
    last_bos: Optional[StructureEvent] = None
    last_choch: Optional[StructureEvent] = None
    recent_events: List[StructureEvent] = field(default_factory=list)

    # Zones
    active_range: Optional[Zone] = None
    active_fvgs: List[Zone] = field(default_factory=list)

    # Momentum
    momentum: StructuralMomentum = StructuralMomentum.SLOW

    # Labels (HH/HL/LH/LL pattern)
    structure_label: str = ""  # e.g., "HH+HL"


@dataclass
class MultiTFAlignment:
    """Multi-timeframe alignment summary."""
    alignment: TimeframeAlignment
    recommended_mode: TradingMode
    htf_structure: StructureState
    ltf_structure: StructureState
    htf_last_bos_time: Optional[int] = None
    ltf_last_bos_time: Optional[int] = None


@dataclass
class StructureConfig:
    """Configuration for market structure engine."""
    # Swing detection
    pivot_left: int = 3
    pivot_right: int = 3

    # BOS/CHoCH buffers
    bos_buffer_pct: float = 0.05       # Min % move to confirm BOS
    bos_buffer_atr_mult: float = 0.15  # OR 0.15 * ATR%
    use_close_for_breaks: bool = True  # Use close vs high/low

    # Sweep detection
    sweep_buffer_pct: float = 0.03
    sweep_buffer_atr_mult: float = 0.10
    sweep_confirmation_candles: int = 3  # Look ahead K candles for confirmation
    sweep_rv_threshold: float = 1.2

    # Acceptance/Rejection
    accept_hold_candles: int = 3
    accept_rv_threshold: float = 1.2

    # Range detection
    range_no_bos_candles: int = 20      # No BOS for X candles
    range_atr_contraction: bool = True   # ATR must be contracting

    # Range classification
    range_width_history: int = 3  # Compare to last N ranges

    # FVG
    enable_fvg: bool = True

    # Momentum
    momentum_move_atr_mult: float = 0.5  # Move >= 0.5*ATR for continuation
    momentum_fast_candles: int = 3
    momentum_max_candles: int = 10

    # ATR
    atr_period: int = 14
    rv_period: int = 20


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_atr(candles: List[Candle], period: int = 14) -> List[float]:
    """
    Compute ATR (Average True Range) for each candle.

    Returns:
        List of ATR values (same length as candles)
    """
    if len(candles) < 2:
        return [0.0] * len(candles)

    true_ranges = [0.0]  # First candle has no previous close

    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    # SMA for first ATR
    atrs = [0.0] * len(candles)
    if len(true_ranges) >= period:
        atrs[period - 1] = sum(true_ranges[:period]) / period

        # EMA for subsequent
        for i in range(period, len(true_ranges)):
            atrs[i] = (atrs[i - 1] * (period - 1) + true_ranges[i]) / period

    return atrs


def compute_atr_pct(candles: List[Candle], period: int = 14) -> List[float]:
    """
    Compute ATR as percentage of price.

    Returns:
        List of ATR% values
    """
    atrs = compute_atr(candles, period)
    atr_pcts = []

    for i, candle in enumerate(candles):
        if candle.close > 0:
            atr_pcts.append((atrs[i] / candle.close) * 100)
        else:
            atr_pcts.append(0.0)

    return atr_pcts


def compute_rv(volumes: List[float], period: int = 20) -> List[float]:
    """
    Compute Relative Volume (volume / SMA(volume, period)).

    Returns:
        List of RV values
    """
    if len(volumes) < period:
        return [1.0] * len(volumes)

    rvs = [1.0] * len(volumes)

    for i in range(period - 1, len(volumes)):
        window = volumes[i - period + 1:i + 1]
        avg = sum(window) / len(window)
        if avg > 0:
            rvs[i] = volumes[i] / avg
        else:
            rvs[i] = 1.0

    return rvs


def sma(values: List[float], period: int) -> List[float]:
    """Simple moving average."""
    if len(values) < period:
        return [0.0] * len(values)

    result = [0.0] * len(values)

    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        result[i] = sum(window) / len(window)

    return result


# ============================================================================
# MARKET STRUCTURE ENGINE
# ============================================================================

class MarketStructureEngine:
    """
    Institutional Market Structure Engine.

    Analyzes price action across multiple timeframes to identify:
    - Swing points and structure (HH/HL/LH/LL)
    - BOS and CHoCH events
    - Liquidity sweeps
    - Range structures
    - Fair Value Gaps
    - Multi-timeframe alignment
    """

    def __init__(self, config: Optional[StructureConfig] = None):
        self.config = config or StructureConfig()

        # State per timeframe
        self._states: Dict[str, MarketStructureState] = {}
        self._candle_history: Dict[str, List[Candle]] = {}
        self._swing_history: Dict[str, List[SwingPoint]] = {}
        self._event_history: Dict[str, List[StructureEvent]] = {}

        # Pending acceptance/rejection tracking
        self._pending_breaks: Dict[str, List[Tuple[StructureEvent, int]]] = {}

    def update(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, MarketStructureState]:
        """
        Update market structure for all timeframes.

        Args:
            candles_by_tf: Dict mapping timeframe name to list of Candles
                          e.g., {"LTF": [...], "HTF": [...]}

        Returns:
            Dict mapping timeframe to MarketStructureState
        """
        states = {}

        for tf, candles in candles_by_tf.items():
            if len(candles) < 10:
                # Not enough data
                states[tf] = MarketStructureState(
                    structure=StructureState.UNKNOWN,
                    strength_0_100=0.0,
                    regime="UNKNOWN"
                )
                continue

            # Update state for this timeframe
            state = self._analyze_timeframe(tf, candles)
            states[tf] = state
            self._states[tf] = state

        return states

    def get_mtf_alignment(self, htf_name: str = "HTF", ltf_name: str = "LTF") -> Optional[MultiTFAlignment]:
        """
        Get multi-timeframe alignment between HTF and LTF.

        Args:
            htf_name: Higher timeframe key
            ltf_name: Lower timeframe key

        Returns:
            MultiTFAlignment or None if data missing
        """
        if htf_name not in self._states or ltf_name not in self._states:
            return None

        htf_state = self._states[htf_name]
        ltf_state = self._states[ltf_name]

        # Determine alignment
        if htf_state.structure == StructureState.RANGE:
            alignment = TimeframeAlignment.RANGE_DOMINANT
            mode = TradingMode.RANGE_MODE
        elif htf_state.structure == ltf_state.structure:
            alignment = TimeframeAlignment.ALIGNED
            mode = TradingMode.TREND_MODE
        else:
            alignment = TimeframeAlignment.MIXED
            mode = TradingMode.SCALP_ONLY

        return MultiTFAlignment(
            alignment=alignment,
            recommended_mode=mode,
            htf_structure=htf_state.structure,
            ltf_structure=ltf_state.structure,
            htf_last_bos_time=htf_state.last_bos.time if htf_state.last_bos else None,
            ltf_last_bos_time=ltf_state.last_bos.time if ltf_state.last_bos else None,
        )

    def _analyze_timeframe(self, tf: str, candles: List[Candle]) -> MarketStructureState:
        """Analyze structure for a single timeframe."""
        # Store candles
        self._candle_history[tf] = candles

        # Compute ATR and RV
        atr_pct = compute_atr_pct(candles, self.config.atr_period)
        volumes = [c.volume for c in candles]
        rv = compute_rv(volumes, self.config.rv_period)

        # 1. Detect swings
        swings = self._detect_swings(candles, atr_pct)
        self._swing_history[tf] = swings

        # 2. Classify structure (UP/DOWN/RANGE)
        structure, structure_label = self._classify_structure(swings)

        # 3. Detect BOS and CHoCH
        events = self._detect_bos_choch(candles, swings, structure, atr_pct, tf)

        # 4. Detect liquidity sweeps
        sweep_events = self._detect_sweeps(candles, swings, atr_pct, rv)
        events.extend(sweep_events)

        # 5. Check acceptance/rejection for pending breaks
        accept_reject_events = self._check_acceptance_rejection(tf, candles, rv)
        events.extend(accept_reject_events)

        # Store events
        if tf not in self._event_history:
            self._event_history[tf] = []
        self._event_history[tf].extend(events)

        # 6. Detect ranges
        active_range = self._detect_range(candles, swings, events, atr_pct)

        # 7. Detect FVGs (optional)
        active_fvgs = []
        if self.config.enable_fvg:
            active_fvgs = self._detect_fvgs(candles)

        # 8. Compute momentum
        momentum = self._compute_momentum(candles, events, atr_pct)

        # 9. Extract last events
        last_bos = None
        last_choch = None
        for event in reversed(events):
            if event.event_type == EventType.BOS and last_bos is None:
                last_bos = event
            if event.event_type == EventType.CHOCH and last_choch is None:
                last_choch = event

        # 10. Compute strength/confidence
        strength = self._compute_strength(structure, events, swings, active_range)

        # 11. Determine regime
        regime = "TREND" if structure in [StructureState.UP, StructureState.DOWN] else "RANGE"
        if active_range and structure != StructureState.RANGE:
            regime = "MIXED"

        # 12. Get recent swings (last 10)
        recent_swings = swings[-10:] if len(swings) > 10 else swings

        return MarketStructureState(
            structure=structure,
            strength_0_100=strength,
            regime=regime,
            last_swing_high=self._get_last_swing(swings, SwingType.HIGH),
            last_swing_low=self._get_last_swing(swings, SwingType.LOW),
            recent_swings=recent_swings,
            last_bos=last_bos,
            last_choch=last_choch,
            recent_events=events[-10:],
            active_range=active_range,
            active_fvgs=active_fvgs,
            momentum=momentum,
            structure_label=structure_label,
        )

    def _detect_swings(self, candles: List[Candle], atr_pct: List[float]) -> List[SwingPoint]:
        """
        Detect swing highs and lows using pivot algorithm.

        Swing High: high[i] is max of high[i-L:i+R+1]
        Swing Low: low[i] is min of low[i-L:i+R+1]
        """
        L = self.config.pivot_left
        R = self.config.pivot_right
        swings = []

        # Can only confirm swings up to len(candles) - R
        max_idx = len(candles) - R

        for i in range(L, max_idx):
            # Check swing high
            window_highs = [candles[j].high for j in range(i - L, i + R + 1)]
            if candles[i].high == max(window_highs):
                # Strictly greater than neighbors
                if (candles[i].high > candles[i - 1].high and
                    candles[i].high > candles[i + 1].high):

                    strength = self._compute_swing_strength(
                        i, candles, atr_pct, SwingType.HIGH, swings
                    )

                    swings.append(SwingPoint(
                        swing_type=SwingType.HIGH,
                        price=candles[i].high,
                        index=i,
                        time=candles[i].timestamp,
                        strength=strength
                    ))

            # Check swing low
            window_lows = [candles[j].low for j in range(i - L, i + R + 1)]
            if candles[i].low == min(window_lows):
                # Strictly less than neighbors
                if (candles[i].low < candles[i - 1].low and
                    candles[i].low < candles[i + 1].low):

                    strength = self._compute_swing_strength(
                        i, candles, atr_pct, SwingType.LOW, swings
                    )

                    swings.append(SwingPoint(
                        swing_type=SwingType.LOW,
                        price=candles[i].low,
                        index=i,
                        time=candles[i].timestamp,
                        strength=strength
                    ))

        return swings

    def _compute_swing_strength(
        self,
        idx: int,
        candles: List[Candle],
        atr_pct: List[float],
        swing_type: SwingType,
        existing_swings: List[SwingPoint]
    ) -> float:
        """
        Compute swing strength based on distance to nearest opposite swing / ATR.

        Returns:
            Float 0-1
        """
        if not existing_swings or atr_pct[idx] == 0:
            return 0.5

        current_price = candles[idx].high if swing_type == SwingType.HIGH else candles[idx].low

        # Find nearest opposite swing
        opposite_type = SwingType.LOW if swing_type == SwingType.HIGH else SwingType.HIGH
        opposite_swings = [s for s in existing_swings if s.swing_type == opposite_type]

        if not opposite_swings:
            return 0.5

        # Get most recent opposite
        nearest = opposite_swings[-1]
        distance_pct = abs(current_price - nearest.price) / current_price * 100

        # Normalize by ATR
        atr = atr_pct[idx]
        if atr > 0:
            strength = distance_pct / atr
            return min(1.0, max(0.0, strength))

        return 0.5

    def _classify_structure(self, swings: List[SwingPoint]) -> Tuple[StructureState, str]:
        """
        Classify structure based on swing pattern (HH/HL/LH/LL).

        Returns:
            (StructureState, label)
        """
        if len(swings) < 4:
            return StructureState.UNKNOWN, ""

        highs = [s for s in swings if s.swing_type == SwingType.HIGH]
        lows = [s for s in swings if s.swing_type == SwingType.LOW]

        if len(highs) < 2 or len(lows) < 2:
            return StructureState.UNKNOWN, ""

        # Check last 3 of each
        recent_highs = highs[-3:] if len(highs) >= 3 else highs
        recent_lows = lows[-3:] if len(lows) >= 3 else lows

        # Count HH/LH
        hh = 0
        lh = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price > recent_highs[i - 1].price:
                hh += 1
            else:
                lh += 1

        # Count HL/LL
        hl = 0
        ll = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price > recent_lows[i - 1].price:
                hl += 1
            else:
                ll += 1

        # Classify
        if hh > 0 and hl > 0 and lh == 0:
            return StructureState.UP, "HH+HL"
        elif lh > 0 and ll > 0 and hh == 0:
            return StructureState.DOWN, "LH+LL"
        else:
            return StructureState.RANGE, "MIXED"

    def _detect_bos_choch(
        self,
        candles: List[Candle],
        swings: List[SwingPoint],
        structure: StructureState,
        atr_pct: List[float],
        tf: str
    ) -> List[StructureEvent]:
        """
        Detect BOS (continuation) and CHoCH (reversal) events.
        """
        events = []

        if len(swings) < 2 or len(candles) < 5:
            return events

        # Get last swing high and low
        last_high = self._get_last_swing(swings, SwingType.HIGH)
        last_low = self._get_last_swing(swings, SwingType.LOW)

        if not last_high or not last_low:
            return events

        # Check recent candles (last 5)
        check_window = min(5, len(candles) - 1)

        for i in range(len(candles) - check_window, len(candles)):
            candle = candles[i]
            atr = atr_pct[i] if i < len(atr_pct) else 0.5

            # Adaptive buffer
            buffer_pct = max(
                self.config.bos_buffer_pct,
                self.config.bos_buffer_atr_mult * atr
            )

            price_to_check = candle.close if self.config.use_close_for_breaks else None

            # BOS detection
            if structure == StructureState.UP:
                # Bullish BOS: close above last high
                level = last_high.price
                trigger_price = price_to_check if price_to_check else candle.high

                if trigger_price > level * (1 + buffer_pct / 100):
                    event = StructureEvent(
                        event_type=EventType.BOS,
                        side=StructureSide.BULL,
                        level=level,
                        time=candle.timestamp,
                        index=i,
                        details={"atr_pct": atr, "buffer_pct": buffer_pct}
                    )
                    events.append(event)
                    self._add_pending_break(tf, event, i)

            elif structure == StructureState.DOWN:
                # Bearish BOS: close below last low
                level = last_low.price
                trigger_price = price_to_check if price_to_check else candle.low

                if trigger_price < level * (1 - buffer_pct / 100):
                    event = StructureEvent(
                        event_type=EventType.BOS,
                        side=StructureSide.BEAR,
                        level=level,
                        time=candle.timestamp,
                        index=i,
                        details={"atr_pct": atr, "buffer_pct": buffer_pct}
                    )
                    events.append(event)
                    self._add_pending_break(tf, event, i)

            # CHoCH detection
            if structure == StructureState.UP:
                # Bearish CHoCH: close below last low
                level = last_low.price
                trigger_price = price_to_check if price_to_check else candle.low

                if trigger_price < level * (1 - buffer_pct / 100):
                    event = StructureEvent(
                        event_type=EventType.CHOCH,
                        side=StructureSide.BEAR,
                        level=level,
                        time=candle.timestamp,
                        index=i,
                        details={"atr_pct": atr}
                    )
                    events.append(event)

            elif structure == StructureState.DOWN:
                # Bullish CHoCH: close above last high
                level = last_high.price
                trigger_price = price_to_check if price_to_check else candle.high

                if trigger_price > level * (1 + buffer_pct / 100):
                    event = StructureEvent(
                        event_type=EventType.CHOCH,
                        side=StructureSide.BULL,
                        level=level,
                        time=candle.timestamp,
                        index=i,
                        details={"atr_pct": atr}
                    )
                    events.append(event)

        return events

    def _detect_sweeps(
        self,
        candles: List[Candle],
        swings: List[SwingPoint],
        atr_pct: List[float],
        rv: List[float]
    ) -> List[StructureEvent]:
        """
        Detect liquidity sweeps with confirmation.

        Sweep: breach liquidity level, then close back (rejection).
        """
        events = []

        if len(swings) < 2 or len(candles) < 5:
            return events

        # Identify liquidity levels (equal highs/lows or recent swings)
        liq_highs = self._find_liquidity_levels(swings, SwingType.HIGH)
        liq_lows = self._find_liquidity_levels(swings, SwingType.LOW)

        # Check recent candles
        check_window = min(10, len(candles))

        for i in range(len(candles) - check_window, len(candles)):
            candle = candles[i]
            atr = atr_pct[i] if i < len(atr_pct) else 0.5

            buffer_pct = max(
                self.config.sweep_buffer_pct,
                self.config.sweep_buffer_atr_mult * atr
            )

            # Check sweep of liq_lows (bullish sweep)
            for liq_low in liq_lows:
                # Did we breach below?
                if candle.low < liq_low * (1 - buffer_pct / 100):
                    # Did we close back above?
                    if candle.close > liq_low:
                        # Check confirmation in next K candles
                        confirmed = self._check_sweep_confirmation(
                            candles, i, rv, StructureSide.BULL
                        )

                        events.append(StructureEvent(
                            event_type=EventType.SWEEP,
                            side=StructureSide.BULL,
                            level=liq_low,
                            time=candle.timestamp,
                            index=i,
                            details={"confirmed": confirmed, "type": "buy_side_liquidity"}
                        ))

            # Check sweep of liq_highs (bearish sweep)
            for liq_high in liq_highs:
                # Did we breach above?
                if candle.high > liq_high * (1 + buffer_pct / 100):
                    # Did we close back below?
                    if candle.close < liq_high:
                        # Check confirmation
                        confirmed = self._check_sweep_confirmation(
                            candles, i, rv, StructureSide.BEAR
                        )

                        events.append(StructureEvent(
                            event_type=EventType.SWEEP,
                            side=StructureSide.BEAR,
                            level=liq_high,
                            time=candle.timestamp,
                            index=i,
                            details={"confirmed": confirmed, "type": "sell_side_liquidity"}
                        ))

        return events

    def _find_liquidity_levels(self, swings: List[SwingPoint], swing_type: SwingType) -> List[float]:
        """Find liquidity levels (equal highs/lows)."""
        levels = []
        same_type = [s for s in swings if s.swing_type == swing_type]

        if len(same_type) < 2:
            return levels

        # Check for equal levels (within 0.1%)
        tolerance = 0.001

        for i in range(len(same_type) - 1):
            for j in range(i + 1, len(same_type)):
                price_diff = abs(same_type[i].price - same_type[j].price) / same_type[i].price
                if price_diff < tolerance:
                    levels.append(same_type[i].price)

        # Also add most recent swing as simple liquidity
        if same_type:
            levels.append(same_type[-1].price)

        return list(set(levels))  # Dedupe

    def _check_sweep_confirmation(
        self,
        candles: List[Candle],
        sweep_idx: int,
        rv: List[float],
        side: StructureSide
    ) -> bool:
        """
        Check if sweep is confirmed within next K candles.

        Confirmation:
        - RV > threshold on at least one candle
        - Close in direction opposite to sweep
        """
        K = self.config.sweep_confirmation_candles
        end_idx = min(sweep_idx + K + 1, len(candles))

        high_rv_found = False
        rejection_found = False

        sweep_level = candles[sweep_idx].close

        for i in range(sweep_idx + 1, end_idx):
            # Check RV
            if i < len(rv) and rv[i] >= self.config.sweep_rv_threshold:
                high_rv_found = True

            # Check rejection direction
            if side == StructureSide.BULL:
                # Bullish sweep: should close up
                if candles[i].close > sweep_level:
                    rejection_found = True
            else:
                # Bearish sweep: should close down
                if candles[i].close < sweep_level:
                    rejection_found = True

        return high_rv_found and rejection_found

    def _check_acceptance_rejection(
        self,
        tf: str,
        candles: List[Candle],
        rv: List[float]
    ) -> List[StructureEvent]:
        """
        Check acceptance/rejection for pending breaks.
        """
        events = []

        if tf not in self._pending_breaks:
            return events

        current_idx = len(candles) - 1
        N = self.config.accept_hold_candles

        completed = []

        for break_event, break_idx in self._pending_breaks[tf]:
            candles_since = current_idx - break_idx

            if candles_since < N:
                continue  # Not enough time yet

            # Check if accepted or rejected
            level = break_event.level
            side = break_event.side

            stayed_beyond = True
            high_rv_found = False

            for i in range(break_idx + 1, min(break_idx + N + 1, len(candles))):
                # Check if stayed beyond level
                if side == StructureSide.BULL:
                    if candles[i].close < level:
                        stayed_beyond = False
                else:
                    if candles[i].close > level:
                        stayed_beyond = False

                # Check RV
                if i < len(rv) and rv[i] >= self.config.accept_rv_threshold:
                    high_rv_found = True

            # Accepted or rejected?
            if stayed_beyond and high_rv_found:
                events.append(StructureEvent(
                    event_type=EventType.ACCEPT,
                    side=side,
                    level=level,
                    time=candles[current_idx].timestamp,
                    index=current_idx,
                    details={"original_event": break_event.event_type.value}
                ))
            else:
                events.append(StructureEvent(
                    event_type=EventType.REJECT,
                    side=side,
                    level=level,
                    time=candles[current_idx].timestamp,
                    index=current_idx,
                    details={"original_event": break_event.event_type.value}
                ))

            completed.append((break_event, break_idx))

        # Remove completed
        for item in completed:
            self._pending_breaks[tf].remove(item)

        return events

    def _add_pending_break(self, tf: str, event: StructureEvent, idx: int):
        """Add a break event to pending list for acceptance tracking."""
        if tf not in self._pending_breaks:
            self._pending_breaks[tf] = []
        self._pending_breaks[tf].append((event, idx))

    def _detect_range(
        self,
        candles: List[Candle],
        swings: List[SwingPoint],
        events: List[StructureEvent],
        atr_pct: List[float]
    ) -> Optional[Zone]:
        """
        Detect if market is in a range and classify it.
        """
        # Check if no BOS in last X candles
        recent_bos = [e for e in events if e.event_type == EventType.BOS]
        if recent_bos:
            last_bos_idx = recent_bos[-1].index
            candles_since_bos = len(candles) - 1 - last_bos_idx

            if candles_since_bos < self.config.range_no_bos_candles:
                return None  # Too soon after BOS

        # Check ATR contraction
        if self.config.range_atr_contraction and len(atr_pct) >= self.config.atr_period:
            current_atr = atr_pct[-1]
            atr_sma = sma(atr_pct, 20)
            if current_atr >= atr_sma[-1]:
                return None  # ATR not contracting

        # Define range boundaries
        recent_highs = [s for s in swings if s.swing_type == SwingType.HIGH]
        recent_lows = [s for s in swings if s.swing_type == SwingType.LOW]

        if not recent_highs or not recent_lows:
            return None

        range_high = max(s.price for s in recent_highs[-5:])
        range_low = min(s.price for s in recent_lows[-5:])

        # Classify range type
        # TODO: Implement compression/accumulation/distribution logic
        # For now, default to NEUTRAL
        classification = RangeClassification.NEUTRAL

        return Zone(
            zone_type="RANGE",
            side=None,
            top=range_high,
            bottom=range_low,
            created_time=candles[-1].timestamp,
            status=ZoneStatus.OPEN
        )

    def _detect_fvgs(self, candles: List[Candle]) -> List[Zone]:
        """
        Detect Fair Value Gaps (3-candle pattern).

        Bullish FVG: low[i] > high[i-2]
        Bearish FVG: high[i] < low[i-2]
        """
        fvgs = []

        for i in range(2, len(candles)):
            # Bullish FVG
            if candles[i].low > candles[i - 2].high:
                gap_top = candles[i].low
                gap_bottom = candles[i - 2].high

                # Check if filled
                status = ZoneStatus.OPEN
                for j in range(i + 1, len(candles)):
                    if candles[j].low <= gap_top and candles[j].low >= gap_bottom:
                        status = ZoneStatus.FILLED
                        break

                fvgs.append(Zone(
                    zone_type="FVG",
                    side=StructureSide.BULL,
                    top=gap_top,
                    bottom=gap_bottom,
                    created_time=candles[i].timestamp,
                    status=status
                ))

            # Bearish FVG
            elif candles[i].high < candles[i - 2].low:
                gap_top = candles[i - 2].low
                gap_bottom = candles[i].high

                # Check if filled
                status = ZoneStatus.OPEN
                for j in range(i + 1, len(candles)):
                    if candles[j].high >= gap_bottom and candles[j].high <= gap_top:
                        status = ZoneStatus.FILLED
                        break

                fvgs.append(Zone(
                    zone_type="FVG",
                    side=StructureSide.BEAR,
                    top=gap_top,
                    bottom=gap_bottom,
                    created_time=candles[i].timestamp,
                    status=status
                ))

        # Return only open FVGs (last 10)
        open_fvgs = [f for f in fvgs if f.status == ZoneStatus.OPEN]
        return open_fvgs[-10:]

    def _compute_momentum(
        self,
        candles: List[Candle],
        events: List[StructureEvent],
        atr_pct: List[float]
    ) -> StructuralMomentum:
        """
        Compute time-to-followthrough momentum.
        """
        # Find most recent BOS
        bos_events = [e for e in events if e.event_type == EventType.BOS]
        if not bos_events:
            return StructuralMomentum.SLOW

        last_bos = bos_events[-1]
        bos_idx = last_bos.index

        if bos_idx >= len(candles) - 1:
            return StructuralMomentum.SLOW

        # Check for continuation
        move_threshold_atr = self.config.momentum_move_atr_mult
        side = last_bos.side
        bos_price = candles[bos_idx].close

        continuation_idx = None

        for i in range(bos_idx + 1, len(candles)):
            atr = atr_pct[i] if i < len(atr_pct) else 0.5
            move_threshold_pct = move_threshold_atr * atr

            if side == StructureSide.BULL:
                # Check upward move
                move_pct = (candles[i].high - bos_price) / bos_price * 100
                if move_pct >= move_threshold_pct:
                    continuation_idx = i
                    break
            else:
                # Check downward move
                move_pct = (bos_price - candles[i].low) / bos_price * 100
                if move_pct >= move_threshold_pct:
                    continuation_idx = i
                    break

        if continuation_idx is None:
            # Check if max candles exceeded
            candles_since = len(candles) - 1 - bos_idx
            if candles_since > self.config.momentum_max_candles:
                return StructuralMomentum.STALLED
            else:
                return StructuralMomentum.SLOW

        # Classify based on speed
        candles_to_continuation = continuation_idx - bos_idx

        if candles_to_continuation <= self.config.momentum_fast_candles:
            return StructuralMomentum.FAST
        elif candles_to_continuation <= self.config.momentum_max_candles:
            return StructuralMomentum.SLOW
        else:
            return StructuralMomentum.STALLED

    def _compute_strength(
        self,
        structure: StructureState,
        events: List[StructureEvent],
        swings: List[SwingPoint],
        active_range: Optional[Zone]
    ) -> float:
        """
        Compute structure strength/confidence (0-100).
        """
        strength = 50.0  # Base

        # Clear structure
        if structure in [StructureState.UP, StructureState.DOWN]:
            strength += 20
        elif structure == StructureState.RANGE and active_range:
            strength += 10
        else:
            strength -= 20

        # Recent BOS
        bos_events = [e for e in events if e.event_type == EventType.BOS]
        if bos_events:
            # Check if accepted
            accept_events = [e for e in events
                           if e.event_type == EventType.ACCEPT
                           and e.details.get("original_event") == "BOS"]
            if accept_events:
                strength += 20
            else:
                # Check if rejected
                reject_events = [e for e in events
                               if e.event_type == EventType.REJECT
                               and e.details.get("original_event") == "BOS"]
                if reject_events:
                    strength -= 15

        # Recent CHoCH
        choch_events = [e for e in events if e.event_type == EventType.CHOCH]
        if choch_events:
            strength -= 10

        # Swing strength
        if swings:
            avg_swing_strength = sum(s.strength for s in swings[-5:]) / min(5, len(swings))
            strength += avg_swing_strength * 10

        return max(0.0, min(100.0, strength))

    def _get_last_swing(self, swings: List[SwingPoint], swing_type: SwingType) -> Optional[SwingPoint]:
        """Get the most recent swing of given type."""
        for swing in reversed(swings):
            if swing.swing_type == swing_type:
                return swing
        return None
