"""
VWAP Engine Module - Volume-Weighted Average Price Analysis

Real-time VWAP engine supporting:
- Session VWAP (daily reset with timezone support)
- Weekly VWAP (weekly reset on configurable day)
- Anchored VWAP (multiple user-defined anchors)
- Standard deviation bands (with ATR fallback)
- Interaction state machine (ACCEPT/REJECT/RECLAIM/LOSS)

Features:
- Incremental O(1) updates per candle
- Volume-weighted standard deviation for bands
- Multi-timeframe support
- Configurable state confirmation with hold bars
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
from datetime import datetime, timezone, timedelta
import pytz
import math


class PriceSource(Enum):
    """Price source for VWAP calculation"""
    TYPICAL = "typical"  # (H+L+C)/3
    CLOSE = "close"      # Close only


class VWAPKind(Enum):
    """Type of VWAP line"""
    SESSION = "SESSION"
    WEEKLY = "WEEKLY"
    ANCHORED = "ANCHORED"


class PricePosition(Enum):
    """Price position relative to VWAP"""
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    AT = "AT"


class InteractionState(Enum):
    """VWAP interaction state"""
    ACCEPT = "ACCEPT"      # Staying on same side with confirmation
    REJECT = "REJECT"      # Touched and moved away
    RECLAIM = "RECLAIM"    # Crossed from below to above with hold
    LOSS = "LOSS"          # Crossed from above to below with hold
    NEUTRAL = "NEUTRAL"    # No clear interaction


class BandMethod(Enum):
    """Method used for band calculation"""
    STD = "STD"                    # Standard deviation bands
    ATR_FALLBACK = "ATR_FALLBACK"  # ATR-based fallback
    NONE = "NONE"                  # No bands available


@dataclass
class Candle:
    """OHLCV candle structure"""
    timestamp: float  # UTC seconds or milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class VWAPBands:
    """VWAP bands structure"""
    vwap: float
    std: Optional[float]
    bands: Dict[float, Tuple[float, float]]  # multiplier -> (lower, upper)
    method: BandMethod


@dataclass
class VWAPConfig:
    """Configuration for VWAP Engine"""

    # Price source
    price_source: PriceSource = PriceSource.TYPICAL

    # Timeframes
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # Session configuration
    session_reset: str = "UTC_DAY"  # or "LOCAL_DAY"
    timezone: str = "UTC"
    weekly_reset_day: str = "MON"  # ISO week start

    # Bands configuration
    enable_std_bands: bool = True
    std_band_multipliers: List[float] = field(default_factory=lambda: [1.0, 2.0])
    fallback_atr_band_multipliers: List[float] = field(default_factory=lambda: [0.5, 1.0])
    min_bars_for_std: int = 30

    # State machine parameters
    hold_bars: int = 3
    reclaim_tolerance: float = 0.0002  # 0.02%
    touch_tolerance: float = 0.0001    # 0.01%

    # Anchor configuration
    max_anchors_per_tf: int = 3
    default_anchor_types_enabled: List[str] = field(default_factory=lambda: [
        "BOS", "CHOCH", "SWEEP", "SESSION_HIGH", "SESSION_LOW"
    ])
    anchor_expire_mode: str = "AGE"  # or "RELEVANCE"
    anchor_max_age_bars: Dict[str, int] = field(default_factory=lambda: {
        "1m": 1440,
        "5m": 1000,
        "1h": 300
    })

    # ATR configuration (for fallback bands)
    atr_period: int = 14


@dataclass
class VWAPState:
    """State for a single VWAP line"""

    # Identity
    kind: VWAPKind
    anchor_id: Optional[str] = None
    anchor_time: Optional[float] = None
    anchor_note: str = ""

    # VWAP accumulators
    pv_sum: float = 0.0   # Σ(price * volume)
    v_sum: float = 0.0    # Σ(volume)
    p2v_sum: float = 0.0  # Σ(price^2 * volume) for std dev
    vwap: float = 0.0

    # Bands
    bands: Optional[VWAPBands] = None

    # Price interaction
    price_position: PricePosition = PricePosition.AT
    interaction_state: InteractionState = InteractionState.NEUTRAL
    distance: Dict[str, float] = field(default_factory=dict)

    # State machine tracking
    hold_count: int = 0
    prev_position: Optional[PricePosition] = None
    position_history: deque = field(default_factory=lambda: deque(maxlen=5))

    # Metadata
    bar_count: int = 0
    last_update_time: float = 0.0
    last_reset_time: float = 0.0
    debug: Dict = field(default_factory=dict)


@dataclass
class VWAPMultiTFState:
    """Multi-timeframe VWAP state"""
    session_by_tf: Dict[str, VWAPState] = field(default_factory=dict)
    weekly_by_tf: Dict[str, VWAPState] = field(default_factory=dict)
    anchors_by_tf: Dict[str, List[VWAPState]] = field(default_factory=dict)
    summary_by_tf: Dict[str, Dict] = field(default_factory=dict)


class VWAPEngine:
    """
    Real-time VWAP Engine with session, weekly, and anchored VWAP support.

    Provides incremental O(1) updates with volume-weighted standard deviation bands
    and interaction state tracking.
    """

    def __init__(self, config: Optional[VWAPConfig] = None):
        """
        Initialize VWAP Engine.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or VWAPConfig()

        # Per-timeframe state
        self._session_vwap: Dict[str, VWAPState] = {}
        self._weekly_vwap: Dict[str, VWAPState] = {}
        self._anchored_vwaps: Dict[str, List[VWAPState]] = {}

        # Timezone handling
        self._tz = pytz.timezone(self.config.timezone)

        # ATR tracking for fallback bands
        self._atr_by_tf: Dict[str, float] = {}

    def _get_price(self, candle: Candle) -> float:
        """
        Get price based on configured source.

        Args:
            candle: Input candle

        Returns:
            Price value (typical price or close)
        """
        if self.config.price_source == PriceSource.TYPICAL:
            return (candle.high + candle.low + candle.close) / 3.0
        else:
            return candle.close

    def _is_session_boundary(self, prev_time: float, curr_time: float) -> bool:
        """
        Check if session boundary crossed.

        Args:
            prev_time: Previous timestamp
            curr_time: Current timestamp

        Returns:
            True if session boundary crossed
        """
        if prev_time <= 0:
            return False

        if self.config.session_reset == "UTC_DAY":
            prev_dt = datetime.fromtimestamp(prev_time, tz=timezone.utc)
            curr_dt = datetime.fromtimestamp(curr_time, tz=timezone.utc)
            return prev_dt.date() != curr_dt.date()
        else:  # LOCAL_DAY
            prev_dt = datetime.fromtimestamp(prev_time, tz=self._tz)
            curr_dt = datetime.fromtimestamp(curr_time, tz=self._tz)
            return prev_dt.date() != curr_dt.date()

    def _is_weekly_boundary(self, prev_time: float, curr_time: float) -> bool:
        """
        Check if weekly boundary crossed.

        Args:
            prev_time: Previous timestamp
            curr_time: Current timestamp

        Returns:
            True if weekly boundary crossed
        """
        if prev_time <= 0:
            return False

        prev_dt = datetime.fromtimestamp(prev_time, tz=self._tz)
        curr_dt = datetime.fromtimestamp(curr_time, tz=self._tz)

        # Check if we crossed into a new ISO week (Monday start)
        prev_week = prev_dt.isocalendar()[1]
        curr_week = curr_dt.isocalendar()[1]
        prev_year = prev_dt.isocalendar()[0]
        curr_year = curr_dt.isocalendar()[0]

        return (prev_year, prev_week) != (curr_year, curr_week)

    def _update_vwap_accumulators(self, state: VWAPState, price: float,
                                  volume: float, timestamp: float) -> None:
        """
        Update VWAP accumulators incrementally.

        Args:
            state: VWAP state to update
            price: Price value (TP or close)
            volume: Volume
            timestamp: Current timestamp
        """
        eps = 1e-10

        # Skip zero volume
        if volume <= 0:
            return

        # Update accumulators
        state.pv_sum += price * volume
        state.v_sum += volume
        state.p2v_sum += (price * price) * volume

        # Calculate VWAP
        state.vwap = state.pv_sum / (state.v_sum + eps)

        # Update metadata
        state.bar_count += 1
        state.last_update_time = timestamp

    def _compute_std_bands(self, state: VWAPState, close: float,
                          atr_percent: Optional[float]) -> VWAPBands:
        """
        Compute standard deviation bands or ATR fallback.

        Args:
            state: VWAP state
            close: Current close price
            atr_percent: Optional ATR as percentage

        Returns:
            VWAPBands object
        """
        eps = 1e-10
        bands_dict = {}

        # Try standard deviation bands first
        if (self.config.enable_std_bands and
            state.bar_count >= self.config.min_bars_for_std and
            state.v_sum > eps):

            # Volume-weighted variance
            mean = state.vwap
            variance = (state.p2v_sum / state.v_sum) - (mean * mean)
            variance = max(variance, 0.0)  # Ensure non-negative
            std = math.sqrt(variance)

            # Only use if std is meaningful
            if std > eps:
                for k in self.config.std_band_multipliers:
                    lower = state.vwap - k * std
                    upper = state.vwap + k * std
                    bands_dict[k] = (lower, upper)

                return VWAPBands(
                    vwap=state.vwap,
                    std=std,
                    bands=bands_dict,
                    method=BandMethod.STD
                )

        # Fallback to ATR-based bands
        if atr_percent is not None and atr_percent > 0:
            for k in self.config.fallback_atr_band_multipliers:
                band_width = k * atr_percent * close
                lower = state.vwap - band_width
                upper = state.vwap + band_width
                bands_dict[k] = (lower, upper)

            return VWAPBands(
                vwap=state.vwap,
                std=None,
                bands=bands_dict,
                method=BandMethod.ATR_FALLBACK
            )

        # No bands available
        return VWAPBands(
            vwap=state.vwap,
            std=None,
            bands={},
            method=BandMethod.NONE
        )

    def _compute_price_position(self, close: float, vwap: float) -> PricePosition:
        """
        Compute price position relative to VWAP.

        Args:
            close: Current close price
            vwap: VWAP value

        Returns:
            PricePosition
        """
        eps = 1e-10
        dist_pct = (close - vwap) / (vwap + eps)

        if abs(dist_pct) <= self.config.touch_tolerance:
            return PricePosition.AT
        elif dist_pct > 0:
            return PricePosition.ABOVE
        else:
            return PricePosition.BELOW

    def _update_interaction_state(self, state: VWAPState, close: float) -> None:
        """
        Update interaction state machine with hold confirmation.

        Args:
            state: VWAP state to update
            close: Current close price
        """
        current_position = self._compute_price_position(close, state.vwap)

        # Track position history
        state.position_history.append(current_position)

        # Initialize if first update
        if state.prev_position is None:
            state.prev_position = current_position
            state.price_position = current_position
            state.interaction_state = InteractionState.NEUTRAL
            state.hold_count = 0
            return

        # Determine if we should check for state transition
        # We need to track the "previous stable position" for RECLAIM/LOSS detection
        if current_position != state.prev_position and current_position != PricePosition.AT:
            # Position changed significantly, reset hold counter
            state.hold_count = 1
        elif current_position == state.prev_position:
            # Position maintained, increment hold counter
            state.hold_count += 1
        # If current is AT, don't reset - let it accumulate

        # Always update the displayed position immediately
        state.price_position = current_position

        # Update state based on hold confirmation
        if state.hold_count >= self.config.hold_bars:
            recent_positions = list(state.position_history)[-self.config.hold_bars:]

            # Check for RECLAIM (was BELOW, now consistently ABOVE)
            if len(state.position_history) >= self.config.hold_bars + 1:
                # Look back before the recent hold window to see where we came from
                prev_stable_positions = list(state.position_history)[-(self.config.hold_bars+3):-(self.config.hold_bars)]
                came_from_below = any(p == PricePosition.BELOW for p in prev_stable_positions) if prev_stable_positions else False

                if came_from_below and all(p in [PricePosition.ABOVE, PricePosition.AT] for p in recent_positions):
                    if any(p == PricePosition.ABOVE for p in recent_positions):
                        state.interaction_state = InteractionState.RECLAIM
                        state.prev_position = current_position
                        return

            # Check for LOSS (was ABOVE, now consistently BELOW)
            if len(state.position_history) >= self.config.hold_bars + 1:
                prev_stable_positions = list(state.position_history)[-(self.config.hold_bars+3):-(self.config.hold_bars)]
                came_from_above = any(p == PricePosition.ABOVE for p in prev_stable_positions) if prev_stable_positions else False

                if came_from_above and all(p in [PricePosition.BELOW, PricePosition.AT] for p in recent_positions):
                    if any(p == PricePosition.BELOW for p in recent_positions):
                        state.interaction_state = InteractionState.LOSS
                        state.prev_position = current_position
                        return

            # ACCEPT: maintained same side with hold
            if current_position != PricePosition.AT:
                if state.interaction_state not in [InteractionState.RECLAIM, InteractionState.LOSS]:
                    state.interaction_state = InteractionState.ACCEPT

        # Check for REJECT (touch and bounce)
        if PricePosition.AT in state.position_history and len(state.position_history) >= 2:
            if state.position_history[-1] != PricePosition.AT:
                # Touched recently and now moved away
                if state.hold_count >= self.config.hold_bars:
                    state.interaction_state = InteractionState.REJECT

        # Update prev_position
        state.prev_position = current_position

    def _compute_distance_metrics(self, state: VWAPState, close: float) -> Dict[str, float]:
        """
        Compute distance metrics from VWAP.

        Args:
            state: VWAP state
            close: Current close price

        Returns:
            Dictionary with distance metrics
        """
        eps = 1e-10
        metrics = {}

        # Percentage distance
        dist_pct = (close - state.vwap) / (state.vwap + eps)
        metrics['pct'] = dist_pct

        # Sigma distance (if std available)
        if state.bands and state.bands.std is not None and state.bands.std > eps:
            dist_sigma = (close - state.vwap) / state.bands.std
            metrics['sigma'] = dist_sigma
        else:
            metrics['sigma'] = None

        return metrics

    def _reset_vwap_state(self, state: VWAPState, timestamp: float) -> None:
        """
        Reset VWAP accumulators for new session/week.

        Args:
            state: VWAP state to reset
            timestamp: Reset timestamp
        """
        state.pv_sum = 0.0
        state.v_sum = 0.0
        state.p2v_sum = 0.0
        state.vwap = 0.0
        state.bar_count = 0
        state.last_reset_time = timestamp
        state.bands = None
        state.position_history.clear()
        state.hold_count = 0
        state.prev_position = None

    def _get_or_create_session_vwap(self, tf: str, timestamp: float) -> VWAPState:
        """Get or create session VWAP state for timeframe"""
        if tf not in self._session_vwap:
            self._session_vwap[tf] = VWAPState(
                kind=VWAPKind.SESSION,
                last_reset_time=timestamp
            )
        return self._session_vwap[tf]

    def _get_or_create_weekly_vwap(self, tf: str, timestamp: float) -> VWAPState:
        """Get or create weekly VWAP state for timeframe"""
        if tf not in self._weekly_vwap:
            self._weekly_vwap[tf] = VWAPState(
                kind=VWAPKind.WEEKLY,
                last_reset_time=timestamp
            )
        return self._weekly_vwap[tf]

    def warmup(self, candles_by_tf: Dict[str, List[Candle]],
               atr_percent_by_tf: Optional[Dict[str, float]] = None) -> None:
        """
        Warm up VWAP engine with historical candles.

        Args:
            candles_by_tf: Dictionary of candle lists by timeframe
            atr_percent_by_tf: Optional ATR% values by timeframe
        """
        for tf, candles in candles_by_tf.items():
            atr_pct = atr_percent_by_tf.get(tf) if atr_percent_by_tf else None

            for candle in candles:
                self.on_candle_close(tf, candle, atr_pct)

    def on_candle_close(self, tf: str, candle: Candle,
                       atr_percent: Optional[float] = None) -> VWAPMultiTFState:
        """
        Process a candle close and update all VWAP lines for the timeframe.

        Args:
            tf: Timeframe string
            candle: Closed candle
            atr_percent: Optional ATR as percentage of close

        Returns:
            VWAPMultiTFState with all updated states
        """
        price = self._get_price(candle)
        timestamp = candle.timestamp

        # Update session VWAP
        session_state = self._get_or_create_session_vwap(tf, timestamp)

        # Check for session reset
        if self._is_session_boundary(session_state.last_update_time, timestamp):
            self._reset_vwap_state(session_state, timestamp)

        self._update_vwap_accumulators(session_state, price, candle.volume, timestamp)
        session_state.bands = self._compute_std_bands(session_state, candle.close, atr_percent)
        self._update_interaction_state(session_state, candle.close)
        session_state.distance = self._compute_distance_metrics(session_state, candle.close)

        # Update weekly VWAP
        weekly_state = self._get_or_create_weekly_vwap(tf, timestamp)

        # Check for weekly reset
        if self._is_weekly_boundary(weekly_state.last_update_time, timestamp):
            self._reset_vwap_state(weekly_state, timestamp)

        self._update_vwap_accumulators(weekly_state, price, candle.volume, timestamp)
        weekly_state.bands = self._compute_std_bands(weekly_state, candle.close, atr_percent)
        self._update_interaction_state(weekly_state, candle.close)
        weekly_state.distance = self._compute_distance_metrics(weekly_state, candle.close)

        # Update anchored VWAPs
        if tf not in self._anchored_vwaps:
            self._anchored_vwaps[tf] = []

        for anchor_state in self._anchored_vwaps[tf]:
            # Only update if candle is after anchor time
            if timestamp >= anchor_state.anchor_time:
                self._update_vwap_accumulators(anchor_state, price, candle.volume, timestamp)
                anchor_state.bands = self._compute_std_bands(anchor_state, candle.close, atr_percent)
                self._update_interaction_state(anchor_state, candle.close)
                anchor_state.distance = self._compute_distance_metrics(anchor_state, candle.close)

        # Build result
        result = VWAPMultiTFState()
        result.session_by_tf[tf] = session_state
        result.weekly_by_tf[tf] = weekly_state
        result.anchors_by_tf[tf] = self._anchored_vwaps.get(tf, [])

        # Build summary
        result.summary_by_tf[tf] = {
            'session_vwap': session_state.vwap,
            'weekly_vwap': weekly_state.vwap,
            'session_state': session_state.interaction_state.value,
            'num_anchors': len(self._anchored_vwaps.get(tf, []))
        }

        return result

    def add_anchor(self, tf: str, anchor_time: float, anchor_id: str,
                   note: str = "", kind: str = "EVENT") -> VWAPState:
        """
        Add an anchored VWAP starting at specified time.

        Args:
            tf: Timeframe
            anchor_time: Timestamp to anchor from
            anchor_id: Unique identifier for this anchor
            note: Optional note/description
            kind: Type of anchor event

        Returns:
            Created VWAPState
        """
        if tf not in self._anchored_vwaps:
            self._anchored_vwaps[tf] = []

        # Check if anchor already exists
        for existing in self._anchored_vwaps[tf]:
            if existing.anchor_id == anchor_id:
                return existing

        # Create new anchored VWAP
        anchor_state = VWAPState(
            kind=VWAPKind.ANCHORED,
            anchor_id=anchor_id,
            anchor_time=anchor_time,
            anchor_note=note,
            last_reset_time=anchor_time
        )

        self._anchored_vwaps[tf].append(anchor_state)

        # Prune if too many anchors
        self.prune_anchors(tf, anchor_time)

        return anchor_state

    def remove_anchor(self, tf: str, anchor_id: str) -> bool:
        """
        Remove an anchored VWAP.

        Args:
            tf: Timeframe
            anchor_id: Anchor identifier

        Returns:
            True if removed, False if not found
        """
        if tf not in self._anchored_vwaps:
            return False

        original_len = len(self._anchored_vwaps[tf])
        self._anchored_vwaps[tf] = [
            a for a in self._anchored_vwaps[tf] if a.anchor_id != anchor_id
        ]

        return len(self._anchored_vwaps[tf]) < original_len

    def prune_anchors(self, tf: str, now_time: float) -> int:
        """
        Prune expired or excess anchors.

        Args:
            tf: Timeframe
            now_time: Current timestamp

        Returns:
            Number of anchors pruned
        """
        if tf not in self._anchored_vwaps:
            return 0

        anchors = self._anchored_vwaps[tf]
        original_count = len(anchors)

        # Prune by age if configured
        if self.config.anchor_expire_mode == "AGE":
            max_age = self.config.anchor_max_age_bars.get(tf, 1000)

            # Keep only anchors within age limit
            anchors = [a for a in anchors if a.bar_count <= max_age]

        # Keep only max_anchors_per_tf most recent
        if len(anchors) > self.config.max_anchors_per_tf:
            # Sort by anchor time, keep newest
            anchors.sort(key=lambda a: a.anchor_time, reverse=True)
            anchors = anchors[:self.config.max_anchors_per_tf]

        self._anchored_vwaps[tf] = anchors
        return original_count - len(anchors)

    def update(self, candles_by_tf: Dict[str, List[Candle]],
               atr_percent_by_tf: Optional[Dict[str, float]] = None) -> VWAPMultiTFState:
        """
        Convenience method to update all timeframes.

        Args:
            candles_by_tf: Dictionary of candle lists by timeframe (latest candle last)
            atr_percent_by_tf: Optional ATR% values by timeframe

        Returns:
            VWAPMultiTFState with all timeframes
        """
        result = VWAPMultiTFState()

        for tf, candles in candles_by_tf.items():
            if not candles:
                continue

            latest_candle = candles[-1]
            atr_percent = atr_percent_by_tf.get(tf) if atr_percent_by_tf else None

            tf_result = self.on_candle_close(tf, latest_candle, atr_percent)

            # Merge results
            result.session_by_tf.update(tf_result.session_by_tf)
            result.weekly_by_tf.update(tf_result.weekly_by_tf)
            result.anchors_by_tf.update(tf_result.anchors_by_tf)
            result.summary_by_tf.update(tf_result.summary_by_tf)

        return result


def format_vwap_output(vwap_state: VWAPMultiTFState, compact: bool = True) -> str:
    """
    Format VWAP states for display.

    Args:
        vwap_state: VWAP multi-timeframe state
        compact: If True, use compact format

    Returns:
        Formatted string
    """
    lines = ["VWAP CONTEXT"]

    for tf in sorted(vwap_state.session_by_tf.keys()):
        session = vwap_state.session_by_tf.get(tf)
        weekly = vwap_state.weekly_by_tf.get(tf)
        anchors = vwap_state.anchors_by_tf.get(tf, [])

        if compact:
            # Session VWAP
            if session:
                dist_pct = session.distance.get('pct', 0) * 100
                dist_sigma = session.distance.get('sigma')
                sigma_str = f" ({dist_sigma:+.1f}σ)" if dist_sigma is not None else ""
                band_method = session.bands.method.value if session.bands else "NONE"

                line = (f"{tf} Session: {session.price_position.value:5s} | "
                       f"{session.interaction_state.value:7s} | "
                       f"dist={dist_pct:+.2f}%{sigma_str:10s} | "
                       f"bands={band_method}")
                lines.append(line)

            # Weekly VWAP
            if weekly:
                dist_pct = weekly.distance.get('pct', 0) * 100
                dist_sigma = weekly.distance.get('sigma')
                sigma_str = f" ({dist_sigma:+.1f}σ)" if dist_sigma is not None else ""

                line = (f"{tf} Weekly : {weekly.price_position.value:5s} | "
                       f"{weekly.interaction_state.value:7s} | "
                       f"dist={dist_pct:+.2f}%{sigma_str}")
                lines.append(line)

            # Anchors
            if anchors:
                lines.append(f"{tf} Anchors:")
                for anchor in anchors:
                    dist_pct = anchor.distance.get('pct', 0) * 100
                    note = anchor.anchor_note or anchor.anchor_id
                    line = (f"  - {note}: {anchor.interaction_state.value} | "
                           f"dist={dist_pct:+.2f}% | vwap={anchor.vwap:.2f}")
                    lines.append(line)

        else:
            # Verbose format
            lines.append(f"\n{tf}:")

            if session:
                lines.append(f"  Session VWAP: {session.vwap:.2f}")
                lines.append(f"    Position: {session.price_position.value}")
                lines.append(f"    State: {session.interaction_state.value}")
                lines.append(f"    Distance: {session.distance.get('pct', 0)*100:+.2f}%")

            if weekly:
                lines.append(f"  Weekly VWAP: {weekly.vwap:.2f}")
                lines.append(f"    State: {weekly.interaction_state.value}")

            if anchors:
                lines.append(f"  Anchored VWAPs: {len(anchors)}")
                for anchor in anchors:
                    lines.append(f"    - {anchor.anchor_id}: {anchor.vwap:.2f}")

    return "\n".join(lines)
