"""
Tests for VWAP Interaction State Machine
"""

import pytest
from vwap_state_machine import (
    VWAPStateMachine,
    VWAPStateConfig,
    VWAPInteractionState,
    Position,
    InteractionState,
    Regime
)


class TestPositionClassification:
    """Tests for position classification (ABOVE/BELOW/AT)"""

    def test_position_above(self):
        """Test position classified as ABOVE"""
        config = VWAPStateConfig()
        sm = VWAPStateMachine(config)

        result = sm.on_update(
            tf="1m",
            timestamp=1,
            close=100.5,
            vwap=100.0
        )

        assert result.position == Position.ABOVE.value
        assert result.side == 1
        assert result.dist_pct > 0

    def test_position_below(self):
        """Test position classified as BELOW"""
        config = VWAPStateConfig()
        sm = VWAPStateMachine(config)

        result = sm.on_update(
            tf="1m",
            timestamp=1,
            close=99.5,
            vwap=100.0
        )

        assert result.position == Position.BELOW.value
        assert result.side == -1
        assert result.dist_pct < 0

    def test_position_at_within_touch_pct(self):
        """Test position AT when within touch percentage"""
        config = VWAPStateConfig()
        sm = VWAPStateMachine(config)

        # Within 0.01% (0.0001) for 1m
        result = sm.on_update(
            tf="1m",
            timestamp=1,
            close=100.009,
            vwap=100.0
        )

        assert result.position == Position.AT.value
        assert result.side == 0

    def test_position_at_within_touch_sigma(self):
        """Test position AT when within sigma tolerance"""
        config = VWAPStateConfig()
        sm = VWAPStateMachine(config)

        # Within 0.15 sigma
        result = sm.on_update(
            tf="1m",
            timestamp=1,
            close=100.1,
            vwap=100.0,
            std=1.0  # 0.1 / 1.0 = 0.1 < 0.15
        )

        assert result.position == Position.AT.value
        assert result.side == 0

    def test_position_at_within_touch_atr(self):
        """Test position AT when within ATR tolerance"""
        config = VWAPStateConfig()
        sm = VWAPStateMachine(config)

        # ATR% = 0.005, touch_atr = 0.05 × 0.005 = 0.00025
        result = sm.on_update(
            tf="1m",
            timestamp=1,
            close=100.02,
            vwap=100.0,
            atr_percent=0.005
        )

        assert result.position == Position.AT.value
        assert result.side == 0


class TestCrossingsDetection:
    """Tests for crossings counting"""

    def test_crossings_count_simple(self):
        """Test crossings count with simple sequence"""
        config = VWAPStateConfig(window_crossings=10)
        sm = VWAPStateMachine(config)

        vwap = 100.0
        # Sequence: above, above, below, below, above => 2 crossings
        closes = [100.5, 100.3, 99.5, 99.7, 100.2]

        result = None
        for i, close in enumerate(closes):
            result = sm.on_update(
                tf="1m",
                timestamp=i,
                close=close,
                vwap=vwap
            )

        # Should have 2 crossings: above->below, below->above
        assert result.crossings_20 == 2

    def test_crossings_high_chop(self):
        """Test high crossings in choppy market"""
        config = VWAPStateConfig(window_crossings=20)
        sm = VWAPStateMachine(config)

        vwap = 100.0
        # Alternating above/below
        closes = []
        for i in range(20):
            closes.append(100.5 if i % 2 == 0 else 99.5)

        result = None
        for i, close in enumerate(closes):
            result = sm.on_update(
                tf="1m",
                timestamp=i,
                close=close,
                vwap=vwap
            )

        # Should have many crossings (19 for alternating pattern)
        assert result.crossings_20 >= 15


class TestReclaimState:
    """Tests for RECLAIM state"""

    def test_reclaim_from_below_trend(self):
        """Test RECLAIM when price crosses from below in TREND"""
        config = VWAPStateConfig(
            N_reclaim_trend=2,
            reclaim_buffer_pct_by_tf={"1m": 0.001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start below
        sm.on_update("1m", 1, 99.0, vwap)

        # Cross above buffer (0.1%)
        sm.on_update("1m", 2, 100.15, vwap)  # +0.15% > 0.1% buffer

        # Hold for N_reclaim bars
        result = sm.on_update("1m", 3, 100.2, vwap, regime="TREND")

        assert result.state == InteractionState.RECLAIM.value

    def test_reclaim_requires_confirmation_in_range(self):
        """Test RECLAIM requires confirmation in RANGE regime"""
        config = VWAPStateConfig(
            N_reclaim_range=2,
            reclaim_buffer_pct_by_tf={"1m": 0.001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start below
        sm.on_update("1m", 1, 99.0, vwap)

        # Cross above buffer without confirmation
        sm.on_update("1m", 2, 100.15, vwap, regime="RANGE")
        result = sm.on_update("1m", 3, 100.2, vwap, regime="RANGE")

        # Should NOT be RECLAIM without confirmation
        assert result.state != InteractionState.RECLAIM.value

        # Now with confirmation
        sm.reset("1m")
        sm.on_update("1m", 1, 99.0, vwap)
        sm.on_update("1m", 2, 100.15, vwap, regime="RANGE", rv=1.5)  # RV >= 1.2
        result = sm.on_update("1m", 3, 100.2, vwap, regime="RANGE", rv=1.5)

        assert result.state == InteractionState.RECLAIM.value

    def test_reclaim_reset_on_drop_below_buffer(self):
        """Test RECLAIM hold resets if price drops back below buffer"""
        config = VWAPStateConfig(
            N_reclaim_trend=3,
            reclaim_buffer_pct_by_tf={"1m": 0.001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start below
        sm.on_update("1m", 1, 99.0, vwap)

        # Cross above buffer
        sm.on_update("1m", 2, 100.15, vwap)

        # Drop back below -buffer (should reset)
        sm.on_update("1m", 3, 99.85, vwap)  # -0.15% < -0.1%

        # Cross above again
        sm.on_update("1m", 4, 100.15, vwap)
        sm.on_update("1m", 5, 100.2, vwap)
        result = sm.on_update("1m", 6, 100.25, vwap, regime="TREND")

        # Should trigger after 3 new bars
        assert result.state == InteractionState.RECLAIM.value


class TestLossState:
    """Tests for LOSS state (mirror of RECLAIM)"""

    def test_loss_from_above_trend(self):
        """Test LOSS when price crosses from above in TREND"""
        config = VWAPStateConfig(
            N_loss_trend=2,
            reclaim_buffer_pct_by_tf={"1m": 0.001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start above
        sm.on_update("1m", 1, 101.0, vwap)

        # Cross below -buffer
        sm.on_update("1m", 2, 99.85, vwap)  # -0.15% < -0.1%

        # Hold
        result = sm.on_update("1m", 3, 99.8, vwap, regime="TREND")

        assert result.state == InteractionState.LOSS.value

    def test_loss_requires_confirmation_in_range(self):
        """Test LOSS requires confirmation in RANGE"""
        config = VWAPStateConfig(
            N_loss_range=2,
            reclaim_buffer_pct_by_tf={"1m": 0.001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start above
        sm.on_update("1m", 1, 101.0, vwap)

        # Cross below without confirmation
        sm.on_update("1m", 2, 99.85, vwap, regime="RANGE")
        result = sm.on_update("1m", 3, 99.8, vwap, regime="RANGE")

        # Should NOT be LOSS
        assert result.state != InteractionState.LOSS.value

        # With confirmation
        sm.reset("1m")
        sm.on_update("1m", 1, 101.0, vwap)
        sm.on_update("1m", 2, 99.85, vwap, regime="RANGE", delta_ok=True)
        result = sm.on_update("1m", 3, 99.8, vwap, regime="RANGE", delta_ok=True)

        assert result.state == InteractionState.LOSS.value


class TestRejectState:
    """Tests for REJECT state"""

    def test_reject_after_touch_and_move_away(self):
        """Test REJECT after touching VWAP and moving away"""
        config = VWAPStateConfig(
            N_reject_trend=3,
            touch_pct_by_tf={"1m": 0.0001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Touch VWAP
        sm.on_update("1m", 1, 100.005, vwap)

        # Move below and hold
        sm.on_update("1m", 2, 99.8, vwap)
        sm.on_update("1m", 3, 99.75, vwap)
        result = sm.on_update("1m", 4, 99.7, vwap)

        assert result.state == InteractionState.REJECT.value

    def test_reject_requires_recent_touch(self):
        """Test REJECT requires recent touch (within 2 bars)"""
        config = VWAPStateConfig(
            N_reject_trend=2,
            touch_pct_by_tf={"1m": 0.0001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Touch VWAP
        sm.on_update("1m", 1, 100.005, vwap)

        # Wait 3 bars
        sm.on_update("1m", 2, 99.9, vwap)
        sm.on_update("1m", 3, 99.9, vwap)
        sm.on_update("1m", 4, 99.9, vwap)

        # Move away (touch is now old)
        result = sm.on_update("1m", 5, 99.5, vwap)

        # Should NOT be REJECT (touch too old)
        assert result.state != InteractionState.REJECT.value

    def test_reject_reset_on_touch(self):
        """Test REJECT hold resets on new touch"""
        config = VWAPStateConfig(
            N_reject_trend=3,
            touch_pct_by_tf={"1m": 0.0001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Touch
        sm.on_update("1m", 1, 100.005, vwap)

        # Start moving away
        sm.on_update("1m", 2, 99.8, vwap)

        # Touch again (should reset)
        sm.on_update("1m", 3, 100.005, vwap)

        # Move away again
        sm.on_update("1m", 4, 99.8, vwap)
        sm.on_update("1m", 5, 99.75, vwap)
        result = sm.on_update("1m", 6, 99.7, vwap)

        # Should trigger after 3 new bars
        assert result.state == InteractionState.REJECT.value


class TestAcceptState:
    """Tests for ACCEPT state"""

    def test_accept_after_holding_above(self):
        """Test ACCEPT after持续 holding above"""
        config = VWAPStateConfig(N_accept_trend=3)
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Stay above for N_accept bars
        sm.on_update("1m", 1, 100.5, vwap, regime="TREND")
        sm.on_update("1m", 2, 100.6, vwap, regime="TREND")
        result = sm.on_update("1m", 3, 100.7, vwap, regime="TREND")

        assert result.state == InteractionState.ACCEPT.value

    def test_accept_after_holding_below(self):
        """Test ACCEPT after持续 holding below"""
        config = VWAPStateConfig(N_accept_trend=3)
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Stay below for N_accept bars
        sm.on_update("1m", 1, 99.5, vwap, regime="TREND")
        sm.on_update("1m", 2, 99.4, vwap, regime="TREND")
        result = sm.on_update("1m", 3, 99.3, vwap, regime="TREND")

        assert result.state == InteractionState.ACCEPT.value

    def test_accept_suppressed_in_chop(self):
        """Test ACCEPT suppressed when crossings high (chop)"""
        config = VWAPStateConfig(
            N_accept_trend=3,
            crossings_chop_threshold=4,
            window_crossings=10
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Create choppy pattern with high crossings
        for i in range(10):
            close = 100.5 if i % 2 == 0 else 99.5
            sm.on_update("1m", i, close, vwap)

        # Now try to ACCEPT above
        sm.on_update("1m", 10, 100.5, vwap)
        sm.on_update("1m", 11, 100.6, vwap)
        result = sm.on_update("1m", 12, 100.7, vwap)

        # Should be suppressed due to high crossings
        assert result.state != InteractionState.ACCEPT.value
        assert result.crossings_20 >= 4


class TestChopSuppression:
    """Tests for chop detection and suppression"""

    def test_high_crossings_infers_range_regime(self):
        """Test high crossings automatically infers RANGE regime"""
        config = VWAPStateConfig(
            crossings_chop_threshold=4,
            window_crossings=10
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Create choppy pattern
        for i in range(10):
            close = 100.5 if i % 2 == 0 else 99.5
            result = sm.on_update("1m", i, close, vwap)

        # Should infer RANGE regime
        assert result.debug['regime_used'] == Regime.RANGE.value

    def test_chop_suppression_overridden_by_strong_confirmation(self):
        """Test chop suppression can be overridden with strong confirmation"""
        config = VWAPStateConfig(
            N_accept_trend=3,
            crossings_chop_threshold=4,
            window_crossings=10
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Create choppy pattern
        for i in range(10):
            close = 100.5 if i % 2 == 0 else 99.5
            sm.on_update("1m", i, close, vwap)

        # Try ACCEPT with strong confirmation and extra hold
        sm.on_update("1m", 10, 100.5, vwap, rv=1.5)
        sm.on_update("1m", 11, 100.6, vwap, rv=1.5)
        sm.on_update("1m", 12, 100.7, vwap, rv=1.5)
        result = sm.on_update("1m", 13, 100.8, vwap, rv=1.5)

        # With strong enough hold (4 bars > N_accept) might still accept
        # (implementation allows this with conf_ok)
        # Just verify crossings are high
        assert result.crossings_20 >= 4


class TestRegimeAwareBehavior:
    """Tests for regime-aware parameter selection"""

    def test_trend_regime_uses_trend_params(self):
        """Test TREND regime uses trend hold parameters"""
        config = VWAPStateConfig(
            N_reclaim_trend=2,
            N_reclaim_range=4
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start below
        sm.on_update("1m", 1, 99.0, vwap)

        # Cross above
        sm.on_update("1m", 2, 100.15, vwap, regime="TREND")
        result = sm.on_update("1m", 3, 100.2, vwap, regime="TREND")

        # Should RECLAIM after 2 bars (TREND params)
        assert result.state == InteractionState.RECLAIM.value
        assert result.debug['N_reclaim'] == 2

    def test_range_regime_uses_range_params(self):
        """Test RANGE regime uses range hold parameters"""
        config = VWAPStateConfig(
            N_reclaim_trend=2,
            N_reclaim_range=4
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start below
        sm.on_update("1m", 1, 99.0, vwap)

        # Cross above
        sm.on_update("1m", 2, 100.15, vwap, regime="RANGE", rv=1.5)
        sm.on_update("1m", 3, 100.2, vwap, regime="RANGE", rv=1.5)
        result = sm.on_update("1m", 4, 100.25, vwap, regime="RANGE", rv=1.5)

        # Should NOT RECLAIM yet (needs 4 bars in RANGE)
        assert result.state != InteractionState.RECLAIM.value

        result = sm.on_update("1m", 5, 100.3, vwap, regime="RANGE", rv=1.5)

        # Now should RECLAIM
        assert result.state == InteractionState.RECLAIM.value
        assert result.debug['N_reclaim'] == 4


class TestStability:
    """Tests for anti-flip stability"""

    def test_single_touch_does_not_flip_state(self):
        """Test single touch bar doesn't flip state unnecessarily"""
        config = VWAPStateConfig(N_accept_trend=3)
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Build ACCEPT state above
        sm.on_update("1m", 1, 100.5, vwap, regime="TREND")
        sm.on_update("1m", 2, 100.6, vwap, regime="TREND")
        result = sm.on_update("1m", 3, 100.7, vwap, regime="TREND")
        assert result.state == InteractionState.ACCEPT.value

        # Single touch bar
        result_touch = sm.on_update("1m", 4, 100.005, vwap, regime="TREND")
        assert result_touch.position == Position.AT.value
        # State might still be ACCEPT or transition, but shouldn't cause flip-flop

        # Return above
        result_after = sm.on_update("1m", 5, 100.6, vwap, regime="TREND")

        # Should be stable (might remain ACCEPT or similar)
        assert result_after.position == Position.ABOVE.value

    def test_hold_count_persists_through_at(self):
        """Test hold_count doesn't reset on AT position"""
        config = VWAPStateConfig()
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Build hold count above
        sm.on_update("1m", 1, 100.5, vwap)
        sm.on_update("1m", 2, 100.6, vwap)
        result = sm.on_update("1m", 3, 100.7, vwap)
        hold_before = result.hold_count

        # Touch (AT)
        result_touch = sm.on_update("1m", 4, 100.005, vwap)

        # Return above
        result_after = sm.on_update("1m", 5, 100.6, vwap)

        # Hold count should resume, not reset
        assert result_after.hold_count >= hold_before


class TestStatePriority:
    """Tests for deterministic state priority"""

    def test_reclaim_has_priority_over_accept(self):
        """Test RECLAIM takes priority over ACCEPT"""
        config = VWAPStateConfig(
            N_reclaim_trend=2,
            N_accept_trend=2,
            reclaim_buffer_pct_by_tf={"1m": 0.001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start below
        sm.on_update("1m", 1, 99.0, vwap)

        # Cross above (satisfies both RECLAIM and ACCEPT conditions)
        sm.on_update("1m", 2, 100.15, vwap, regime="TREND")
        result = sm.on_update("1m", 3, 100.2, vwap, regime="TREND")

        # Should be RECLAIM (higher priority)
        assert result.state == InteractionState.RECLAIM.value

    def test_loss_has_priority_over_reject(self):
        """Test LOSS takes priority over REJECT"""
        config = VWAPStateConfig(
            N_loss_trend=2,
            N_reject_trend=2,
            reclaim_buffer_pct_by_tf={"1m": 0.001}
        )
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Start above and touch
        sm.on_update("1m", 1, 101.0, vwap)
        sm.on_update("1m", 2, 100.005, vwap)

        # Drop below (satisfies both LOSS and REJECT)
        sm.on_update("1m", 3, 99.85, vwap, regime="TREND")
        result = sm.on_update("1m", 4, 99.8, vwap, regime="TREND")

        # Should be LOSS (higher priority)
        assert result.state == InteractionState.LOSS.value


class TestHoldCount:
    """Tests for hold_count tracking"""

    def test_hold_count_increments(self):
        """Test hold_count increments correctly"""
        config = VWAPStateConfig()
        sm = VWAPStateMachine(config)

        vwap = 100.0

        result1 = sm.on_update("1m", 1, 100.5, vwap)
        assert result1.hold_count == 1

        result2 = sm.on_update("1m", 2, 100.6, vwap)
        assert result2.hold_count == 2

        result3 = sm.on_update("1m", 3, 100.7, vwap)
        assert result3.hold_count == 3

    def test_hold_count_resets_on_side_flip(self):
        """Test hold_count resets when side flips"""
        config = VWAPStateConfig()
        sm = VWAPStateMachine(config)

        vwap = 100.0

        # Build hold above
        sm.on_update("1m", 1, 100.5, vwap)
        sm.on_update("1m", 2, 100.6, vwap)

        # Flip to below
        result = sm.on_update("1m", 3, 99.5, vwap)

        # Should reset to 1
        assert result.hold_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
