use std::collections::VecDeque;

use infra_adapters::Trade;
use infra_core::{NanosDelta, UnixNanos};

/// Trade flow imbalance and trade notional ratios are normalized by the notional traded over
/// this long.
const NOTIONAL_NORM_WINDOW: NanosDelta = NanosDelta::from_secs(60 * 60);

pub(super) struct TradeFactors {
    /// Receive time and signed notional of each trade within `NOTIONAL_NORM_WINDOW`, oldest
    /// first.
    trade_notionals: VecDeque<(UnixNanos, f64)>,
    /// Unsigned notional summed over `trade_notionals`.
    norm_notional: f64,
}

impl TradeFactors {
    pub(super) fn new() -> Self {
        Self {
            trade_notionals: VecDeque::new(),
            norm_notional: 0.0,
        }
    }

    pub(super) fn on_trade(&mut self, trade: Trade) {
        let notional = trade_notional(&trade);
        self.trade_notionals
            .push_back((trade.receive_time, notional));
        self.norm_notional += notional.abs();
        let cutoff = trade.receive_time - NOTIONAL_NORM_WINDOW;
        while let Some(&(t, notional)) = self.trade_notionals.front() {
            if t > cutoff {
                break;
            }
            self.norm_notional -= notional.abs();
            self.trade_notionals.pop_front();
        }
    }

    /// Trade flow imbalance: aggressive buy minus aggressive sell notional of the trades
    /// received in `(now - window, now]`, normalized as described on
    /// `trade_notional_ratio`.
    ///
    /// # Panics
    /// Panics if `window` exceeds `NOTIONAL_NORM_WINDOW`.
    pub(super) fn tfi(&self, now: UnixNanos, window: NanosDelta) -> f64 {
        self.normalized_notional(now, window, |notional| notional)
    }

    /// Notional of the trades received in `(now - window, now]`, both aggressor sides, over
    /// the total notional of `NOTIONAL_NORM_WINDOW` scaled to `window`; 1.0 is a typical
    /// rate.
    ///
    /// # Panics
    /// Panics if `window` exceeds `NOTIONAL_NORM_WINDOW`.
    pub(super) fn trade_notional_ratio(&self, now: UnixNanos, window: NanosDelta) -> f64 {
        self.normalized_notional(now, window, f64::abs)
    }

    /// Sum of `contribution` of the signed notional of the trades in `(now - window, now]`,
    /// normalized as described on `trade_notional_ratio`. NaN if nothing traded within
    /// `NOTIONAL_NORM_WINDOW`.
    fn normalized_notional(
        &self,
        now: UnixNanos,
        window: NanosDelta,
        contribution: fn(f64) -> f64,
    ) -> f64 {
        assert!(
            window <= NOTIONAL_NORM_WINDOW,
            "window exceeds NOTIONAL_NORM_WINDOW"
        );
        if self.norm_notional <= 0.0 {
            return f64::NAN;
        }
        let start = now - window;
        let notional: f64 = self
            .trade_notionals
            .iter()
            .rev()
            .take_while(|&&(t, _)| t > start)
            .map(|&(_, notional)| contribution(notional))
            .sum();
        let expected =
            self.norm_notional * window.as_nanos() as f64 / NOTIONAL_NORM_WINDOW.as_nanos() as f64;
        notional / expected
    }
}

/// The trade's quantity, positive when the aggressor bought and negative when it sold.
fn trade_flow(trade: &Trade) -> f64 {
    if trade.is_buyer_maker {
        -trade.qty.to_f64()
    } else {
        trade.qty.to_f64()
    }
}

/// The trade's notional, positive when the aggressor bought and negative when it sold.
fn trade_notional(trade: &Trade) -> f64 {
    trade.price.to_f64() * trade_flow(trade)
}
