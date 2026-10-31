use std::collections::VecDeque;

use infra_adapters::Bbo;
use infra_core::{NanosDelta, UnixNanos};

/// History older than this is dropped, so no factor window may exceed it.
const MAX_WINDOW: NanosDelta = NanosDelta::from_secs(60);
/// Order flow imbalance is normalized by the time-weighted average L1 depth over this long.
const DEPTH_AVG_WINDOW: NanosDelta = MAX_WINDOW;

pub(super) struct BboFactors {
    /// Bbo updates within `MAX_WINDOW`, oldest first, plus the last one before it so the
    /// bbo as of any window start is known.
    bbos: VecDeque<Bbo>,
    /// Time integral of the L1 depth, in quantity-seconds, from the first to the last bbo in
    /// `bbos`.
    depth_area: f64,
    /// Receive time and order flow of each bbo update within `MAX_WINDOW`, oldest first.
    order_flows: VecDeque<(UnixNanos, f64)>,
}

impl BboFactors {
    pub(super) fn new() -> Self {
        Self {
            bbos: VecDeque::new(),
            depth_area: 0.0,
            order_flows: VecDeque::new(),
        }
    }

    pub(super) fn on_bbo(&mut self, bbo: Bbo) {
        if let Some(prev) = self.bbos.back() {
            self.order_flows
                .push_back((bbo.receive_time, order_flow(prev, &bbo)));
            self.depth_area += l1_depth(prev) * secs(bbo.receive_time - prev.receive_time);
        }
        self.bbos.push_back(bbo);

        let cutoff = bbo.receive_time - MAX_WINDOW;
        while self.order_flows.front().is_some_and(|&(t, _)| t <= cutoff) {
            self.order_flows.pop_front();
        }
        while let Some(next) = self.bbos.get(1).filter(|b| b.receive_time <= cutoff) {
            let first = &self.bbos[0];
            self.depth_area -= l1_depth(first) * secs(next.receive_time - first.receive_time);
            self.bbos.pop_front();
        }
    }

    /// Order flow imbalance (Cont, Kukanov & Stoikov): the summed order flow of the bbo
    /// updates received in `(now - window, now]`, over the time-weighted average L1 depth of
    /// `(now - DEPTH_AVG_WINDOW, now]`. NaN if that average is unknown or zero.
    ///
    /// # Panics
    /// Panics if `window` exceeds `MAX_WINDOW`.
    pub(super) fn ofi(&self, now: UnixNanos, window: NanosDelta) -> f64 {
        assert!(window <= MAX_WINDOW, "ofi window exceeds MAX_WINDOW");
        let start = now - window;
        let flow: f64 = self
            .order_flows
            .iter()
            .rev()
            .take_while(|&&(t, _)| t > start)
            .map(|&(_, flow)| flow)
            .sum();
        flow / self.avg_l1_depth(now)
    }

    /// `(bid - ask) / (bid + ask)` of the best bid and ask quantity. NaN if no bbo has been
    /// received.
    pub(super) fn l1_imbalance(&self, now: UnixNanos) -> f64 {
        let Some(bbo) = self.bbo_at(now) else {
            return f64::NAN;
        };
        let bid = bbo.bid_qty.to_f64();
        let ask = bbo.ask_qty.to_f64();
        (bid - ask) / (bid + ask)
    }

    /// Current L1 depth over its time-weighted average over `(now - DEPTH_AVG_WINDOW, now]`.
    /// NaN if no bbo has been received or that average is zero.
    pub(super) fn l1_depth_ratio(&self, now: UnixNanos) -> f64 {
        let Some(bbo) = self.bbo_at(now) else {
            return f64::NAN;
        };
        l1_depth(bbo) / self.avg_l1_depth(now)
    }

    /// Time-weighted average L1 depth over `(now - DEPTH_AVG_WINDOW, now]`, or over the time
    /// since the first bbo if that is shorter. NaN if no bbo has been received or the average
    /// is zero, so that dividing by it gives NaN rather than infinity.
    fn avg_l1_depth(&self, now: UnixNanos) -> f64 {
        let (Some(first), Some(last)) = (self.bbos.front(), self.bbos.back()) else {
            return f64::NAN;
        };
        let cutoff = now - DEPTH_AVG_WINDOW;
        let span = secs(now - first.receive_time.max(cutoff));
        if span <= 0.0 {
            // Only the first bbo, received just now.
            return nan_if_zero(l1_depth(last));
        }
        let mut area = self.depth_area + l1_depth(last) * secs(now - last.receive_time);
        // Remove the part of each bbo's holding time that falls before `cutoff`.
        let next_times = self
            .bbos
            .iter()
            .skip(1)
            .map(|b| b.receive_time)
            .chain([now]);
        for (bbo, next_time) in self.bbos.iter().zip(next_times) {
            if bbo.receive_time >= cutoff {
                break;
            }
            area -= l1_depth(bbo) * secs(next_time.min(cutoff) - bbo.receive_time);
        }
        nan_if_zero(area / span)
    }

    /// Relative change of the mid price over `(now - window, now]`, NaN if no bbo was received
    /// at or before either end.
    ///
    /// # Panics
    /// Panics if `window` exceeds `MAX_WINDOW`.
    pub(super) fn mid_ret(&self, now: UnixNanos, window: NanosDelta) -> f64 {
        assert!(window <= MAX_WINDOW, "window exceeds MAX_WINDOW");
        let (Some(end), Some(start)) = (self.bbo_at(now), self.bbo_at(now - window)) else {
            return f64::NAN;
        };
        end.mid_price() / start.mid_price() - 1.0
    }

    /// The last bbo received at or before `time`.
    fn bbo_at(&self, time: UnixNanos) -> Option<&Bbo> {
        self.bbos.iter().rev().find(|b| b.receive_time <= time)
    }
}

/// Mean of the best bid and best ask quantity.
fn l1_depth(bbo: &Bbo) -> f64 {
    (bbo.bid_qty.to_f64() + bbo.ask_qty.to_f64()) / 2.0
}

pub(super) fn nan_if_zero(x: f64) -> f64 {
    if x > 0.0 { x } else { f64::NAN }
}

pub(super) fn secs(delta: NanosDelta) -> f64 {
    delta.as_nanos() as f64 / 1e9
}

/// Bid queue growth minus ask queue growth from `prev` to `bbo`: a better price adds the new
/// queue, a worse price removes the old one, an unchanged price adds the change in size.
fn order_flow(prev: &Bbo, bbo: &Bbo) -> f64 {
    let bid = if bbo.bid_price > prev.bid_price {
        bbo.bid_qty.to_f64()
    } else if bbo.bid_price < prev.bid_price {
        -prev.bid_qty.to_f64()
    } else {
        bbo.bid_qty.to_f64() - prev.bid_qty.to_f64()
    };
    let ask = if bbo.ask_price < prev.ask_price {
        bbo.ask_qty.to_f64()
    } else if bbo.ask_price > prev.ask_price {
        -prev.ask_qty.to_f64()
    } else {
        bbo.ask_qty.to_f64() - prev.ask_qty.to_f64()
    };
    bid - ask
}
