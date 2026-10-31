use std::collections::VecDeque;

use infra_adapters::{Level, OrderBook};
use infra_core::{NanosDelta, PriceI64, UnixNanos};

use super::bbo_factors::{nan_if_zero, secs};

/// Binance's deepest depth snapshot; the local book beyond it is only partially known.
const BOOK_LEVELS: usize = 1000;
/// Levels weighted below `0.5^MAX_HALF_LIVES` (about 1e-6) are not summed.
const MAX_HALF_LIVES: f64 = 20.0;
/// Half-lives, in bps, that multi-level order flow imbalance is tracked for.
const MLOFI_HALF_LIVES_BPS: [f64; 4] = [0.5, 1.0, 2.5, 5.0];
/// Distance from the mid in bps beyond which a level weighs nothing at any of
/// `MLOFI_HALF_LIVES_BPS`, the longest of which is last.
const MLOFI_MAX_BPS: f64 = MAX_HALF_LIVES * MLOFI_HALF_LIVES_BPS[3];
/// History older than this is dropped, so no mlofi window may exceed it.
const MAX_WINDOW: NanosDelta = NanosDelta::from_secs(60);
/// Multi-level order flow imbalance is normalized by the time-weighted average decayed depth
/// over this long.
const DEPTH_AVG_WINDOW: NanosDelta = MAX_WINDOW;

type PerHalfLife = [f64; MLOFI_HALF_LIVES_BPS.len()];

/// The levels of a depth book within `MLOFI_MAX_BPS` of its mid, best first on each side.
struct Book {
    bids: Vec<Level>,
    asks: Vec<Level>,
}

pub(super) struct DepthFactors {
    /// The depth book as of the last update, `None` before the first one and after it became
    /// unavailable, so no flow is measured across such a gap.
    prev_book: Option<Book>,
    /// Receive time and decayed order flow, per `MLOFI_HALF_LIVES_BPS`, of each depth book
    /// update within `MAX_WINDOW`, oldest first.
    order_flows: VecDeque<(UnixNanos, PerHalfLife)>,
    /// Receive time and decayed depth, per `MLOFI_HALF_LIVES_BPS`, of each depth book update
    /// within `MAX_WINDOW`, oldest first, plus the last one before it so the depth as of any
    /// window start is known.
    depths: VecDeque<(UnixNanos, PerHalfLife)>,
    /// Time integral of each decayed depth, in quantity-seconds, from the first to the last
    /// update in `depths`.
    depth_areas: PerHalfLife,
}

impl DepthFactors {
    pub(super) fn new() -> Self {
        Self {
            prev_book: None,
            order_flows: VecDeque::new(),
            depths: VecDeque::new(),
            depth_areas: [0.0; MLOFI_HALF_LIVES_BPS.len()],
        }
    }

    /// Called after every depth book update.
    pub(super) fn on_depth(&mut self, now: UnixNanos, book: &OrderBook) {
        let Some((book, mid)) = near_levels(book) else {
            self.prev_book = None;
            return;
        };
        // While the mid is unchanged every level keeps its weight, so the decayed depths only
        // move by the weighted quantity changes; otherwise they are recomputed.
        let mut depths = None;
        if let Some(prev) = &self.prev_book
            && let Some(prev_mid) = mid_price(prev)
        {
            let (flows, depth_changes) = decayed_order_flows(prev, &book, prev_mid);
            self.order_flows.push_back((now, flows));
            if prev_mid == mid
                && let Some(&(_, prev_depths)) = self.depths.back()
            {
                let mut d = prev_depths;
                for (depth, change) in d.iter_mut().zip(depth_changes) {
                    *depth += change;
                }
                depths = Some(d);
            }
        }
        let depths = depths.unwrap_or_else(|| decayed_depths(&book, mid));
        if let Some(&(t, prev_depths)) = self.depths.back() {
            for (area, depth) in self.depth_areas.iter_mut().zip(prev_depths) {
                *area += depth * secs(now - t);
            }
        }
        self.depths.push_back((now, depths));
        self.prev_book = Some(book);

        let cutoff = now - MAX_WINDOW;
        while self.order_flows.front().is_some_and(|&(t, _)| t <= cutoff) {
            self.order_flows.pop_front();
        }
        while let Some(&(next_time, _)) = self.depths.get(1).filter(|&&(t, _)| t <= cutoff) {
            let (t, first_depths) = self.depths.pop_front().expect("depths has a second entry");
            for (area, depth) in self.depth_areas.iter_mut().zip(first_depths) {
                *area -= depth * secs(next_time - t);
            }
        }
    }

    /// Called when the depth book becomes unavailable, so no flow is measured across the gap.
    pub(super) fn reset(&mut self) {
        self.prev_book = None;
    }
}

impl DepthFactors {
    /// `(bid - ask) / (bid + ask)` of the book quantity, each level weighted by
    /// `0.5^(distance / half_life_bps)`, where `distance` is its distance from the mid price
    /// in bps. Measuring distance relative to price keeps one half-life comparable across
    /// symbols whose ticks differ by orders of magnitude in bps. Levels beyond
    /// `MAX_HALF_LIVES` half-lives are skipped, as is everything past `BOOK_LEVELS`.
    /// `half_life_bps` must be positive. NaN if either side is empty.
    pub(super) fn obi(&self, book: &OrderBook, half_life_bps: f64) -> f64 {
        let (Some(best_bid), Some(best_ask)) = (book.bids().next(), book.asks().next()) else {
            return f64::NAN;
        };
        let mid = (best_bid.price.to_f64() + best_ask.price.to_f64()) / 2.0;
        let bid = decayed_qty(book.bids().take(BOOK_LEVELS), mid, half_life_bps);
        let ask = decayed_qty(book.asks().take(BOOK_LEVELS), mid, half_life_bps);
        (bid - ask) / (bid + ask)
    }

    /// Multi-level order flow imbalance: the summed decayed order flow (see
    /// `decayed_order_flows`) of the depth book updates received in `(now - window, now]`,
    /// over the time-weighted average decayed depth, `(bid + ask) / 2` weighted as in `obi`,
    /// of `(now - DEPTH_AVG_WINDOW, now]`. NaN if that average is unknown or zero.
    ///
    /// # Panics
    /// Panics if `window` exceeds `MAX_WINDOW` or `half_life_bps` is not in
    /// `MLOFI_HALF_LIVES_BPS`.
    pub(super) fn mlofi(&self, now: UnixNanos, window: NanosDelta, half_life_bps: f64) -> f64 {
        assert!(window <= MAX_WINDOW, "mlofi window exceeds MAX_WINDOW");
        let i = MLOFI_HALF_LIVES_BPS
            .iter()
            .position(|&h| h == half_life_bps)
            .expect("half_life_bps is not in MLOFI_HALF_LIVES_BPS");
        let start = now - window;
        let flow: f64 = self
            .order_flows
            .iter()
            .rev()
            .take_while(|&&(t, _)| t > start)
            .map(|(_, flows)| flows[i])
            .sum();
        flow / self.avg_depth(now, i)
    }

    /// Time-weighted average of `depths[..][i]` over `(now - DEPTH_AVG_WINDOW, now]`, or over
    /// the time since the first update if that is shorter. NaN if there has been no update or
    /// the average is zero.
    fn avg_depth(&self, now: UnixNanos, i: usize) -> f64 {
        let (Some(&(first_time, _)), Some(&(last_time, last))) =
            (self.depths.front(), self.depths.back())
        else {
            return f64::NAN;
        };
        let cutoff = now - DEPTH_AVG_WINDOW;
        let span = secs(now - first_time.max(cutoff));
        if span <= 0.0 {
            // Only the first update, received just now.
            return nan_if_zero(last[i]);
        }
        let mut area = self.depth_areas[i] + last[i] * secs(now - last_time);
        // Remove the part of each update's holding time that falls before `cutoff`.
        let next_times = self.depths.iter().skip(1).map(|&(t, _)| t).chain([now]);
        for (&(t, depth), next_time) in self.depths.iter().zip(next_times) {
            if t >= cutoff {
                break;
            }
            area -= depth[i] * secs(next_time.min(cutoff) - t);
        }
        nan_if_zero(area / span)
    }
}

/// Bid quantity growth minus ask quantity growth from `prev` to `book`, per
/// `MLOFI_HALF_LIVES_BPS`, compared price by price, each price weighted by
/// `0.5^(distance / half_life_bps)` of its distance from `prev_mid` in bps. Prices beyond
/// `MAX_HALF_LIVES` half-lives are skipped. At the best prices this is the Cont, Kukanov &
/// Stoikov order flow: a better price adds the new queue, a worse price removes the old one,
/// an unchanged price adds the change in size.
///
/// Also returns the change in `decayed_depths` weighted the same way, which is exact when the
/// mid price is unchanged.
fn decayed_order_flows(prev: &Book, book: &Book, prev_mid: f64) -> (PerHalfLife, PerHalfLife) {
    // Prices beyond `MLOFI_MAX_BPS` weigh nothing at any half-life, so the books are only
    // compared up to it, and only once for all half-lives.
    let near = |levels: &[Level]| -> usize {
        levels.partition_point(|l| half_lives(l.price, prev_mid, 1.0) <= MLOFI_MAX_BPS)
    };
    // Distance from `prev_mid` in bps, quantity change, and +1 for a bid or -1 for an ask.
    let mut changes: Vec<(f64, f64, f64)> = Vec::new();
    let sides: [(&[Level], &[Level], fn(PriceI64, PriceI64) -> bool, f64); 2] = [
        (&prev.bids, &book.bids, |a, b| a > b, 1.0),
        (&prev.asks, &book.asks, |a, b| a < b, -1.0),
    ];
    for (prev, cur, is_better, sign) in sides {
        changes.extend(
            qty_changes(&prev[..near(prev)], &cur[..near(cur)], is_better)
                .map(|(price, change)| (half_lives(price, prev_mid, 1.0), change, sign)),
        );
    }
    let mut flows = [0.0; MLOFI_HALF_LIVES_BPS.len()];
    let mut depth_changes = [0.0; MLOFI_HALF_LIVES_BPS.len()];
    for (bps, change, sign) in changes {
        for (k, half_life_bps) in MLOFI_HALF_LIVES_BPS.into_iter().enumerate() {
            let h = bps / half_life_bps;
            if h <= MAX_HALF_LIVES {
                let weighted = change * (-h).exp2();
                flows[k] += sign * weighted;
                depth_changes[k] += weighted / 2.0;
            }
        }
    }
    (flows, depth_changes)
}

/// The nonzero quantity changes from `prev` to `cur`, one side of the book each, by price,
/// best first. `is_better(a, b)` tells whether price `a` is nearer the top than `b`.
fn qty_changes<'a>(
    prev: &'a [Level],
    cur: &'a [Level],
    is_better: fn(PriceI64, PriceI64) -> bool,
) -> impl Iterator<Item = (PriceI64, f64)> + 'a {
    let (mut i, mut j) = (0, 0);
    std::iter::from_fn(move || {
        loop {
            let change = match (prev.get(i), cur.get(j)) {
                (None, None) => return None,
                (Some(p), Some(c)) if p.price == c.price => {
                    i += 1;
                    j += 1;
                    (c.price, c.qty.to_f64() - p.qty.to_f64())
                }
                (Some(p), Some(c)) if is_better(c.price, p.price) => {
                    j += 1;
                    (c.price, c.qty.to_f64())
                }
                (Some(p), _) => {
                    i += 1;
                    (p.price, -p.qty.to_f64())
                }
                (None, Some(c)) => {
                    j += 1;
                    (c.price, c.qty.to_f64())
                }
            };
            if change.1 != 0.0 {
                return Some(change);
            }
        }
    })
}

/// `(bid + ask) / 2` of the book quantity weighted as in `decayed_qty`, per
/// `MLOFI_HALF_LIVES_BPS`. Each level's distance is computed once for all half-lives.
fn decayed_depths(book: &Book, mid: f64) -> PerHalfLife {
    let mut depths = [0.0; MLOFI_HALF_LIVES_BPS.len()];
    for levels in [&book.bids, &book.asks] {
        for level in levels {
            let bps = half_lives(level.price, mid, 1.0);
            if bps > MLOFI_MAX_BPS {
                break;
            }
            for (depth, half_life_bps) in depths.iter_mut().zip(MLOFI_HALF_LIVES_BPS) {
                let h = bps / half_life_bps;
                if h <= MAX_HALF_LIVES {
                    *depth += level.qty.to_f64() * (-h).exp2() / 2.0;
                }
            }
        }
    }
    depths
}

/// Quantity of `levels`, best first, each weighted by `0.5^(distance / half_life_bps)` of its
/// distance from `mid` in bps. Levels beyond `MAX_HALF_LIVES` half-lives are skipped.
fn decayed_qty(levels: impl Iterator<Item = Level>, mid: f64, half_life_bps: f64) -> f64 {
    levels
        .map(|l| (l.qty.to_f64(), half_lives(l.price, mid, half_life_bps)))
        .take_while(|&(_, h)| h <= MAX_HALF_LIVES)
        .map(|(qty, h)| qty * (-h).exp2())
        .sum()
}

/// Distance of `price` from `mid` in half-lives of `half_life_bps`.
fn half_lives(price: PriceI64, mid: f64, half_life_bps: f64) -> f64 {
    (price.to_f64() - mid).abs() / mid * 1e4 / half_life_bps
}

/// The levels of `book` that mlofi weighs, at most `BOOK_LEVELS` per side, and its mid.
/// `None` if either side is empty.
fn near_levels(book: &OrderBook) -> Option<(Book, f64)> {
    let (best_bid, best_ask) = (book.bids().next()?, book.asks().next()?);
    let mid = (best_bid.price.to_f64() + best_ask.price.to_f64()) / 2.0;
    let near = |l: &Level| half_lives(l.price, mid, 1.0) <= MLOFI_MAX_BPS;
    let book = Book {
        bids: book.bids().take_while(near).take(BOOK_LEVELS).collect(),
        asks: book.asks().take_while(near).take(BOOK_LEVELS).collect(),
    };
    Some((book, mid))
}

/// Mid of the best bid and ask, `None` if either side is empty.
fn mid_price(book: &Book) -> Option<f64> {
    let (best_bid, best_ask) = (book.bids.first()?, book.asks.first()?);
    Some((best_bid.price.to_f64() + best_ask.price.to_f64()) / 2.0)
}

#[cfg(test)]
mod tests {
    use infra_core::QuantityI64;

    use super::*;

    fn levels(levels: &[(f64, f64)]) -> Vec<Level> {
        levels
            .iter()
            .map(|&(price, qty)| Level {
                price: PriceI64::from_f64(price),
                qty: QuantityI64::from_f64(qty),
            })
            .collect()
    }

    fn book(bids: &[(f64, f64)], asks: &[(f64, f64)]) -> Book {
        Book {
            bids: levels(bids),
            asks: levels(asks),
        }
    }

    fn changes(prev: &[(f64, f64)], cur: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let (prev, cur) = (levels(prev), levels(cur));
        qty_changes(&prev, &cur, |a, b| a > b)
            .map(|(price, change)| (price.to_f64(), change))
            .collect()
    }

    #[test]
    fn qty_changes_by_price() {
        // Same prices: only the size change shows.
        assert_eq!(
            changes(&[(10.0, 5.0), (9.0, 3.0)], &[(10.0, 7.0), (9.0, 3.0)]),
            [(10.0, 2.0)]
        );
        // A better price adds its queue.
        assert_eq!(
            changes(&[(10.0, 5.0)], &[(11.0, 4.0), (10.0, 5.0)]),
            [(11.0, 4.0)]
        );
        // A level that disappears removes its queue, the one behind it is unchanged.
        assert_eq!(
            changes(&[(10.0, 5.0), (9.0, 3.0)], &[(9.0, 3.0)]),
            [(10.0, -5.0)]
        );
        // Interleaved prices on both lists.
        assert_eq!(
            changes(
                &[(10.0, 1.0), (8.0, 2.0)],
                &[(9.0, 3.0), (8.0, 1.0), (7.0, 4.0)]
            ),
            [(10.0, -1.0), (9.0, 3.0), (8.0, -1.0), (7.0, 4.0)]
        );
    }

    #[test]
    fn weights_each_change_by_distance_from_prev_mid() {
        // Mid 100.015. The bid improves to 100.01 (+4, 0.5bps out), the bid at 99.98 grows
        // (+8, 3.5bps out) and the ask at 100.03 grows (+2, 1.5bps out).
        let prev = book(&[(100.0, 5.0), (99.98, 1.0)], &[(100.03, 6.0)]);
        let cur = book(
            &[(100.01, 4.0), (100.0, 5.0), (99.98, 9.0)],
            &[(100.03, 8.0)],
        );
        let prev_mid = mid_price(&prev).unwrap();
        let w = |p: f64| 0.5f64.powf((p - prev_mid).abs() / prev_mid * 1e4);
        let expected = 4.0 * w(100.01) + 8.0 * w(99.98) - 2.0 * w(100.03);
        let flow = decayed_order_flows(&prev, &cur, prev_mid).0[1];
        assert!((flow - expected).abs() < 1e-9, "{flow} != {expected}");
        assert!(
            expected > 2.0,
            "weights should be far from zero: {expected}"
        );
    }
}
