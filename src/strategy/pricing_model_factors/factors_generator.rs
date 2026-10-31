mod bbo_factors;
mod depth_factors;
mod trade_factors;

use std::collections::HashMap;

use infra_adapters::{Bbo, OrderBook, SymbolId, Trade};
use infra_core::{NanosDelta, UnixNanos};

use bbo_factors::BboFactors;
use depth_factors::DepthFactors;
use trade_factors::TradeFactors;

/// Half-lives in bps of the depth book obi, cached on every depth update.
const OBI_HALF_LIVES_BPS: [f64; 6] = [0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

// multi-symbol factors generator
pub(super) struct FactorsGenerator {
    symbols: HashMap<SymbolId, SymbolFactors>,
}

struct SymbolFactors {
    bbo_factors: BboFactors,
    depth_factors: DepthFactors,
    trade_factors: TradeFactors,
    /// Depth book obi per `OBI_HALF_LIVES_BPS`, as of the last depth update; None before the
    /// first one.
    obis: Option<[f64; 6]>,
}

impl SymbolFactors {
    fn new() -> Self {
        Self {
            bbo_factors: BboFactors::new(),
            depth_factors: DepthFactors::new(),
            trade_factors: TradeFactors::new(),
            obis: None,
        }
    }
}

/// Number of factors in a `FactorSample`, the models' input width.
pub(super) const NUM_FEATURES: usize = 152;

macro_rules! factor_sample {
    ($($name:ident),* $(,)?) => {
        pub(super) struct FactorSample {
            $(pub(super) $name: f64,)*
        }

        impl FactorSample {
            /// Factor names in the order of `features`, which is the samples parquet's column
            /// order the models are trained on.
            pub(super) const FEATURE_NAMES: [&str; NUM_FEATURES] = [$(stringify!($name)),*];

            pub(super) fn features(&self) -> [f64; NUM_FEATURES] {
                [$(self.$name),*]
            }
        }
    };
}

factor_sample! {
    // obi from the depth book with levels decayed by their distance from mid, by half-life in bps
    obi_0bps, // L1 only, from bbo
    obi_0_25bps,
    obi_0_5bps,
    obi_1bps,
    obi_2_5bps,
    obi_5bps,
    obi_10bps,
    // ofi from bbo, over the 1min time-weighted average L1 depth
    ofi_30us,
    ofi_150us,
    ofi_500us,
    ofi_1ms,
    ofi_3ms,
    ofi_5ms,
    ofi_10ms,
    ofi_25ms,
    ofi_50ms,
    ofi_75ms,
    ofi_100ms,
    ofi_200ms,
    ofi_300ms,
    ofi_400ms,
    ofi_500ms,
    ofi_750ms,
    ofi_1s,
    ofi_3s,
    // multi-level ofi from the depth book, levels decayed by their distance from mid as in obi,
    // over the 1min time-weighted average decayed depth; by half-life in bps, then window
    mlofi_0_5bps_30us,
    mlofi_0_5bps_150us,
    mlofi_0_5bps_500us,
    mlofi_0_5bps_1ms,
    mlofi_0_5bps_3ms,
    mlofi_0_5bps_5ms,
    mlofi_0_5bps_10ms,
    mlofi_0_5bps_25ms,
    mlofi_0_5bps_50ms,
    mlofi_0_5bps_75ms,
    mlofi_0_5bps_100ms,
    mlofi_0_5bps_200ms,
    mlofi_0_5bps_300ms,
    mlofi_0_5bps_400ms,
    mlofi_0_5bps_500ms,
    mlofi_0_5bps_750ms,
    mlofi_0_5bps_1s,
    mlofi_0_5bps_3s,
    mlofi_1bps_30us,
    mlofi_1bps_150us,
    mlofi_1bps_500us,
    mlofi_1bps_1ms,
    mlofi_1bps_3ms,
    mlofi_1bps_5ms,
    mlofi_1bps_10ms,
    mlofi_1bps_25ms,
    mlofi_1bps_50ms,
    mlofi_1bps_75ms,
    mlofi_1bps_100ms,
    mlofi_1bps_200ms,
    mlofi_1bps_300ms,
    mlofi_1bps_400ms,
    mlofi_1bps_500ms,
    mlofi_1bps_750ms,
    mlofi_1bps_1s,
    mlofi_1bps_3s,
    mlofi_2_5bps_30us,
    mlofi_2_5bps_150us,
    mlofi_2_5bps_500us,
    mlofi_2_5bps_1ms,
    mlofi_2_5bps_3ms,
    mlofi_2_5bps_5ms,
    mlofi_2_5bps_10ms,
    mlofi_2_5bps_25ms,
    mlofi_2_5bps_50ms,
    mlofi_2_5bps_75ms,
    mlofi_2_5bps_100ms,
    mlofi_2_5bps_200ms,
    mlofi_2_5bps_300ms,
    mlofi_2_5bps_400ms,
    mlofi_2_5bps_500ms,
    mlofi_2_5bps_750ms,
    mlofi_2_5bps_1s,
    mlofi_2_5bps_3s,
    mlofi_5bps_30us,
    mlofi_5bps_150us,
    mlofi_5bps_500us,
    mlofi_5bps_1ms,
    mlofi_5bps_3ms,
    mlofi_5bps_5ms,
    mlofi_5bps_10ms,
    mlofi_5bps_25ms,
    mlofi_5bps_50ms,
    mlofi_5bps_75ms,
    mlofi_5bps_100ms,
    mlofi_5bps_200ms,
    mlofi_5bps_300ms,
    mlofi_5bps_400ms,
    mlofi_5bps_500ms,
    mlofi_5bps_750ms,
    mlofi_5bps_1s,
    mlofi_5bps_3s,
    // current L1 depth over its 1min time-weighted average
    l1_depth_ratio,
    // tfi: signed trade notional over its 1h rate
    tfi_30us,
    tfi_150us,
    tfi_500us,
    tfi_1ms,
    tfi_3ms,
    tfi_5ms,
    tfi_10ms,
    tfi_25ms,
    tfi_50ms,
    tfi_75ms,
    tfi_100ms,
    tfi_200ms,
    tfi_300ms,
    tfi_400ms,
    tfi_500ms,
    tfi_750ms,
    tfi_1s,
    tfi_3s,
    // trade notional over its 1h rate, both aggressor sides
    trade_notional_ratio_30us,
    trade_notional_ratio_150us,
    trade_notional_ratio_500us,
    trade_notional_ratio_1ms,
    trade_notional_ratio_3ms,
    trade_notional_ratio_5ms,
    trade_notional_ratio_10ms,
    trade_notional_ratio_25ms,
    trade_notional_ratio_50ms,
    trade_notional_ratio_75ms,
    trade_notional_ratio_100ms,
    trade_notional_ratio_200ms,
    trade_notional_ratio_300ms,
    trade_notional_ratio_400ms,
    trade_notional_ratio_500ms,
    trade_notional_ratio_750ms,
    trade_notional_ratio_1s,
    trade_notional_ratio_3s,
    // mid return
    mid_ret_30us,
    mid_ret_150us,
    mid_ret_500us,
    mid_ret_1ms,
    mid_ret_3ms,
    mid_ret_5ms,
    mid_ret_10ms,
    mid_ret_25ms,
    mid_ret_50ms,
    mid_ret_75ms,
    mid_ret_100ms,
    mid_ret_200ms,
    mid_ret_300ms,
    mid_ret_400ms,
    mid_ret_500ms,
    mid_ret_750ms,
    mid_ret_1s,
    mid_ret_3s,
}

impl FactorsGenerator {
    pub(super) fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }

    fn symbol(&mut self, symbol_id: SymbolId) -> &mut SymbolFactors {
        self.symbols
            .entry(symbol_id)
            .or_insert_with(SymbolFactors::new)
    }
}

// update its internal state
impl FactorsGenerator {
    pub(super) fn on_bbo(&mut self, symbol_id: SymbolId, bbo: Bbo) {
        self.symbol(symbol_id).bbo_factors.on_bbo(bbo);
    }

    pub(super) fn on_trade(&mut self, symbol_id: SymbolId, trade: Trade) {
        self.symbol(symbol_id).trade_factors.on_trade(trade);
    }

    pub(super) fn on_depth(&mut self, symbol_id: SymbolId, order_book: &OrderBook) {
        let s = self.symbol(symbol_id);
        s.depth_factors
            .on_depth(order_book.receive_time(), order_book);
        s.obis = Some(OBI_HALF_LIVES_BPS.map(|h| s.depth_factors.obi(order_book, h)));
    }
}

impl FactorsGenerator {
    /// # Panics
    /// Panics if the symbol is unseen or has had no depth update yet.
    pub(super) fn get_sample(&self, symbol_id: SymbolId, now: UnixNanos) -> FactorSample {
        let s = self
            .symbols
            .get(&symbol_id)
            .expect("get_sample called for an unseen symbol");
        let [
            obi_0_25bps,
            obi_0_5bps,
            obi_1bps,
            obi_2_5bps,
            obi_5bps,
            obi_10bps,
        ] = s
            .obis
            .expect("get_sample called before the symbol's first depth update");
        FactorSample {
            obi_0bps: s.bbo_factors.l1_imbalance(now),
            obi_0_25bps,
            obi_0_5bps,
            obi_1bps,
            obi_2_5bps,
            obi_5bps,
            obi_10bps,
            ofi_30us: s.bbo_factors.ofi(now, NanosDelta::from_micros(30)),
            ofi_150us: s.bbo_factors.ofi(now, NanosDelta::from_micros(150)),
            ofi_500us: s.bbo_factors.ofi(now, NanosDelta::from_micros(500)),
            ofi_1ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(1)),
            ofi_3ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(3)),
            ofi_5ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(5)),
            ofi_10ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(10)),
            ofi_25ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(25)),
            ofi_50ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(50)),
            ofi_75ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(75)),
            ofi_100ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(100)),
            ofi_200ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(200)),
            ofi_300ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(300)),
            ofi_400ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(400)),
            ofi_500ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(500)),
            ofi_750ms: s.bbo_factors.ofi(now, NanosDelta::from_millis(750)),
            ofi_1s: s.bbo_factors.ofi(now, NanosDelta::from_secs(1)),
            ofi_3s: s.bbo_factors.ofi(now, NanosDelta::from_secs(3)),
            mlofi_0_5bps_30us: s.depth_factors.mlofi(now, NanosDelta::from_micros(30), 0.5),
            mlofi_0_5bps_150us: s
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(150), 0.5),
            mlofi_0_5bps_500us: s
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(500), 0.5),
            mlofi_0_5bps_1ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(1), 0.5),
            mlofi_0_5bps_3ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(3), 0.5),
            mlofi_0_5bps_5ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(5), 0.5),
            mlofi_0_5bps_10ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(10), 0.5),
            mlofi_0_5bps_25ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(25), 0.5),
            mlofi_0_5bps_50ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(50), 0.5),
            mlofi_0_5bps_75ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(75), 0.5),
            mlofi_0_5bps_100ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(100), 0.5),
            mlofi_0_5bps_200ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(200), 0.5),
            mlofi_0_5bps_300ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(300), 0.5),
            mlofi_0_5bps_400ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(400), 0.5),
            mlofi_0_5bps_500ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(500), 0.5),
            mlofi_0_5bps_750ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(750), 0.5),
            mlofi_0_5bps_1s: s.depth_factors.mlofi(now, NanosDelta::from_secs(1), 0.5),
            mlofi_0_5bps_3s: s.depth_factors.mlofi(now, NanosDelta::from_secs(3), 0.5),
            mlofi_1bps_30us: s.depth_factors.mlofi(now, NanosDelta::from_micros(30), 1.0),
            mlofi_1bps_150us: s
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(150), 1.0),
            mlofi_1bps_500us: s
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(500), 1.0),
            mlofi_1bps_1ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(1), 1.0),
            mlofi_1bps_3ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(3), 1.0),
            mlofi_1bps_5ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(5), 1.0),
            mlofi_1bps_10ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(10), 1.0),
            mlofi_1bps_25ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(25), 1.0),
            mlofi_1bps_50ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(50), 1.0),
            mlofi_1bps_75ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(75), 1.0),
            mlofi_1bps_100ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(100), 1.0),
            mlofi_1bps_200ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(200), 1.0),
            mlofi_1bps_300ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(300), 1.0),
            mlofi_1bps_400ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(400), 1.0),
            mlofi_1bps_500ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(500), 1.0),
            mlofi_1bps_750ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(750), 1.0),
            mlofi_1bps_1s: s.depth_factors.mlofi(now, NanosDelta::from_secs(1), 1.0),
            mlofi_1bps_3s: s.depth_factors.mlofi(now, NanosDelta::from_secs(3), 1.0),
            mlofi_2_5bps_30us: s.depth_factors.mlofi(now, NanosDelta::from_micros(30), 2.5),
            mlofi_2_5bps_150us: s
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(150), 2.5),
            mlofi_2_5bps_500us: s
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(500), 2.5),
            mlofi_2_5bps_1ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(1), 2.5),
            mlofi_2_5bps_3ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(3), 2.5),
            mlofi_2_5bps_5ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(5), 2.5),
            mlofi_2_5bps_10ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(10), 2.5),
            mlofi_2_5bps_25ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(25), 2.5),
            mlofi_2_5bps_50ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(50), 2.5),
            mlofi_2_5bps_75ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(75), 2.5),
            mlofi_2_5bps_100ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(100), 2.5),
            mlofi_2_5bps_200ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(200), 2.5),
            mlofi_2_5bps_300ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(300), 2.5),
            mlofi_2_5bps_400ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(400), 2.5),
            mlofi_2_5bps_500ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(500), 2.5),
            mlofi_2_5bps_750ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(750), 2.5),
            mlofi_2_5bps_1s: s.depth_factors.mlofi(now, NanosDelta::from_secs(1), 2.5),
            mlofi_2_5bps_3s: s.depth_factors.mlofi(now, NanosDelta::from_secs(3), 2.5),
            mlofi_5bps_30us: s.depth_factors.mlofi(now, NanosDelta::from_micros(30), 5.0),
            mlofi_5bps_150us: s
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(150), 5.0),
            mlofi_5bps_500us: s
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(500), 5.0),
            mlofi_5bps_1ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(1), 5.0),
            mlofi_5bps_3ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(3), 5.0),
            mlofi_5bps_5ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(5), 5.0),
            mlofi_5bps_10ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(10), 5.0),
            mlofi_5bps_25ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(25), 5.0),
            mlofi_5bps_50ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(50), 5.0),
            mlofi_5bps_75ms: s.depth_factors.mlofi(now, NanosDelta::from_millis(75), 5.0),
            mlofi_5bps_100ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(100), 5.0),
            mlofi_5bps_200ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(200), 5.0),
            mlofi_5bps_300ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(300), 5.0),
            mlofi_5bps_400ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(400), 5.0),
            mlofi_5bps_500ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(500), 5.0),
            mlofi_5bps_750ms: s
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(750), 5.0),
            mlofi_5bps_1s: s.depth_factors.mlofi(now, NanosDelta::from_secs(1), 5.0),
            mlofi_5bps_3s: s.depth_factors.mlofi(now, NanosDelta::from_secs(3), 5.0),
            l1_depth_ratio: s.bbo_factors.l1_depth_ratio(now),
            tfi_30us: s.trade_factors.tfi(now, NanosDelta::from_micros(30)),
            tfi_150us: s.trade_factors.tfi(now, NanosDelta::from_micros(150)),
            tfi_500us: s.trade_factors.tfi(now, NanosDelta::from_micros(500)),
            tfi_1ms: s.trade_factors.tfi(now, NanosDelta::from_millis(1)),
            tfi_3ms: s.trade_factors.tfi(now, NanosDelta::from_millis(3)),
            tfi_5ms: s.trade_factors.tfi(now, NanosDelta::from_millis(5)),
            tfi_10ms: s.trade_factors.tfi(now, NanosDelta::from_millis(10)),
            tfi_25ms: s.trade_factors.tfi(now, NanosDelta::from_millis(25)),
            tfi_50ms: s.trade_factors.tfi(now, NanosDelta::from_millis(50)),
            tfi_75ms: s.trade_factors.tfi(now, NanosDelta::from_millis(75)),
            tfi_100ms: s.trade_factors.tfi(now, NanosDelta::from_millis(100)),
            tfi_200ms: s.trade_factors.tfi(now, NanosDelta::from_millis(200)),
            tfi_300ms: s.trade_factors.tfi(now, NanosDelta::from_millis(300)),
            tfi_400ms: s.trade_factors.tfi(now, NanosDelta::from_millis(400)),
            tfi_500ms: s.trade_factors.tfi(now, NanosDelta::from_millis(500)),
            tfi_750ms: s.trade_factors.tfi(now, NanosDelta::from_millis(750)),
            tfi_1s: s.trade_factors.tfi(now, NanosDelta::from_secs(1)),
            tfi_3s: s.trade_factors.tfi(now, NanosDelta::from_secs(3)),
            trade_notional_ratio_30us: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_micros(30)),
            trade_notional_ratio_150us: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_micros(150)),
            trade_notional_ratio_500us: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_micros(500)),
            trade_notional_ratio_1ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(1)),
            trade_notional_ratio_3ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(3)),
            trade_notional_ratio_5ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(5)),
            trade_notional_ratio_10ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(10)),
            trade_notional_ratio_25ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(25)),
            trade_notional_ratio_50ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(50)),
            trade_notional_ratio_75ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(75)),
            trade_notional_ratio_100ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(100)),
            trade_notional_ratio_200ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(200)),
            trade_notional_ratio_300ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(300)),
            trade_notional_ratio_400ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(400)),
            trade_notional_ratio_500ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(500)),
            trade_notional_ratio_750ms: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(750)),
            trade_notional_ratio_1s: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_secs(1)),
            trade_notional_ratio_3s: s
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_secs(3)),
            mid_ret_30us: s.bbo_factors.mid_ret(now, NanosDelta::from_micros(30)),
            mid_ret_150us: s.bbo_factors.mid_ret(now, NanosDelta::from_micros(150)),
            mid_ret_500us: s.bbo_factors.mid_ret(now, NanosDelta::from_micros(500)),
            mid_ret_1ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(1)),
            mid_ret_3ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(3)),
            mid_ret_5ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(5)),
            mid_ret_10ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(10)),
            mid_ret_25ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(25)),
            mid_ret_50ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(50)),
            mid_ret_75ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(75)),
            mid_ret_100ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(100)),
            mid_ret_200ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(200)),
            mid_ret_300ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(300)),
            mid_ret_400ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(400)),
            mid_ret_500ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(500)),
            mid_ret_750ms: s.bbo_factors.mid_ret(now, NanosDelta::from_millis(750)),
            mid_ret_1s: s.bbo_factors.mid_ret(now, NanosDelta::from_secs(1)),
            mid_ret_3s: s.bbo_factors.mid_ret(now, NanosDelta::from_secs(3)),
        }
    }
}
