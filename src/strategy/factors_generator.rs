mod bbo_factors;
mod depth_factors;
mod mix_factors;
mod sample_writer;
mod trade_factors;

use infra_adapters::{
    BboEvent, BinanceFuturesBboApi, BinanceFuturesBboSimAheadApi, BinanceFuturesDepthApi,
    DepthEvent, ExecutionEvent, SymbolId, TimerApi, TimerEvent, TimerId, TradeEvent,
};
use infra_core::NanosDelta;

use bbo_factors::BboFactors;
use depth_factors::DepthFactors;
use infra_kernel::{Ctx, Feeds, Strategy};
use mix_factors::MixFactors;
use sample_writer::SampleWriter;
use trade_factors::TradeFactors;

const SAMPLE_SYMBOL_ID: SymbolId = 1;
const WARMUP_TIMER_ID: TimerId = 1;
const WARMUP_DURATION: NanosDelta = NanosDelta::from_secs(60 * 60);
/// One sample is taken every this many bbo, trade and depth events, counted together, so the
/// event mix is kept while the rows shrink.
const SAMPLE_STRIDE: u64 = 100;

pub(crate) struct FactorsGeneratorStrategy {
    bbo_factors: BboFactors,
    depth_factors: DepthFactors,
    trade_factors: TradeFactors,
    mix_factors: MixFactors,
    sample_writer: SampleWriter,
    has_warmup: bool,
    events_after_warmup: u64,
}

impl FactorsGeneratorStrategy {
    pub(crate) fn new() -> Self {
        Self {
            bbo_factors: BboFactors::new(),
            depth_factors: DepthFactors::new(),
            trade_factors: TradeFactors::new(),
            mix_factors: MixFactors::new(),
            sample_writer: SampleWriter::new(),
            has_warmup: false,
            events_after_warmup: 0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Sample {
    time: String,
    time_ns: i64,
    // the callback that triggered the sample: bbo, trade or depth
    event: String,
    fmid_0: f64,
    fmid_350us: f64,
    fmid_700us: f64,
    fmid_1ms: f64,
    fmid_2ms: f64,
    fmid_5ms: f64,
    fmid_25ms: f64,
    fmid_50ms: f64,
    fmid_100ms: f64,
    fmid_200ms: f64,
    fmid_300ms: f64,
    fmid_500ms: f64,
    fmid_750ms: f64,
    fmid_1s: f64,
    fmid_2s: f64,
    fmid_5s: f64,
    fmid_10s: f64,
    fmid_30s: f64,
    fmid_60s: f64,
    // obi from the depth book with levels decayed by their distance from mid, by half-life in bps
    obi_0bps: f64, // L1 only, from bbo
    obi_0_25bps: f64,
    obi_0_5bps: f64,
    obi_1bps: f64,
    obi_2_5bps: f64,
    obi_5bps: f64,
    obi_10bps: f64,
    // ofi from bbo, over the 1min time-weighted average L1 depth
    ofi_30us: f64,
    ofi_150us: f64,
    ofi_500us: f64,
    ofi_1ms: f64,
    ofi_3ms: f64,
    ofi_5ms: f64,
    ofi_10ms: f64,
    ofi_25ms: f64,
    ofi_50ms: f64,
    ofi_75ms: f64,
    ofi_100ms: f64,
    ofi_200ms: f64,
    ofi_300ms: f64,
    ofi_400ms: f64,
    ofi_500ms: f64,
    ofi_750ms: f64,
    ofi_1s: f64,
    ofi_3s: f64,
    // multi-level ofi from the depth book, levels decayed by their distance from mid as in obi,
    // over the 1min time-weighted average decayed depth; by half-life in bps, then window
    mlofi_0_5bps_30us: f64,
    mlofi_0_5bps_150us: f64,
    mlofi_0_5bps_500us: f64,
    mlofi_0_5bps_1ms: f64,
    mlofi_0_5bps_3ms: f64,
    mlofi_0_5bps_5ms: f64,
    mlofi_0_5bps_10ms: f64,
    mlofi_0_5bps_25ms: f64,
    mlofi_0_5bps_50ms: f64,
    mlofi_0_5bps_75ms: f64,
    mlofi_0_5bps_100ms: f64,
    mlofi_0_5bps_200ms: f64,
    mlofi_0_5bps_300ms: f64,
    mlofi_0_5bps_400ms: f64,
    mlofi_0_5bps_500ms: f64,
    mlofi_0_5bps_750ms: f64,
    mlofi_0_5bps_1s: f64,
    mlofi_0_5bps_3s: f64,
    mlofi_1bps_30us: f64,
    mlofi_1bps_150us: f64,
    mlofi_1bps_500us: f64,
    mlofi_1bps_1ms: f64,
    mlofi_1bps_3ms: f64,
    mlofi_1bps_5ms: f64,
    mlofi_1bps_10ms: f64,
    mlofi_1bps_25ms: f64,
    mlofi_1bps_50ms: f64,
    mlofi_1bps_75ms: f64,
    mlofi_1bps_100ms: f64,
    mlofi_1bps_200ms: f64,
    mlofi_1bps_300ms: f64,
    mlofi_1bps_400ms: f64,
    mlofi_1bps_500ms: f64,
    mlofi_1bps_750ms: f64,
    mlofi_1bps_1s: f64,
    mlofi_1bps_3s: f64,
    mlofi_2_5bps_30us: f64,
    mlofi_2_5bps_150us: f64,
    mlofi_2_5bps_500us: f64,
    mlofi_2_5bps_1ms: f64,
    mlofi_2_5bps_3ms: f64,
    mlofi_2_5bps_5ms: f64,
    mlofi_2_5bps_10ms: f64,
    mlofi_2_5bps_25ms: f64,
    mlofi_2_5bps_50ms: f64,
    mlofi_2_5bps_75ms: f64,
    mlofi_2_5bps_100ms: f64,
    mlofi_2_5bps_200ms: f64,
    mlofi_2_5bps_300ms: f64,
    mlofi_2_5bps_400ms: f64,
    mlofi_2_5bps_500ms: f64,
    mlofi_2_5bps_750ms: f64,
    mlofi_2_5bps_1s: f64,
    mlofi_2_5bps_3s: f64,
    mlofi_5bps_30us: f64,
    mlofi_5bps_150us: f64,
    mlofi_5bps_500us: f64,
    mlofi_5bps_1ms: f64,
    mlofi_5bps_3ms: f64,
    mlofi_5bps_5ms: f64,
    mlofi_5bps_10ms: f64,
    mlofi_5bps_25ms: f64,
    mlofi_5bps_50ms: f64,
    mlofi_5bps_75ms: f64,
    mlofi_5bps_100ms: f64,
    mlofi_5bps_200ms: f64,
    mlofi_5bps_300ms: f64,
    mlofi_5bps_400ms: f64,
    mlofi_5bps_500ms: f64,
    mlofi_5bps_750ms: f64,
    mlofi_5bps_1s: f64,
    mlofi_5bps_3s: f64,
    // current L1 depth over its 1min time-weighted average
    l1_depth_ratio: f64,
    // tfi: signed trade notional over its 1h rate
    tfi_30us: f64,
    tfi_150us: f64,
    tfi_500us: f64,
    tfi_1ms: f64,
    tfi_3ms: f64,
    tfi_5ms: f64,
    tfi_10ms: f64,
    tfi_25ms: f64,
    tfi_50ms: f64,
    tfi_75ms: f64,
    tfi_100ms: f64,
    tfi_200ms: f64,
    tfi_300ms: f64,
    tfi_400ms: f64,
    tfi_500ms: f64,
    tfi_750ms: f64,
    tfi_1s: f64,
    tfi_3s: f64,
    // trade notional over its 1h rate, both aggressor sides
    trade_notional_ratio_30us: f64,
    trade_notional_ratio_150us: f64,
    trade_notional_ratio_500us: f64,
    trade_notional_ratio_1ms: f64,
    trade_notional_ratio_3ms: f64,
    trade_notional_ratio_5ms: f64,
    trade_notional_ratio_10ms: f64,
    trade_notional_ratio_25ms: f64,
    trade_notional_ratio_50ms: f64,
    trade_notional_ratio_75ms: f64,
    trade_notional_ratio_100ms: f64,
    trade_notional_ratio_200ms: f64,
    trade_notional_ratio_300ms: f64,
    trade_notional_ratio_400ms: f64,
    trade_notional_ratio_500ms: f64,
    trade_notional_ratio_750ms: f64,
    trade_notional_ratio_1s: f64,
    trade_notional_ratio_3s: f64,
    // mid return
    mid_ret_30us: f64,
    mid_ret_150us: f64,
    mid_ret_500us: f64,
    mid_ret_1ms: f64,
    mid_ret_3ms: f64,
    mid_ret_5ms: f64,
    mid_ret_10ms: f64,
    mid_ret_25ms: f64,
    mid_ret_50ms: f64,
    mid_ret_75ms: f64,
    mid_ret_100ms: f64,
    mid_ret_200ms: f64,
    mid_ret_300ms: f64,
    mid_ret_400ms: f64,
    mid_ret_500ms: f64,
    mid_ret_750ms: f64,
    mid_ret_1s: f64,
    mid_ret_3s: f64,
}

impl<F: Feeds> Strategy<F> for FactorsGeneratorStrategy {
    fn on_start(&mut self, ctx: &mut Ctx<F>) {
        ctx.timer.after(WARMUP_TIMER_ID, WARMUP_DURATION);
    }

    fn on_execution(&mut self, event: ExecutionEvent, _ctx: &mut Ctx<F>) {
        match event {
            ExecutionEvent::StateChanged(_state) => {}
            ExecutionEvent::OrderSent {
                id: _id,
                symbol_id: _symbol_id,
            } => {}
            ExecutionEvent::CancelSent {
                id: _id,
                symbol_id: _symbol_id,
            } => {}
            ExecutionEvent::OrderAccepted {
                id: _id,
                symbol_id: _symbol_id,
                venue_id: _venue_id,
            } => {}
            ExecutionEvent::OrderFilled {
                id: _id,
                symbol_id: _symbol_id,
                side: _side,
                price: _price,
                qty: _qty,
                leaves_qty: _leaves_qty,
                is_maker: _is_maker,
            } => {}
            ExecutionEvent::OrderClosed {
                id: _id,
                symbol_id: _symbol_id,
                reason: _reason,
            } => {}
            ExecutionEvent::PositionChanged {
                symbol_id: _symbol_id,
                delta: _delta,
            } => {}
        }
    }

    fn on_bbo(&mut self, event: BboEvent, ctx: &mut Ctx<F>) {
        match event {
            BboEvent::Connected => {}
            BboEvent::Disconnected => {}
            BboEvent::Updated(symbol_id) if symbol_id == SAMPLE_SYMBOL_ID => {
                let bbo = ctx.bbo.bbo(symbol_id).unwrap();
                self.bbo_factors.on_bbo(bbo);
                self.mix_factors.on_bbo(bbo);
                if !self.has_warmup {
                    return;
                }
                if !self.is_sample_event() {
                    return;
                }
                let Some(sample) = self.get_sample(ctx, "bbo") else {
                    return;
                };
                self.sample_writer.add_sample(sample);
            }
            BboEvent::Updated(_) => {}
        }
    }

    fn on_trade(&mut self, event: TradeEvent, ctx: &mut Ctx<F>) {
        match event {
            TradeEvent::Connected => {}
            TradeEvent::Disconnected => {}
            TradeEvent::Trade(trade) if trade.symbol_id == SAMPLE_SYMBOL_ID => {
                self.trade_factors.on_trade(trade);
                self.mix_factors.on_trade(trade);
                if !self.has_warmup {
                    return;
                }
                if !self.is_sample_event() {
                    return;
                }
                let Some(sample) = self.get_sample(ctx, "trade") else {
                    return;
                };
                self.sample_writer.add_sample(sample);
            }
            TradeEvent::Trade(_) => {}
        }
    }

    fn on_depth(&mut self, event: DepthEvent, ctx: &mut Ctx<F>) {
        match event {
            DepthEvent::Connected => {}
            DepthEvent::Disconnected => self.depth_factors.reset(),
            DepthEvent::Synced(symbol_id) | DepthEvent::Updated(symbol_id)
                if symbol_id == SAMPLE_SYMBOL_ID =>
            {
                let book = ctx.depth.book(symbol_id).unwrap();
                self.depth_factors.on_depth(ctx.now(), book);
                self.mix_factors.on_depth(book);
                if !self.has_warmup {
                    return;
                }
                if !self.is_sample_event() {
                    return;
                }
                let Some(sample) = self.get_sample(ctx, "depth") else {
                    return;
                };
                self.sample_writer.add_sample(sample);
            }
            DepthEvent::Synced(_) | DepthEvent::Updated(_) => {}
            DepthEvent::Desynced(symbol_id) if symbol_id == SAMPLE_SYMBOL_ID => {
                self.depth_factors.reset();
            }
            DepthEvent::Desynced(_) => {}
        }
    }

    fn on_timer(&mut self, event: TimerEvent, _ctx: &mut Ctx<F>) {
        match event.id {
            WARMUP_TIMER_ID => {
                self.has_warmup = true;
            }
            _ => {}
        }
    }

    fn on_stop(&mut self, _ctx: &mut Ctx<F>) {
        self.sample_writer.finish();
    }
}

impl FactorsGeneratorStrategy {
    /// Counts the event before deciding, so the stride stays aligned when get_sample has no
    /// sample at the end of the data.
    fn is_sample_event(&mut self) -> bool {
        let is_sample_event = self.events_after_warmup % SAMPLE_STRIDE == 0;
        self.events_after_warmup += 1;
        is_sample_event
    }

    fn get_sample<F: Feeds>(&self, ctx: &mut Ctx<F>, event: &str) -> Option<Sample> {
        let now = ctx.now();
        let book = ctx
            .depth
            .book(SAMPLE_SYMBOL_ID)
            .expect("depth book must exist");
        let [
            obi_0_25bps,
            obi_0_5bps,
            obi_1bps,
            obi_2_5bps,
            obi_5bps,
            obi_10bps,
        ] = [0.25, 0.5, 1.0, 2.5, 5.0, 10.0].map(|h| self.depth_factors.obi(book, h));
        Some(Sample {
            time: now.to_string(),
            time_ns: now.as_nanos(),
            event: event.to_owned(),
            fmid_0: self.get_fmid(ctx, NanosDelta::ZERO)?,
            fmid_350us: self.get_fmid(ctx, NanosDelta::from_micros(350))?,
            fmid_700us: self.get_fmid(ctx, NanosDelta::from_micros(700))?,
            fmid_1ms: self.get_fmid(ctx, NanosDelta::from_millis(1))?,
            fmid_2ms: self.get_fmid(ctx, NanosDelta::from_millis(2))?,
            fmid_5ms: self.get_fmid(ctx, NanosDelta::from_millis(5))?,
            fmid_25ms: self.get_fmid(ctx, NanosDelta::from_millis(25))?,
            fmid_50ms: self.get_fmid(ctx, NanosDelta::from_millis(50))?,
            fmid_100ms: self.get_fmid(ctx, NanosDelta::from_millis(100))?,
            fmid_200ms: self.get_fmid(ctx, NanosDelta::from_millis(200))?,
            fmid_300ms: self.get_fmid(ctx, NanosDelta::from_millis(300))?,
            fmid_500ms: self.get_fmid(ctx, NanosDelta::from_millis(500))?,
            fmid_750ms: self.get_fmid(ctx, NanosDelta::from_millis(750))?,
            fmid_1s: self.get_fmid(ctx, NanosDelta::from_secs(1))?,
            fmid_2s: self.get_fmid(ctx, NanosDelta::from_secs(2))?,
            fmid_5s: self.get_fmid(ctx, NanosDelta::from_secs(5))?,
            fmid_10s: self.get_fmid(ctx, NanosDelta::from_secs(10))?,
            fmid_30s: self.get_fmid(ctx, NanosDelta::from_secs(30))?,
            fmid_60s: self.get_fmid(ctx, NanosDelta::from_secs(60))?,
            obi_0bps: self.bbo_factors.l1_imbalance(now),
            obi_0_25bps,
            obi_0_5bps,
            obi_1bps,
            obi_2_5bps,
            obi_5bps,
            obi_10bps,
            ofi_30us: self.bbo_factors.ofi(now, NanosDelta::from_micros(30)),
            ofi_150us: self.bbo_factors.ofi(now, NanosDelta::from_micros(150)),
            ofi_500us: self.bbo_factors.ofi(now, NanosDelta::from_micros(500)),
            ofi_1ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(1)),
            ofi_3ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(3)),
            ofi_5ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(5)),
            ofi_10ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(10)),
            ofi_25ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(25)),
            ofi_50ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(50)),
            ofi_75ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(75)),
            ofi_100ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(100)),
            ofi_200ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(200)),
            ofi_300ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(300)),
            ofi_400ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(400)),
            ofi_500ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(500)),
            ofi_750ms: self.bbo_factors.ofi(now, NanosDelta::from_millis(750)),
            ofi_1s: self.bbo_factors.ofi(now, NanosDelta::from_secs(1)),
            ofi_3s: self.bbo_factors.ofi(now, NanosDelta::from_secs(3)),
            mlofi_0_5bps_30us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(30), 0.5),
            mlofi_0_5bps_150us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(150), 0.5),
            mlofi_0_5bps_500us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(500), 0.5),
            mlofi_0_5bps_1ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(1), 0.5),
            mlofi_0_5bps_3ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(3), 0.5),
            mlofi_0_5bps_5ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(5), 0.5),
            mlofi_0_5bps_10ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(10), 0.5),
            mlofi_0_5bps_25ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(25), 0.5),
            mlofi_0_5bps_50ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(50), 0.5),
            mlofi_0_5bps_75ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(75), 0.5),
            mlofi_0_5bps_100ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(100), 0.5),
            mlofi_0_5bps_200ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(200), 0.5),
            mlofi_0_5bps_300ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(300), 0.5),
            mlofi_0_5bps_400ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(400), 0.5),
            mlofi_0_5bps_500ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(500), 0.5),
            mlofi_0_5bps_750ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(750), 0.5),
            mlofi_0_5bps_1s: self.depth_factors.mlofi(now, NanosDelta::from_secs(1), 0.5),
            mlofi_0_5bps_3s: self.depth_factors.mlofi(now, NanosDelta::from_secs(3), 0.5),
            mlofi_1bps_30us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(30), 1.0),
            mlofi_1bps_150us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(150), 1.0),
            mlofi_1bps_500us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(500), 1.0),
            mlofi_1bps_1ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(1), 1.0),
            mlofi_1bps_3ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(3), 1.0),
            mlofi_1bps_5ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(5), 1.0),
            mlofi_1bps_10ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(10), 1.0),
            mlofi_1bps_25ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(25), 1.0),
            mlofi_1bps_50ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(50), 1.0),
            mlofi_1bps_75ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(75), 1.0),
            mlofi_1bps_100ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(100), 1.0),
            mlofi_1bps_200ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(200), 1.0),
            mlofi_1bps_300ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(300), 1.0),
            mlofi_1bps_400ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(400), 1.0),
            mlofi_1bps_500ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(500), 1.0),
            mlofi_1bps_750ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(750), 1.0),
            mlofi_1bps_1s: self.depth_factors.mlofi(now, NanosDelta::from_secs(1), 1.0),
            mlofi_1bps_3s: self.depth_factors.mlofi(now, NanosDelta::from_secs(3), 1.0),
            mlofi_2_5bps_30us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(30), 2.5),
            mlofi_2_5bps_150us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(150), 2.5),
            mlofi_2_5bps_500us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(500), 2.5),
            mlofi_2_5bps_1ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(1), 2.5),
            mlofi_2_5bps_3ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(3), 2.5),
            mlofi_2_5bps_5ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(5), 2.5),
            mlofi_2_5bps_10ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(10), 2.5),
            mlofi_2_5bps_25ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(25), 2.5),
            mlofi_2_5bps_50ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(50), 2.5),
            mlofi_2_5bps_75ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(75), 2.5),
            mlofi_2_5bps_100ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(100), 2.5),
            mlofi_2_5bps_200ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(200), 2.5),
            mlofi_2_5bps_300ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(300), 2.5),
            mlofi_2_5bps_400ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(400), 2.5),
            mlofi_2_5bps_500ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(500), 2.5),
            mlofi_2_5bps_750ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(750), 2.5),
            mlofi_2_5bps_1s: self.depth_factors.mlofi(now, NanosDelta::from_secs(1), 2.5),
            mlofi_2_5bps_3s: self.depth_factors.mlofi(now, NanosDelta::from_secs(3), 2.5),
            mlofi_5bps_30us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(30), 5.0),
            mlofi_5bps_150us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(150), 5.0),
            mlofi_5bps_500us: self
                .depth_factors
                .mlofi(now, NanosDelta::from_micros(500), 5.0),
            mlofi_5bps_1ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(1), 5.0),
            mlofi_5bps_3ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(3), 5.0),
            mlofi_5bps_5ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(5), 5.0),
            mlofi_5bps_10ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(10), 5.0),
            mlofi_5bps_25ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(25), 5.0),
            mlofi_5bps_50ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(50), 5.0),
            mlofi_5bps_75ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(75), 5.0),
            mlofi_5bps_100ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(100), 5.0),
            mlofi_5bps_200ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(200), 5.0),
            mlofi_5bps_300ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(300), 5.0),
            mlofi_5bps_400ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(400), 5.0),
            mlofi_5bps_500ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(500), 5.0),
            mlofi_5bps_750ms: self
                .depth_factors
                .mlofi(now, NanosDelta::from_millis(750), 5.0),
            mlofi_5bps_1s: self.depth_factors.mlofi(now, NanosDelta::from_secs(1), 5.0),
            mlofi_5bps_3s: self.depth_factors.mlofi(now, NanosDelta::from_secs(3), 5.0),
            l1_depth_ratio: self.bbo_factors.l1_depth_ratio(now),
            tfi_30us: self.trade_factors.tfi(now, NanosDelta::from_micros(30)),
            tfi_150us: self.trade_factors.tfi(now, NanosDelta::from_micros(150)),
            tfi_500us: self.trade_factors.tfi(now, NanosDelta::from_micros(500)),
            tfi_1ms: self.trade_factors.tfi(now, NanosDelta::from_millis(1)),
            tfi_3ms: self.trade_factors.tfi(now, NanosDelta::from_millis(3)),
            tfi_5ms: self.trade_factors.tfi(now, NanosDelta::from_millis(5)),
            tfi_10ms: self.trade_factors.tfi(now, NanosDelta::from_millis(10)),
            tfi_25ms: self.trade_factors.tfi(now, NanosDelta::from_millis(25)),
            tfi_50ms: self.trade_factors.tfi(now, NanosDelta::from_millis(50)),
            tfi_75ms: self.trade_factors.tfi(now, NanosDelta::from_millis(75)),
            tfi_100ms: self.trade_factors.tfi(now, NanosDelta::from_millis(100)),
            tfi_200ms: self.trade_factors.tfi(now, NanosDelta::from_millis(200)),
            tfi_300ms: self.trade_factors.tfi(now, NanosDelta::from_millis(300)),
            tfi_400ms: self.trade_factors.tfi(now, NanosDelta::from_millis(400)),
            tfi_500ms: self.trade_factors.tfi(now, NanosDelta::from_millis(500)),
            tfi_750ms: self.trade_factors.tfi(now, NanosDelta::from_millis(750)),
            tfi_1s: self.trade_factors.tfi(now, NanosDelta::from_secs(1)),
            tfi_3s: self.trade_factors.tfi(now, NanosDelta::from_secs(3)),
            trade_notional_ratio_30us: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_micros(30)),
            trade_notional_ratio_150us: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_micros(150)),
            trade_notional_ratio_500us: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_micros(500)),
            trade_notional_ratio_1ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(1)),
            trade_notional_ratio_3ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(3)),
            trade_notional_ratio_5ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(5)),
            trade_notional_ratio_10ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(10)),
            trade_notional_ratio_25ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(25)),
            trade_notional_ratio_50ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(50)),
            trade_notional_ratio_75ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(75)),
            trade_notional_ratio_100ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(100)),
            trade_notional_ratio_200ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(200)),
            trade_notional_ratio_300ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(300)),
            trade_notional_ratio_400ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(400)),
            trade_notional_ratio_500ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(500)),
            trade_notional_ratio_750ms: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_millis(750)),
            trade_notional_ratio_1s: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_secs(1)),
            trade_notional_ratio_3s: self
                .trade_factors
                .trade_notional_ratio(now, NanosDelta::from_secs(3)),
            mid_ret_30us: self.bbo_factors.mid_ret(now, NanosDelta::from_micros(30)),
            mid_ret_150us: self.bbo_factors.mid_ret(now, NanosDelta::from_micros(150)),
            mid_ret_500us: self.bbo_factors.mid_ret(now, NanosDelta::from_micros(500)),
            mid_ret_1ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(1)),
            mid_ret_3ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(3)),
            mid_ret_5ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(5)),
            mid_ret_10ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(10)),
            mid_ret_25ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(25)),
            mid_ret_50ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(50)),
            mid_ret_75ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(75)),
            mid_ret_100ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(100)),
            mid_ret_200ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(200)),
            mid_ret_300ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(300)),
            mid_ret_400ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(400)),
            mid_ret_500ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(500)),
            mid_ret_750ms: self.bbo_factors.mid_ret(now, NanosDelta::from_millis(750)),
            mid_ret_1s: self.bbo_factors.mid_ret(now, NanosDelta::from_secs(1)),
            mid_ret_3s: self.bbo_factors.mid_ret(now, NanosDelta::from_secs(3)),
        })
    }

    fn get_fmid<F: Feeds>(&self, ctx: &mut Ctx<F>, delta: NanosDelta) -> Option<f64> {
        let mid_price = ctx.bbo.bbo(SAMPLE_SYMBOL_ID)?.mid_price();
        let future_bbo = ctx
            .ahead
            .future_bbo_by_transaction_time(SAMPLE_SYMBOL_ID, delta)?;
        Some(future_bbo.mid_price() / mid_price - 1.0)
    }
}
