mod factors_generator;
mod pricing_model;

use infra_adapters::{
    BboEvent, BinanceFuturesBboApi, BinanceFuturesDepthApi, DepthEvent, ExecutionEvent, TimerApi,
    TimerEvent, TimerId, TradeEvent,
};
use infra_core::NanosDelta;

use infra_kernel::{Ctx, Feeds, Strategy};

use crate::strategy::pricing_model_factors::pricing_model::PricingModel;

const WARMUP_TIMER_ID: TimerId = 1;
/// Predictions are withheld for this long after start, so that every factor window, up to
/// the 1h trade notional normalization, is filled.
const WARMUP_DURATION: NanosDelta = NanosDelta::from_secs(60 * 60);

pub(crate) struct PricingModelFactorsStrategy {
    pricing_model: PricingModel,
    has_warmup: bool,
}

impl PricingModelFactorsStrategy {
    pub(crate) fn new() -> Self {
        Self {
            pricing_model: PricingModel::new(),
            has_warmup: false,
        }
    }
}

impl<F: Feeds> Strategy<F> for PricingModelFactorsStrategy {
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
            BboEvent::Updated(symbol_id) => {
                let bbo = ctx.bbo.bbo(symbol_id).unwrap();
                self.pricing_model.on_bbo(symbol_id, bbo);
                if !self.has_warmup {
                    return;
                }
                let _prediction_set = self.pricing_model.predict(symbol_id, ctx.now());
            }
        }
    }

    fn on_trade(&mut self, event: TradeEvent, ctx: &mut Ctx<F>) {
        match event {
            TradeEvent::Connected => {}
            TradeEvent::Disconnected => {}
            TradeEvent::Trade(trade) => {
                self.pricing_model.on_trade(trade.symbol_id, trade);
                if !self.has_warmup {
                    return;
                }
                let _prediction_set = self.pricing_model.predict(trade.symbol_id, ctx.now());
            }
        }
    }

    fn on_depth(&mut self, event: DepthEvent, ctx: &mut Ctx<F>) {
        match event {
            DepthEvent::Connected => {}
            DepthEvent::Disconnected => {}
            DepthEvent::Synced(symbol_id) | DepthEvent::Updated(symbol_id) => {
                let book = ctx.depth.book(symbol_id).unwrap();
                self.pricing_model.on_depth(symbol_id, book);
                if !self.has_warmup {
                    return;
                }
                let _prediction_set = self.pricing_model.predict(symbol_id, ctx.now());
            }
            DepthEvent::Desynced(_symbol_id) => {}
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

    fn on_stop(&mut self, _ctx: &mut Ctx<F>) {}
}
