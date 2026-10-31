use infra_adapters::{BboEvent, DepthEvent, ExecutionEvent, TimerEvent, TradeEvent};

use infra_kernel::{Ctx, Feeds, Strategy};

pub(crate) struct NoopStrategy {}

impl NoopStrategy {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<F: Feeds> Strategy<F> for NoopStrategy {
    fn on_start(&mut self, _ctx: &mut Ctx<F>) {}

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

    fn on_bbo(&mut self, event: BboEvent, _ctx: &mut Ctx<F>) {
        match event {
            BboEvent::Connected => {}
            BboEvent::Disconnected => {}
            BboEvent::Updated(_symbol_id) => {}
        }
    }

    fn on_trade(&mut self, event: TradeEvent, _ctx: &mut Ctx<F>) {
        match event {
            TradeEvent::Connected => {}
            TradeEvent::Disconnected => {}
            TradeEvent::Trade(_trade) => {}
        }
    }

    fn on_depth(&mut self, event: DepthEvent, _ctx: &mut Ctx<F>) {
        match event {
            DepthEvent::Connected => {}
            DepthEvent::Disconnected => {}
            DepthEvent::Synced(_symbol_id) | DepthEvent::Updated(_symbol_id) => {}
            DepthEvent::Desynced(_symbol_id) => {}
        }
    }

    fn on_timer(&mut self, event: TimerEvent, _ctx: &mut Ctx<F>) {
        match event {
            TimerEvent {
                id: _id,
                scheduled: _scheduled,
            } => {}
        }
    }

    fn on_stop(&mut self, _ctx: &mut Ctx<F>) {}
}
