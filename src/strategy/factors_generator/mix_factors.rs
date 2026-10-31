use infra_adapters::{Bbo, OrderBook, Trade};

pub(super) struct MixFactors {}

impl MixFactors {
    pub(super) fn new() -> Self {
        Self {}
    }

    pub(super) fn on_bbo(&mut self, _bbo: Bbo) {}

    pub(super) fn on_depth(&mut self, _book: &OrderBook) {}

    pub(super) fn on_trade(&mut self, _trade: Trade) {}
}
