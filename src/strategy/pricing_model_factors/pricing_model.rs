use std::{collections::HashMap, path::Path};

use infra_adapters::{Bbo, OrderBook, SymbolId, Trade};
use infra_core::UnixNanos;
use lightgbm3::Booster;

use super::factors_generator::{FactorSample, FactorsGenerator};

/// Models trained by `train/train.py`, one `<label>.txt` per horizon.
const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/train/models");
/// The fmid label each `PredictionSet` horizon is trained on, in field order.
const HORIZON_LABELS: [&str; 19] = [
    "fmid_0",
    "fmid_350us",
    "fmid_700us",
    "fmid_1ms",
    "fmid_2ms",
    "fmid_5ms",
    "fmid_25ms",
    "fmid_50ms",
    "fmid_100ms",
    "fmid_200ms",
    "fmid_300ms",
    "fmid_500ms",
    "fmid_750ms",
    "fmid_1s",
    "fmid_2s",
    "fmid_5s",
    "fmid_10s",
    "fmid_30s",
    "fmid_60s",
];

// multi-symbol pricing model
pub(super) struct PricingModel {
    factors_generator: FactorsGenerator,
    /// Model per `HORIZON_LABELS` entry, each predicting that fmid return.
    models: [Booster; 19],
    /// Mid price of each symbol's last bbo.
    mids: HashMap<SymbolId, f64>,
}

// per symbol's multi-horizon predicted exchange's midprice, in price units
pub(super) struct PredictionSet {
    pub(super) horizon_0us: f64,
    pub(super) horizon_350us: f64,
    pub(super) horizon_700us: f64,
    pub(super) horizon_1ms: f64,
    pub(super) horizon_2ms: f64,
    pub(super) horizon_5ms: f64,
    pub(super) horizon_25ms: f64,
    pub(super) horizon_50ms: f64,
    pub(super) horizon_100ms: f64,
    pub(super) horizon_200ms: f64,
    pub(super) horizon_300ms: f64,
    pub(super) horizon_500ms: f64,
    pub(super) horizon_750ms: f64,
    pub(super) horizon_1s: f64,
    pub(super) horizon_2s: f64,
    pub(super) horizon_5s: f64,
    pub(super) horizon_10s: f64,
    pub(super) horizon_30s: f64,
    pub(super) horizon_60s: f64,
}

impl PricingModel {
    pub(super) fn new() -> Self {
        Self {
            factors_generator: FactorsGenerator::new(),
            models: load_models(Path::new(MODEL_DIR)),
            mids: HashMap::new(),
        }
    }
}

/// Loads `<label>.txt` of every horizon from `dir`.
///
/// # Panics
/// Panics if a model is missing or invalid, or its features are not
/// `FactorSample::FEATURE_NAMES`.
fn load_models(dir: &Path) -> [Booster; 19] {
    HORIZON_LABELS.map(|label| {
        let path = dir.join(format!("{label}.txt"));
        let mut model = Booster::from_file(path.to_str().expect("model path must be UTF-8"))
            .unwrap_or_else(|e| panic!("failed to load model {}: {e}", path.display()));
        let feature_names = model
            .feature_name()
            .unwrap_or_else(|e| panic!("failed to read features of {}: {e}", path.display()));
        assert!(
            feature_names == FactorSample::FEATURE_NAMES,
            "{} was trained on different features than FactorSample",
            path.display()
        );
        model
            .single_row_fast_init::<f64>(Some(&serde_json::json!({ "num_threads": 1 })))
            .unwrap_or_else(|e| panic!("failed to prepare {}: {e}", path.display()));
        model
    })
}

/// The model's prediction for one row of `FactorSample::features`.
fn predict_row(model: &Booster, features: &[f64]) -> f64 {
    model
        .predict_single_row_fast(features)
        .expect("lightgbm prediction failed")[0]
}

// update its internal state
impl PricingModel {
    pub(super) fn on_bbo(&mut self, symbol_id: SymbolId, bbo: Bbo) {
        self.mids.insert(symbol_id, bbo.mid_price());
        self.factors_generator.on_bbo(symbol_id, bbo);
    }
    pub(super) fn on_trade(&mut self, symbol_id: SymbolId, trade: Trade) {
        self.factors_generator.on_trade(symbol_id, trade);
    }
    pub(super) fn on_depth(&mut self, symbol_id: SymbolId, order_book: &OrderBook) {
        self.factors_generator.on_depth(symbol_id, order_book);
    }
}

// PricingModel is abstractly a function that takes past bbo/trade/depth data and current time,
// then outputs multi-horizon predicted exchange's midprice from current time
impl PricingModel {
    /// # Panics
    /// Panics if the symbol has had no bbo or no depth update yet; callers predict only after
    /// warmup.
    pub(super) fn predict(&self, symbol_id: SymbolId, now: UnixNanos) -> PredictionSet {
        let mid = *self
            .mids
            .get(&symbol_id)
            .expect("predict called before the symbol's first bbo");
        let features = self.factors_generator.get_sample(symbol_id, now).features();
        let [
            horizon_0us,
            horizon_350us,
            horizon_700us,
            horizon_1ms,
            horizon_2ms,
            horizon_5ms,
            horizon_25ms,
            horizon_50ms,
            horizon_100ms,
            horizon_200ms,
            horizon_300ms,
            horizon_500ms,
            horizon_750ms,
            horizon_1s,
            horizon_2s,
            horizon_5s,
            horizon_10s,
            horizon_30s,
            horizon_60s,
        ] = self
            .models
            .each_ref()
            .map(|model| mid * (1.0 + predict_row(model, &features)));
        PredictionSet {
            horizon_0us,
            horizon_350us,
            horizon_700us,
            horizon_1ms,
            horizon_2ms,
            horizon_5ms,
            horizon_25ms,
            horizon_50ms,
            horizon_100ms,
            horizon_200ms,
            horizon_300ms,
            horizon_500ms,
            horizon_750ms,
            horizon_1s,
            horizon_2s,
            horizon_5s,
            horizon_10s,
            horizon_30s,
            horizon_60s,
        }
    }
}
