use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::event_log::get_state_dir;
use crate::types::{test_to_timing_key, DEFAULT_PREDICTED_DURATION, EMA_NEW_WEIGHT};

/// Build type for timing cache segmentation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BuildType {
    Debug,
    Release,
    RelWithDebInfo,
}

impl BuildType {
    /// Detect build type from slang-test path
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        let path_str = path.to_string_lossy().to_lowercase();
        if path_str.contains("relwithdebinfo") {
            Some(BuildType::RelWithDebInfo)
        } else if path_str.contains("debug") {
            Some(BuildType::Debug)
        } else if path_str.contains("release") {
            Some(BuildType::Release)
        } else {
            None
        }
    }
}

impl std::fmt::Display for BuildType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildType::Debug => write!(f, "debug"),
            BuildType::Release => write!(f, "release"),
            BuildType::RelWithDebInfo => write!(f, "relwithdebinfo"),
        }
    }
}

/// Timing cache stores per-test durations, segmented by build type.
/// Keys are test identifiers like "tests/foo.slang" or "tests/foo.slang.4" (with variant suffix).
/// The variant suffix is included when there are multiple tests from the same file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingCache {
    pub version: u32,
    /// Map from build type to (test identifier -> duration in seconds)
    /// Using String keys for JSON serialization compatibility
    #[serde(default)]
    pub timings_by_build: HashMap<String, HashMap<String, f64>>,
    /// Map from build type to (test identifier -> ETA fudge factor)
    /// Fudge factor = actual_elapsed / predicted_eta, used to correct displayed ETAs
    #[serde(default)]
    pub fudge_factors_by_build: HashMap<String, HashMap<String, f64>>,
    /// Legacy: flat timings map (for migration from version 2)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub timings: HashMap<String, f64>,
}

impl TimingCache {
    pub fn load() -> Self {
        if let Some(state_dir) = get_state_dir() {
            let path = state_dir.join("timing.json");
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(mut cache) = serde_json::from_str::<TimingCache>(&contents) {
                    // Migrate from version 2 (flat timings) to version 3 (segmented)
                    if cache.version == 2 && !cache.timings.is_empty() {
                        cache.timings_by_build.insert("release".to_string(), cache.timings.clone());
                        cache.timings.clear();
                        cache.version = 4;
                    }
                    // Migrate from version 3 to 4 (add fudge factors)
                    if cache.version == 3 {
                        cache.version = 4;
                    }
                    if cache.version >= 4 {
                        return cache;
                    }
                }
                // Old format or version 1 - just start fresh
            }
        }
        Self {
            version: 4,
            timings_by_build: HashMap::new(),
            fudge_factors_by_build: HashMap::new(),
            timings: HashMap::new(),
        }
    }

    pub fn save(&self) {
        if let Some(state_dir) = get_state_dir() {
            let _ = std::fs::create_dir_all(&state_dir);
            let path = state_dir.join("timing.json");
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = std::fs::write(&path, json);
            }
        }
    }

    /// Get the timings map for a specific build type
    fn get_timings(&self, build_type: BuildType) -> Option<&HashMap<String, f64>> {
        self.timings_by_build.get(&build_type.to_string())
    }

    /// Get or create the timings map for a specific build type
    fn get_timings_mut(&mut self, build_type: BuildType) -> &mut HashMap<String, f64> {
        self.timings_by_build
            .entry(build_type.to_string())
            .or_insert_with(HashMap::new)
    }

    /// Record a test's duration for a specific build type. Uses EMA to smooth out variations.
    pub fn record(&mut self, build_type: BuildType, test_id: &str, duration: f64) {
        let timings = self.get_timings_mut(build_type);
        let existing = timings.entry(test_id.to_string()).or_insert(0.0);
        if *existing == 0.0 {
            *existing = duration;
        } else {
            *existing = duration * EMA_NEW_WEIGHT + *existing * (1.0 - EMA_NEW_WEIGHT);
        }
    }

    /// Predict duration for a test with a specific build type.
    /// Returns DEFAULT_PREDICTED_DURATION if unknown.
    pub fn predict(&self, build_type: BuildType, test_id: &str) -> f64 {
        self.get_timings(build_type)
            .and_then(|t| t.get(test_id).copied())
            .unwrap_or(DEFAULT_PREDICTED_DURATION)
    }

    /// Check if there's timing data for a specific build type
    pub fn has_timing_data(&self, build_type: BuildType) -> bool {
        self.get_timings(build_type)
            .map(|t| !t.is_empty())
            .unwrap_or(false)
    }

    /// Merge observed timings into the cache for a specific build type
    pub fn merge(&mut self, build_type: BuildType, observed: &HashMap<String, f64>) {
        for (test_id, duration) in observed {
            self.record(build_type, test_id, *duration);
        }
    }

    /// Get the fudge factors map for a specific build type
    fn get_fudge_factors(&self, build_type: BuildType) -> Option<&HashMap<String, f64>> {
        self.fudge_factors_by_build.get(&build_type.to_string())
    }

    /// Get or create the fudge factors map for a specific build type
    fn get_fudge_factors_mut(&mut self, build_type: BuildType) -> &mut HashMap<String, f64> {
        self.fudge_factors_by_build
            .entry(build_type.to_string())
            .or_insert_with(HashMap::new)
    }

    /// Record a fudge factor for a test. Uses EMA to smooth out variations.
    pub fn record_fudge_factor(&mut self, build_type: BuildType, test_id: &str, fudge: f64) {
        let factors = self.get_fudge_factors_mut(build_type);
        let existing = factors.entry(test_id.to_string()).or_insert(1.0);
        // EMA: weight new measurement, but be conservative (slower to change)
        *existing = fudge * 0.3 + *existing * 0.7;
    }

    /// Get the average fudge factor for a set of tests.
    /// Returns 1.0 if no fudge data is available.
    pub fn average_fudge_factor(&self, build_type: BuildType, test_ids: &[String]) -> f64 {
        let Some(factors) = self.get_fudge_factors(build_type) else {
            return 1.0;
        };

        let mut sum = 0.0;
        let mut count = 0;
        for test_id in test_ids {
            let timing_key = test_to_timing_key(test_id);
            if let Some(&fudge) = factors.get(&timing_key) {
                sum += fudge;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            1.0
        }
    }

    /// Record fudge factors for all tests in a run.
    /// fudge = actual_elapsed / predicted_eta
    pub fn record_fudge_factors(&mut self, build_type: BuildType, test_ids: &[String], fudge: f64) {
        // Clamp fudge factor to reasonable range (0.5x to 3x)
        let clamped_fudge = fudge.clamp(0.5, 3.0);
        for test_id in test_ids {
            let timing_key = test_to_timing_key(test_id);
            self.record_fudge_factor(build_type, &timing_key, clamped_fudge);
        }
    }
}
