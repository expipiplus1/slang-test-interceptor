use colored::control::SHOULD_COLORIZE;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use sysinfo::{CpuRefreshKind, RefreshKind, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::event_log::log_event;
use crate::types::{TestStats, DEBUG_START};

/// Helper to conditionally apply ANSI dim code
fn dim(s: &str) -> String {
    if SHOULD_COLORIZE.should_colorize() {
        format!("\x1b[2m{}\x1b[0m", s)
    } else {
        s.to_string()
    }
}

/// Helper to conditionally apply ANSI green code
fn green(s: &str) -> String {
    if SHOULD_COLORIZE.should_colorize() {
        format!("\x1b[32m{}\x1b[0m", s)
    } else {
        s.to_string()
    }
}

/// Helper to conditionally apply ANSI red code
fn red(s: &str) -> String {
    if SHOULD_COLORIZE.should_colorize() {
        format!("\x1b[31m{}\x1b[0m", s)
    } else {
        s.to_string()
    }
}

/// Helper to conditionally apply ANSI yellow code
fn yellow(s: &str) -> String {
    if SHOULD_COLORIZE.should_colorize() {
        format!("\x1b[33m{}\x1b[0m", s)
    } else {
        s.to_string()
    }
}

/// Sentinel value meaning "no test running" (worker is idle or between batches)
pub const WORKER_IDLE: usize = usize::MAX;

/// Progress update interval in milliseconds (used by spawn_progress_thread)
pub const PROGRESS_UPDATE_INTERVAL_MS: u64 = 16;

/// Target interval for CPU/GPU queries in milliseconds
const SYS_QUERY_INTERVAL_MS: u64 = 500;

/// How many progress updates between CPU/GPU queries
/// Calculated as SYS_QUERY_INTERVAL_MS / PROGRESS_UPDATE_INTERVAL_MS
const SYS_QUERY_THRESHOLD: u32 = (SYS_QUERY_INTERVAL_MS / PROGRESS_UPDATE_INTERVAL_MS) as u32;

/// State for a single worker, used to track what test is currently running.
/// The worker writes to this, and the progress thread reads from it.
pub struct WorkerState {
    /// Index into the current batch's test list. WORKER_IDLE means not running.
    pub current_test_idx: AtomicUsize,
    /// The current batch of tests (set when batch starts, cleared when batch ends)
    pub current_batch: Mutex<Vec<String>>,
    /// When the current test started, as microseconds since DEBUG_START (0 = not set)
    pub current_test_start_us: AtomicU64,
}

impl WorkerState {
    pub fn new() -> Self {
        Self {
            current_test_idx: AtomicUsize::new(WORKER_IDLE),
            current_batch: Mutex::new(Vec::new()),
            current_test_start_us: AtomicU64::new(0),
        }
    }

    /// Get current time as microseconds since DEBUG_START
    fn now_us() -> u64 {
        DEBUG_START.elapsed().as_micros() as u64
    }

    /// Called when a worker starts a new batch
    pub fn start_batch(&self, batch: &[String]) {
        *self.current_batch.lock().unwrap() = batch.to_vec();
        self.current_test_start_us.store(Self::now_us(), Ordering::Release);
        self.current_test_idx.store(0, Ordering::Release);
    }

    /// Called when a test completes - advance to next test
    pub fn advance(&self) {
        self.current_test_start_us.store(Self::now_us(), Ordering::Release);
        self.current_test_idx.fetch_add(1, Ordering::AcqRel);
    }

    /// Called when batch completes or worker goes idle
    pub fn clear(&self) {
        self.current_test_idx.store(WORKER_IDLE, Ordering::Release);
        self.current_test_start_us.store(0, Ordering::Release);
        self.current_batch.lock().unwrap().clear();
    }

    /// Get the currently running test name, if any
    pub fn current_test(&self) -> Option<String> {
        let idx = self.current_test_idx.load(Ordering::Acquire);
        if idx == WORKER_IDLE {
            return None;
        }
        let batch = self.current_batch.lock().unwrap();
        batch.get(idx).cloned()
    }

    /// Get how long the current test has been running, in seconds.
    /// Infrastructure for potential test-level ETA refinement.
    #[allow(dead_code)]
    pub fn current_test_elapsed_secs(&self) -> Option<f64> {
        let start_us = self.current_test_start_us.load(Ordering::Acquire);
        if start_us == 0 {
            return None;
        }
        let now_us = Self::now_us();
        Some((now_us.saturating_sub(start_us)) as f64 / 1_000_000.0)
    }
}

impl Default for WorkerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Container for all worker states
pub struct WorkerStates {
    states: Vec<WorkerState>,
}

impl WorkerStates {
    pub fn new(num_workers: usize) -> Self {
        Self {
            states: (0..num_workers).map(|_| WorkerState::new()).collect(),
        }
    }

    pub fn get(&self, worker_id: usize) -> &WorkerState {
        &self.states[worker_id]
    }
}

pub struct ProgressDisplay {
    total_files: usize,
    start_time: Instant,
    machine_output: bool,
    /// For machine output: last report time in milliseconds since start
    last_report_time_ms: AtomicUsize,
    /// For machine output: tracks if we've reported 0% and 99%
    reported_milestones: AtomicUsize, // bit 0 = 0%, bit 1 = 99%
    main_progress_bar: Option<ProgressBar>,
    worker_bars: Vec<Option<ProgressBar>>,
    verbose: bool,
    /// Cached GPU load percentage (updated periodically)
    last_gpu_load: Option<u32>,
    /// Cached CPU load percentage (updated periodically)
    last_cpu_load: f32,
    /// System info for CPU queries
    sys: System,
    /// Counter for throttling system queries
    sys_query_counter: u32,
    /// Fudge factor for ETA display (actual/predicted from historical runs)
    eta_fudge_factor: f64,
    /// Initial ETA (for calculating time column width in machine output)
    initial_eta: Option<f64>,
}

impl ProgressDisplay {
    pub fn new(total_files: usize, machine_output: bool, num_workers: usize, verbose: bool, eta_fudge_factor: f64) -> Self {
        let (main_progress_bar, worker_bars) = if machine_output {
            (None, Vec::new())
        } else {
            let mp = MultiProgress::new();

            // Worker progress bars first (so they appear above the main bar)
            // Only create them in verbose mode
            let worker_bars: Vec<Option<ProgressBar>> = if verbose {
                (0..num_workers)
                    .map(|_| {
                        let pb = mp.add(ProgressBar::new_spinner());
                        pb.set_style(
                            ProgressStyle::default_spinner()
                                .template("{msg:.dim}")
                                .unwrap(),
                        );
                        pb.set_message(""); // Empty initially
                        Some(pb)
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // Main progress bar at the bottom
            let main_pb = mp.add(ProgressBar::new(total_files as u64));
            main_pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg}")
                    .unwrap(),
            );

            // Note: mp is dropped here but the progress bars remain functional
            // because mp.add() returns handles that are independent of MultiProgress lifetime
            (Some(main_pb), worker_bars)
        };

        Self {
            total_files,
            start_time: Instant::now(),
            machine_output,
            last_report_time_ms: AtomicUsize::new(0),
            reported_milestones: AtomicUsize::new(0),
            main_progress_bar,
            worker_bars,
            verbose,
            last_gpu_load: None,
            last_cpu_load: 0.0,
            sys: System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything())),
            sys_query_counter: 0,
            eta_fudge_factor,
            initial_eta: None,
        }
    }

    /// Calculate time column width based on initial ETA (max 10x expected)
    fn time_width(&self) -> usize {
        match self.initial_eta {
            Some(eta) => {
                // Allow for 10x the initial ETA
                let max_time = (eta * 10.0).max(10.0);
                (max_time as usize).to_string().len()
            }
            None => 3, // Default fallback
        }
    }

    pub fn update(&mut self, stats: &TestStats, _files_completed: usize, batches_running: usize, _batches_remaining: usize, has_pending_batches: bool, eta_seconds: Option<f64>, worker_states: Option<&WorkerStates>) {
        // Update CPU/GPU load periodically, only in verbose mode
        if self.verbose {
            self.sys_query_counter += 1;
            if self.sys_query_counter >= SYS_QUERY_THRESHOLD {
                self.sys_query_counter = 0;
                self.last_cpu_load = get_cpu_usage(&mut self.sys);
                self.last_gpu_load = get_gpu_usage();
            }
        }

        // Apply fudge factor to ETA (historical actual/predicted ratio)
        let adjusted_eta = eta_seconds.map(|eta| eta * self.eta_fudge_factor);

        let passed = stats.passed.load(Ordering::SeqCst);
        let failed = stats.failed.load(Ordering::SeqCst);
        let expected_failed = stats.expected_failed.load(Ordering::SeqCst);
        let ignored = stats.ignored.load(Ordering::SeqCst);
        let tests_done = passed + failed + expected_failed + ignored;
        let total_failed = failed + expected_failed;
        let tests_remaining = self.total_files.saturating_sub(tests_done);
        let elapsed = self.start_time.elapsed().as_secs_f64();

        if self.machine_output {
            // Report at 0%, every 3 seconds, and at 99%
            let elapsed_ms = (elapsed * 1000.0) as usize;
            let last_report_ms = self.last_report_time_ms.load(Ordering::SeqCst);
            let reported_0 = self.reported_milestones.load(Ordering::SeqCst) & 1 != 0;

            let mut percent = (tests_done as f64 / self.total_files.max(1) as f64) * 100.0;
            // Cap at 99.9% while batches are still running
            if batches_running > 0 && percent >= 100.0 {
                percent = 99.9;
            }

            let should_report = if !reported_0 {
                true  // First report (0%)
            } else {
                elapsed_ms >= last_report_ms + 3000  // Every 3 seconds
            };

            if should_report {
                self.last_report_time_ms.store(elapsed_ms, Ordering::SeqCst);
                self.reported_milestones.store(1, Ordering::SeqCst);

                // Capture initial ETA for column width calculation
                if self.initial_eta.is_none() {
                    if let Some(eta) = adjusted_eta {
                        self.initial_eta = Some(eta);
                    }
                }

                // Calculate column widths
                let count_width = self.total_files.max(1).to_string().len();
                let time_width = self.time_width();

                let eta = match adjusted_eta {
                    Some(secs) if secs >= 1.0 => format!(" | ETA: {:>w$.0}s", secs.ceil(), w = time_width),
                    _ => String::new(),
                };
                eprintln!(
                    "[{:>2}/{:>cw$}/{:>cw$}] {:>5.1}% | {:>cw$} passed, {:>cw$} failed, {:>cw$} ignored | Elapsed: {:>tw$.0}s{}",
                    batches_running, tests_remaining, self.total_files,
                    percent, passed, total_failed, ignored, elapsed.round(), eta,
                    cw = count_width, tw = time_width
                );
            }
        } else if let Some(ref pb) = self.main_progress_bar {
            // Percentage based on tests completed vs total
            let mut percent = (tests_done as f64 / self.total_files.max(1) as f64) * 100.0;
            // Cap at 99.9% while batches are still running
            if batches_running > 0 && percent >= 100.0 {
                percent = 99.9;
            }

            // Format ETA string
            let eta = match adjusted_eta {
                Some(secs) if secs > 1.0 && tests_remaining > 0 => {
                    format!(" {}", dim(&format!("| ETA: {:.0}s", secs.ceil())))
                }
                Some(_) if tests_remaining > 0 => {
                    format!(" {}", dim("| ETA: <1s"))
                }
                _ => String::new(),
            };

            let compiling_info = if let Some(secs) = stats.get_compiling_time() {
                format!(" COMPILING({:.0}s)", secs)
            } else {
                String::new()
            };

            let stuck_info = if let Some(secs) = stats.seconds_since_last_output() {
                if secs >= 5.0 {
                    format!(" {}", dim(&format!("[no output for {:.0}s]", secs)))
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let load_info = if self.verbose {
                match self.last_gpu_load {
                    Some(gpu) => format!(" {}", dim(&format!("| CPU: {:.0}% GPU: {}%", self.last_cpu_load, gpu))),
                    None if self.last_cpu_load > 0.0 => format!(" {}", dim(&format!("| CPU: {:.0}%", self.last_cpu_load))),
                    None => String::new(),
                }
            } else {
                String::new()
            };

            // Colorize counts: green for passed, red/yellow for failed, dim for ignored
            // Red if any unexpected failures, yellow if only expected failures
            let passed_str = if passed > 0 {
                green(&format!("{} passed", passed))
            } else {
                format!("{} passed", passed)
            };
            let failed_str = if failed > 0 {
                // Has unexpected failures - show in red
                red(&format!("{} failed", total_failed))
            } else if expected_failed > 0 {
                // Only expected failures - show in yellow
                yellow(&format!("{} failed", total_failed))
            } else {
                format!("{} failed", total_failed)
            };
            let ignored_str = format!("{} ignored", ignored);

            let msg = format!(
                "[{}/{}/{}] {:.1}% {} {}, {}, {} {}{}{}{}{}",
                batches_running, tests_remaining, self.total_files,
                percent, dim("|"), passed_str, failed_str, ignored_str,
                dim(&format!("| Elapsed: {:.1}s", elapsed)),
                eta, load_info, compiling_info, stuck_info
            );
            pb.set_message(msg);

            // Update per-worker progress bars (verbose mode only)
            if self.verbose {
                if let Some(states) = worker_states {
                    for (worker_id, worker_bar_opt) in self.worker_bars.iter_mut().enumerate() {
                        if let Some(worker_bar) = worker_bar_opt {
                            let state = states.get(worker_id);

                            if let Some(test_name) = state.current_test() {
                                // Show worker line with current test
                                worker_bar.set_message(format!("  worker {}: {}", worker_id, test_name));
                            } else if !has_pending_batches {
                                // Worker idle and no batches waiting - hide the bar
                                worker_bar.set_draw_target(ProgressDrawTarget::hidden());
                                worker_bar.finish_and_clear();
                                *worker_bar_opt = None;
                            } else {
                                // Worker is between tests
                                worker_bar.set_message(format!("  worker {}:", worker_id));
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn finish(&self, stats: &TestStats) {
        // Clear worker bars first
        for worker_bar_opt in &self.worker_bars {
            if let Some(worker_bar) = worker_bar_opt {
                worker_bar.finish_and_clear();
            }
        }

        if let Some(ref pb) = self.main_progress_bar {
            pb.finish_and_clear();
        } else if self.machine_output {
            // Print final 100% status in machine mode
            let passed = stats.passed.load(Ordering::SeqCst);
            let failed = stats.failed.load(Ordering::SeqCst);
            let expected_failed = stats.expected_failed.load(Ordering::SeqCst);
            let total_failed = failed + expected_failed;
            let ignored = stats.ignored.load(Ordering::SeqCst);
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let count_width = self.total_files.max(1).to_string().len();
            let time_width = self.time_width();
            eprintln!(
                "[ 0/{:>cw$}/{:>cw$}] 100.0% | {:>cw$} passed, {:>cw$} failed, {:>cw$} ignored | Elapsed: {:>tw$.0}s",
                0, self.total_files, passed, total_failed, ignored, elapsed.round(),
                cw = count_width, tw = time_width
            );
        }
    }
}

pub struct SystemStats {
    sys: System,
}

impl SystemStats {
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::new().with_cpu(CpuRefreshKind::everything())
        );
        Self { sys }
    }

    pub fn refresh_and_log(&mut self, running: usize, pool_remaining: usize) {
        self.sys.refresh_cpu_all();

        let load = System::load_average();
        let cpu_usage: f32 = self.sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
            / self.sys.cpus().len().max(1) as f32;

        let gpu_usage = get_gpu_usage();

        let gpu_str = gpu_usage.map(|g| format!(" gpu={}%", g)).unwrap_or_default();

        log_event("stats", &format!(
            "load_1m={:.2} load_5m={:.2} cpu_avg={:.1}%{} running={} pool={}",
            load.one, load.five, cpu_usage, gpu_str, running, pool_remaining
        ));
    }
}

impl Default for SystemStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "gpu-info")]
fn get_gpu_usage() -> Option<u32> {
    let gpu = gfxinfo::active_gpu().ok()?;
    let info = gpu.info();
    Some(info.load_pct())
}

#[cfg(not(feature = "gpu-info"))]
fn get_gpu_usage() -> Option<u32> {
    None
}

fn get_cpu_usage(sys: &mut System) -> f32 {
    sys.refresh_cpu_all();
    sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
        / sys.cpus().len().max(1) as f32
}
