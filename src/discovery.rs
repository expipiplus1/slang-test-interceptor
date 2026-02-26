//! Concurrent discovery module
//!
//! Handles the three discovery phases that run concurrently:
//! 1. API detection - runs slang-test to detect supported APIs
//! 2. Timing cache loading - loads cached timing data from disk
//! 3. Test discovery via -dry-run - enumerates all tests
//!
//! All three phases stream their data back via separate channels to a unified
//! loop that uses select! to integrate results and handle interrupts.

use anyhow::{Context, Result};
use colored::Colorize;
use crossbeam_channel::{bounded, select, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::api::UnsupportedApis;
use crate::event_log::log_event;
use crate::runner::{is_interrupted, reap_process, reap_process_with_label};
use crate::timing::{BuildType, TimingCache};
use crate::types::{test_to_timing_key, TestId};

/// Result of the concurrent discovery phase
pub struct DiscoveryResult {
    /// All discovered tests (sorted)
    pub tests: Vec<String>,
    /// Unsupported APIs (or None if check disabled/failed)
    pub unsupported_apis: Option<UnsupportedApis>,
    /// Loaded timing cache
    pub timing_cache: TimingCache,
    /// Count of tests ignored due to unsupported APIs
    pub api_ignored_count: usize,
    /// APIs found in tests but not in the Check output
    pub unknown_apis: HashSet<String>,
    /// Whether API check completed successfully (can skip per-batch detection)
    pub skip_api_detection: bool,
}

/// Configuration for discovery
pub struct DiscoveryConfig<'a> {
    pub slang_test: &'a PathBuf,
    pub root_dir: &'a PathBuf,
    pub filters: &'a [String],
    pub ignore_patterns: &'a [String],
    pub apis: &'a [String],
    pub ignore_apis: &'a [String],
    pub no_early_api_check: bool,
    pub no_timing_cache: bool,
    pub build_type: Option<BuildType>,
    pub gpu_jobs: Option<usize>,
    pub machine_output: bool,
    pub num_workers: usize,
}

/// Run all three discovery phases concurrently and collect results.
/// Returns when all phases are complete or interrupted.
pub fn run_concurrent_discovery(config: &DiscoveryConfig) -> Result<DiscoveryResult> {
    // Create separate channels for each discovery source
    let (test_tx, test_rx) = bounded::<String>(1000);
    let (test_err_tx, test_err_rx) = bounded::<String>(1);
    let (api_tx, api_rx) = bounded::<UnsupportedApis>(1);
    let (timing_tx, timing_rx) = bounded::<TimingCache>(1);
    let (compiling_tx, compiling_rx) = bounded::<()>(1);

    // Interrupt signal channel - wakes up select! on Ctrl-C
    let (sig_tx, sig_rx) = bounded::<()>(1);
    let running = Arc::new(AtomicBool::new(true));

    // Hook into the existing interrupt system
    let r = running.clone();
    thread::Builder::new()
        .name("interrupt-poll".to_string())
        .spawn(move || {
            // Poll for interrupt and signal the channel
            while r.load(Ordering::SeqCst) {
                if is_interrupted() {
                    r.store(false, Ordering::SeqCst);
                    let _ = sig_tx.send(());
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
        })
        .expect("failed to spawn interrupt-poll thread");

    // Spawn all three discovery threads
    spawn_test_discovery(
        test_tx,
        test_err_tx,
        compiling_tx,
        config.slang_test.clone(),
        config.root_dir.clone(),
        config.filters.to_vec(),
        config.ignore_patterns.to_vec(),
        config.apis.to_vec(),
        config.ignore_apis.to_vec(),
    )?;

    if !config.no_early_api_check {
        spawn_api_detection(api_tx, config.slang_test.clone(), config.root_dir.clone());
    }

    if !config.no_timing_cache {
        if let Some(_build_type) = config.build_type {
            spawn_timing_cache_loader(timing_tx);
        }
    }

    // Results
    let mut tests: Vec<String> = Vec::new();
    let mut unsupported_apis: Option<UnsupportedApis> = None;
    let mut timing_cache = TimingCache::default();
    let mut error: Option<String> = None;

    // For progress display with ETA
    let mut total_predicted: f64 = 0.0;
    let mut longest_test: f64 = 0.0;

    // Progress bar for TTY mode
    let discovery_pb = if config.machine_output {
        None
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner().template("{msg}").unwrap());
        pb.set_message("Discovering tests...");
        Some(pb)
    };

    let mut shown_compiling = false;
    let mut dirty = false;
    let mut has_tests = false;

    // Channel state tracking - set to true when channel is disconnected
    let mut test_channel_closed = false;
    let mut api_channel_closed = config.no_early_api_check;
    let mut timing_channel_closed = config.no_timing_cache || config.build_type.is_none();

    // Helper to process a test message
    let handle_test = |test: String,
                           timing_cache: &TimingCache,
                           total_predicted: &mut f64,
                           longest_test: &mut f64| {
        if !timing_cache.timings_by_build.is_empty() {
            if let Some(bt) = config.build_type {
                let pred = timing_cache.predict(bt, &test_to_timing_key(&test));
                *total_predicted += pred;
                *longest_test = longest_test.max(pred);
            }
        }
        test
    };

    // Main discovery loop using select!
    while running.load(Ordering::SeqCst) {
        // 1. Priority drain - fast non-blocking receive of all pending messages
        loop {
            match test_rx.try_recv() {
                Ok(test) => {
                    let test = handle_test(test, &timing_cache, &mut total_predicted, &mut longest_test);
                    tests.push(test);
                    dirty = true;
                    has_tests = true;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    test_channel_closed = true;
                    break;
                }
            }
        }

        if let Ok(err) = test_err_rx.try_recv() {
            error = Some(err);
        }

        match api_rx.try_recv() {
            Ok(result) => {
                unsupported_apis = Some(result);
                api_channel_closed = true;
                dirty = true;
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                api_channel_closed = true;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }

        match timing_rx.try_recv() {
            Ok(cache) => {
                // Recalculate predictions for already-collected tests
                if let Some(bt) = config.build_type {
                    total_predicted = 0.0;
                    longest_test = 0.0;
                    for test in &tests {
                        let pred = cache.predict(bt, &test_to_timing_key(test));
                        total_predicted += pred;
                        longest_test = longest_test.max(pred);
                    }
                }
                timing_cache = cache;
                timing_channel_closed = true;
                dirty = true;
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                timing_channel_closed = true;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }

        if compiling_rx.try_recv().is_ok() && !shown_compiling {
            if let Some(ref pb) = discovery_pb {
                pb.set_message("\x1b[2mCompiling core module...\x1b[0m".to_string());
            }
            shown_compiling = true;
        }

        // 2. Check for errors
        if error.is_some() {
            break;
        }

        // 3. Check if all channels are done
        if test_channel_closed && api_channel_closed && timing_channel_closed {
            break;
        }

        // 4. Update progress display
        if dirty && has_tests {
            if let Some(ref pb) = discovery_pb {
                let displayed_workers = config.num_workers.min(tests.len().max(1));
                let eta = if !timing_cache.timings_by_build.is_empty() && !tests.is_empty() {
                    let parallel_eta = total_predicted / displayed_workers as f64;
                    Some(parallel_eta.max(longest_test))
                } else {
                    None
                };
                pb.set_message(format_running_message(tests.len(), displayed_workers, eta, 0));
            }
            dirty = false;
        }

        // 5. Park until data arrives, disconnect, or interrupt
        // Use select! to wait on all channels simultaneously
        select! {
            recv(test_rx) -> msg => {
                match msg {
                    Ok(test) => {
                        let test = handle_test(test, &timing_cache, &mut total_predicted, &mut longest_test);
                        tests.push(test);
                        dirty = true;
                        has_tests = true;
                    }
                    Err(_) => {
                        test_channel_closed = true;
                    }
                }
            }
            recv(test_err_rx) -> msg => {
                if let Ok(err) = msg {
                    error = Some(err);
                }
            }
            recv(api_rx) -> msg => {
                match msg {
                    Ok(result) => {
                        unsupported_apis = Some(result);
                        api_channel_closed = true;
                        dirty = true;
                    }
                    Err(_) => {
                        api_channel_closed = true;
                    }
                }
            }
            recv(timing_rx) -> msg => {
                match msg {
                    Ok(cache) => {
                        if let Some(bt) = config.build_type {
                            total_predicted = 0.0;
                            longest_test = 0.0;
                            for test in &tests {
                                let pred = cache.predict(bt, &test_to_timing_key(test));
                                total_predicted += pred;
                                longest_test = longest_test.max(pred);
                            }
                        }
                        timing_cache = cache;
                        timing_channel_closed = true;
                        dirty = true;
                    }
                    Err(_) => {
                        timing_channel_closed = true;
                    }
                }
            }
            recv(compiling_rx) -> _ => {
                if !shown_compiling {
                    if let Some(ref pb) = discovery_pb {
                        pb.set_message("\x1b[2mCompiling core module...\x1b[0m".to_string());
                    }
                    shown_compiling = true;
                }
            }
            recv(sig_rx) -> _ => {
                // Interrupt signal received, exit loop
                break;
            }
        }
    }

    // Stop the interrupt polling thread
    running.store(false, Ordering::SeqCst);

    // Finish progress bar
    if let Some(pb) = discovery_pb {
        pb.finish_and_clear();
    }

    // Check for errors
    if let Some(err) = error {
        anyhow::bail!("{}", err);
    }

    // Handle -g 0: mark all GPU APIs as unsupported
    if config.gpu_jobs == Some(0) {
        let mut apis = unsupported_apis.unwrap_or_else(UnsupportedApis::platform_defaults);
        apis.disable_all_gpu_apis();
        unsupported_apis = Some(apis);
    }

    // Now apply API filtering to the collected tests
    let mut api_ignored_count = 0;
    let mut unknown_apis: HashSet<String> = HashSet::new();
    let mut filtered_tests: Vec<String> = Vec::new();

    for test in tests {
        if let Some(ref apis) = unsupported_apis {
            if apis.is_test_unsupported(&test) {
                api_ignored_count += 1;
                continue;
            }
            if let Some(unknown_api) = apis.get_unknown_api(&test) {
                unknown_apis.insert(unknown_api);
            }
        }
        filtered_tests.push(test);
    }

    // Sort tests for deterministic ordering
    filtered_tests.sort();

    // Warn if API detection had errors (we'll have to detect APIs per-batch)
    if let Some(ref apis) = unsupported_apis {
        if let Some(ref err) = apis.error {
            eprintln!(
                "{}",
                format!("Warning: API detection: {}", err).dimmed()
            );
        }
    }

    // Determine if we can skip per-batch API detection
    // Only skip if API check completed successfully (no error) and no unknown APIs found
    let skip_api_detection = unsupported_apis
        .as_ref()
        .map(|u| u.check_completed && u.error.is_none() && unknown_apis.is_empty())
        .unwrap_or(false);

    Ok(DiscoveryResult {
        tests: filtered_tests,
        unsupported_apis,
        timing_cache,
        api_ignored_count,
        unknown_apis,
        skip_api_detection,
    })
}

/// Spawn the test discovery thread (-dry-run)
/// Just returns raw test names - no API filtering here
fn spawn_test_discovery(
    tx: Sender<String>,
    err_tx: Sender<String>,
    compiling_tx: Sender<()>,
    slang_test: PathBuf,
    root_dir: PathBuf,
    filters: Vec<String>,
    ignore_patterns: Vec<String>,
    apis: Vec<String>,
    ignore_apis: Vec<String>,
) -> Result<()> {
    // Compile filter regexes upfront
    let filter_regexes: Vec<Regex> = filters
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("Invalid filter regex: {}", p)))
        .collect::<Result<Vec<_>>>()?;

    let ignore_regexes: Vec<Regex> = ignore_patterns
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("Invalid ignore regex: {}", p)))
        .collect::<Result<Vec<_>>>()?;

    log_event(
        "dry_run",
        &format!("{} -dry-run -skip-api-detection", slang_test.display()),
    );

    let mut child = Command::new(&slang_test)
        .arg("-dry-run")
        .arg("-skip-api-detection")
        .current_dir(&root_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to run {} -dry-run", slang_test.display()))?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let err_tx_for_stderr = err_tx.clone();
    let reaper_label = format!("dry_run:{}", slang_test.display());

    // Stderr reader thread - check for errors and compiling
    thread::Builder::new()
        .name("discovery-stderr".to_string())
        .spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if line.contains("unknown option") && line.contains("-dry-run") {
                        let _ = err_tx_for_stderr.send(
                            "Your slang-test is too old and does not support the -dry-run option. \
                             Please update to a newer version of slang."
                                .to_string(),
                        );
                        return;
                    }
                    if line.contains("Compiling core module") {
                        let _ = compiling_tx.send(());
                    }
                }
            }
        })
        .expect("failed to spawn discovery-stderr thread");

    // Stdout reader thread - parse test names (no API filtering)
    thread::Builder::new()
        .name("discovery-stdout".to_string())
        .spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };

            // "no tests run" means we're done - close channel by dropping sender
            if line == "no tests run" {
                drop(tx);
                reap_process_with_label(child, reaper_label);
                return;
            }

            // Skip header lines
            if line.starts_with("Supported backends:") || line.starts_with("Check ") {
                continue;
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Apply ignore patterns (regex) - these are user-specified ignores
            if ignore_regexes.iter().any(|re| re.is_match(line)) {
                continue;
            }

            // Apply filter patterns (regex) - test must match at least one filter
            if !filter_regexes.is_empty() && !filter_regexes.iter().any(|re| re.is_match(line)) {
                continue;
            }

            // Apply --api and --ignore-api filters (these are different from API support detection)
            let test_id = TestId::parse(line);
            let test_api = test_id.api.as_deref();

            if !apis.is_empty() {
                match test_api {
                    Some(api) if apis.iter().any(|a| a.eq_ignore_ascii_case(api)) => {}
                    _ => continue,
                }
            }

            if !ignore_apis.is_empty() {
                if let Some(api) = test_api {
                    if ignore_apis.iter().any(|a| a.eq_ignore_ascii_case(api)) {
                        continue;
                    }
                }
            }

            // Send test name - if channel closed, exit
            if tx.send(line.to_string()).is_err() {
                break;
            }
        }

        // Channel closes when tx is dropped here
        reap_process_with_label(child, reaper_label);
    })
    .expect("failed to spawn discovery-stdout thread");

    Ok(())
}

/// Spawn the API detection thread
fn spawn_api_detection(tx: Sender<UnsupportedApis>, slang_test: PathBuf, root_dir: PathBuf) {
    thread::Builder::new()
        .name("api-detection".to_string())
        .spawn(move || {
        log_event(
            "api_check_start",
            &format!(
                "{} tests/compute/simple.slang -api cpu",
                slang_test.display()
            ),
        );

        // Start with platform defaults
        let mut unsupported = UnsupportedApis::platform_defaults();

        let child = Command::new(&slang_test)
            .arg("tests/compute/simple.slang")
            .arg("-api")
            .arg("cpu")
            .current_dir(&root_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                unsupported.error =
                    Some(format!("Failed to spawn slang-test for API check: {}", e));
                let _ = tx.send(unsupported);
                return;
            }
        };

        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                unsupported.error = Some("Failed to capture stdout for API check".to_string());
                let _ = child.kill();
                reap_process(child);
                let _ = tx.send(unsupported);
                return;
            }
        };

        let reader = BufReader::new(stdout);
        let mut saw_any_check = false;
        let mut last_check_time = std::time::Instant::now();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };

            // Parse "Check vk,vulkan: Supported" or "Check dx12,d3d12: Not Supported"
            if line.starts_with("Check ") {
                saw_any_check = true;
                last_check_time = std::time::Instant::now();

                if let Some(colon_pos) = line.find(':') {
                    let api_part = &line[6..colon_pos];
                    let status_part = line[colon_pos + 1..].trim();

                    for api in api_part.split(',') {
                        let api = api.trim();
                        if !api.is_empty() {
                            if status_part == "Not Supported" {
                                unsupported.add_unsupported(api);
                            } else if status_part == "Supported" {
                                unsupported.add_supported(api);
                            }
                        }
                    }
                }
                continue;
            }

            // Early exit after Check lines
            if saw_any_check
                && !line.starts_with("Check ")
                && !line.starts_with("Supported backends:")
            {
                break;
            }

            // Timeout safety
            if !saw_any_check && last_check_time.elapsed().as_secs() > 2 {
                break;
            }
        }

        // Kill the process since we're done early
        let _ = child.kill();
        reap_process_with_label(child, "api_check".to_string());

        unsupported.check_completed = saw_any_check;
        if !saw_any_check {
            unsupported.error = Some("No Check lines found in slang-test output".to_string());
        }

        log_event(
            "api_check_end",
            &format!(
                "unsupported={:?} completed={}",
                unsupported.unsupported.iter().collect::<Vec<_>>(),
                unsupported.check_completed
            ),
        );

        // Channel closes when tx is dropped
        let _ = tx.send(unsupported);
    })
    .expect("failed to spawn api-detection thread");
}

/// Spawn the timing cache loader thread
fn spawn_timing_cache_loader(tx: Sender<TimingCache>) {
    thread::Builder::new()
        .name("timing-cache".to_string())
        .spawn(move || {
            let cache = TimingCache::load();
            // Channel closes when tx is dropped
            let _ = tx.send(cache);
        })
        .expect("failed to spawn timing-cache thread");
}

/// Format the "Running N tests with M workers" message
pub(crate) fn format_running_message(
    num_tests: usize,
    num_workers: usize,
    predicted_eta: Option<f64>,
    api_ignored: usize,
) -> String {
    let ignored_part = if api_ignored > 0 {
        format!(
            " {}",
            format!("(ignoring {} tests on unsupported APIs)", api_ignored).dimmed()
        )
    } else {
        String::new()
    };

    match predicted_eta {
        Some(eta) => format!(
            "Running {} tests with {} workers{} {}",
            num_tests,
            num_workers,
            ignored_part,
            format!("(predicted ETA {:.0}s)", eta).dimmed()
        ),
        None => format!(
            "Running {} tests with {} workers{}",
            num_tests, num_workers, ignored_part
        ),
    }
}
