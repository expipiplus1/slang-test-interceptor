use std::collections::HashSet;

use crate::types::TestId;

/// Tracks which APIs are supported/unsupported on this system.
/// This is populated by running slang-test on a simple file at startup.
#[derive(Debug, Default, Clone)]
pub struct UnsupportedApis {
    /// Set of unsupported API names (lowercase): "dx11", "dx12", "vk", "cuda", "mtl", etc.
    pub unsupported: HashSet<String>,
    /// Set of supported API names (lowercase) - used to detect unknown APIs
    pub supported: HashSet<String>,
    /// Whether the API check completed successfully
    pub check_completed: bool,
    /// Error message if the check failed
    pub error: Option<String>,
    /// Whether all GPU APIs were marked unsupported due to -g 0
    pub gpu_disabled: bool,
}

impl UnsupportedApis {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add API aliases to a set
    fn add_with_aliases(set: &mut HashSet<String>, api: &str) {
        let api_lower = api.to_lowercase();
        set.insert(api_lower.clone());

        // Add common aliases
        match api_lower.as_str() {
            "vk" => { set.insert("vulkan".to_string()); }
            "vulkan" => { set.insert("vk".to_string()); }
            "dx11" => { set.insert("d3d11".to_string()); }
            "d3d11" => { set.insert("dx11".to_string()); }
            "dx12" => { set.insert("d3d12".to_string()); }
            "d3d12" => { set.insert("dx12".to_string()); }
            "mtl" => { set.insert("metal".to_string()); }
            "metal" => { set.insert("mtl".to_string()); }
            _ => {}
        }
    }

    /// Add an unsupported API (and its aliases)
    pub fn add_unsupported(&mut self, api: &str) {
        Self::add_with_aliases(&mut self.unsupported, api);
    }

    /// Add a supported API (and its aliases)
    pub fn add_supported(&mut self, api: &str) {
        Self::add_with_aliases(&mut self.supported, api);
    }

    /// Check if a test's API is unsupported
    pub fn is_test_unsupported(&self, test: &str) -> bool {
        let test_id = TestId::parse(test);

        // If GPU is disabled, also filter out gfx-unit-test-tool tests
        if self.gpu_disabled && test_id.path.starts_with("gfx-unit-test-tool/") {
            return true;
        }

        if let Some(api) = test_id.api {
            self.unsupported.contains(&api.to_lowercase())
        } else {
            false
        }
    }

    /// Check if a test's API is unknown (not in supported or unsupported lists)
    /// Returns the API name if unknown, None otherwise
    pub fn get_unknown_api(&self, test: &str) -> Option<String> {
        let test_id = TestId::parse(test);
        if let Some(api) = test_id.api {
            let api_lower = api.to_lowercase();
            // Tests without API suffix (like internal tests) are fine
            // Only flag APIs that we haven't seen in Check lines
            if !self.supported.contains(&api_lower) && !self.unsupported.contains(&api_lower) {
                return Some(api_lower);
            }
        }
        None
    }

    /// Get platform-default unsupported APIs based on OS.
    ///
    /// This is used purely for cosmetic purposes during discovery to show a conservative
    /// test count before the real API detection completes. It does not affect which tests
    /// actually run - that is determined by the real API detection or falls back to
    /// running all tests if detection fails.
    pub fn platform_defaults() -> Self {
        let mut result = Self::new();

        // CPU is always supported
        result.add_supported("cpu");

        #[cfg(target_os = "macos")]
        {
            // On macOS, DirectX is not available
            result.add_unsupported("dx11");
            result.add_unsupported("dx12");
            result.add_unsupported("d3d11");
            result.add_unsupported("d3d12");
            // WGPU is not supported on macOS
            result.add_unsupported("wgpu");
        }

        #[cfg(target_os = "linux")]
        {
            result.add_unsupported("dx11");
            result.add_unsupported("dx12");
            result.add_unsupported("d3d11");
            result.add_unsupported("d3d12");
            // On Linux, Metal is not available
            result.add_unsupported("mtl");
            result.add_unsupported("metal");
            // WGPU is not supported on Linux
            result.add_unsupported("wgpu");
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, Metal is not available
            result.add_unsupported("mtl");
            result.add_unsupported("metal");
            // WGPU may be supported on Windows - don't mark as unsupported
        }

        result
    }

    /// Mark all GPU APIs as unsupported (used when -g 0 is specified)
    pub fn disable_all_gpu_apis(&mut self) {
        self.add_unsupported("vk");
        self.add_unsupported("vulkan");
        self.add_unsupported("cuda");
        self.add_unsupported("dx11");
        self.add_unsupported("dx12");
        self.add_unsupported("d3d11");
        self.add_unsupported("d3d12");
        self.add_unsupported("metal");
        self.add_unsupported("mtl");
        self.gpu_disabled = true;
    }

    /// Get the list of GPU APIs that were disabled
    pub fn disabled_gpu_apis() -> &'static [&'static str] {
        &["vk", "cuda", "dx11", "dx12", "metal"]
    }
}
