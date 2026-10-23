//! Lazy time series of [`UnstructuredGrid`] steps with a bounded in-memory cache (`M1.7`).
//!
//! Per `docs/design-plan.md`, a [`TimeSeries`] holds a list of `(time, lazy_loader)` entries.
//! Only the requested step and a small LRU cache stay resident in memory so transient datasets
//! with many steps do not require parsing or retaining every step at once.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::error::CoreError;
use crate::grid::UnstructuredGrid;

/// Default maximum number of [`UnstructuredGrid`] steps retained in a [`TimeSeries`] cache.
pub const DEFAULT_STEP_CACHE_CAPACITY: usize = 4;

/// Shared lazy loader function for a single time step in a [`TimeSeries`].
pub type StepLoader<E> = Arc<dyn Fn() -> Result<UnstructuredGrid, E> + Send + Sync>;

#[derive(Debug)]
struct StepCache {
    capacity: usize,
    entries: VecDeque<(usize, Arc<UnstructuredGrid>)>,
    load_executions: usize,
    cache_hits: usize,
}

impl StepCache {
    const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
            load_executions: 0,
            cache_hits: 0,
        }
    }

    fn get(&mut self, step_index: usize) -> Option<Arc<UnstructuredGrid>> {
        let pos = self
            .entries
            .iter()
            .position(|(idx, _)| *idx == step_index)?;
        let entry = self.entries.remove(pos)?;
        let grid = Arc::clone(&entry.1);
        self.entries.push_back(entry);
        self.cache_hits += 1;
        Some(grid)
    }

    fn insert(&mut self, step_index: usize, grid: Arc<UnstructuredGrid>) {
        if self.capacity == 0 {
            return;
        }
        if let Some(pos) = self.entries.iter().position(|(idx, _)| *idx == step_index) {
            self.entries.remove(pos);
        } else if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((step_index, grid));
    }
}

/// Lazily evaluated time series of [`UnstructuredGrid`] steps (`M1.7`).
///
/// Each step stores a simulation timestamp (`f64`) and a thread-safe loader closure
/// (`StepLoader<E>`). Calling [`Self::step`] parses and caches only the requested step,
/// evicting the least-recently-used step when the cache exceeds its capacity
/// ([`DEFAULT_STEP_CACHE_CAPACITY`] by default).
pub struct TimeSeries<E = CoreError> {
    times: Vec<f64>,
    loaders: Vec<StepLoader<E>>,
    cache: Mutex<StepCache>,
}

impl<E> fmt::Debug for TimeSeries<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (cached_steps, load_executions, cache_hits, capacity) = self
            .cache
            .lock()
            .map(|c| {
                let indices: Vec<usize> = c.entries.iter().map(|(i, _)| *i).collect();
                (indices, c.load_executions, c.cache_hits, c.capacity)
            })
            .unwrap_or_default();
        f.debug_struct("TimeSeries")
            .field("times", &self.times)
            .field("cache_capacity", &capacity)
            .field("cached_steps", &cached_steps)
            .field("load_executions", &load_executions)
            .field("cache_hits", &cache_hits)
            .finish()
    }
}

impl<E> Clone for TimeSeries<E> {
    fn clone(&self) -> Self {
        let (capacity, entries, load_executions, cache_hits) = self
            .cache
            .lock()
            .map(|c| {
                (
                    c.capacity,
                    c.entries.clone(),
                    c.load_executions,
                    c.cache_hits,
                )
            })
            .unwrap_or((DEFAULT_STEP_CACHE_CAPACITY, VecDeque::new(), 0, 0));
        Self {
            times: self.times.clone(),
            loaders: self.loaders.clone(),
            cache: Mutex::new(StepCache {
                capacity,
                entries,
                load_executions,
                cache_hits,
            }),
        }
    }
}

impl<E> TimeSeries<E> {
    /// Creates a [`TimeSeries`] from an iterator of `(time, loader)` pairs.
    ///
    /// Returns [`CoreError::EmptyTimeSeries`] if the iterator is empty, or
    /// [`CoreError::InvalidTimeValue`] if any time value is `NaN` or infinite.
    pub fn from_steps<I, F>(steps: I) -> Result<Self, CoreError>
    where
        I: IntoIterator<Item = (f64, F)>,
        F: Fn() -> Result<UnstructuredGrid, E> + Send + Sync + 'static,
    {
        let mut times = Vec::new();
        let mut loaders: Vec<StepLoader<E>> = Vec::new();
        for (idx, (t, loader)) in steps.into_iter().enumerate() {
            if !t.is_finite() {
                return Err(CoreError::InvalidTimeValue { step_index: idx });
            }
            times.push(t);
            loaders.push(Arc::new(loader));
        }
        if times.is_empty() {
            return Err(CoreError::EmptyTimeSeries);
        }
        Ok(Self {
            times,
            loaders,
            cache: Mutex::new(StepCache::new(DEFAULT_STEP_CACHE_CAPACITY)),
        })
    }

    /// Creates a [`TimeSeries`] from pre-boxed [`StepLoader`]s.
    pub fn from_boxed_steps(steps: Vec<(f64, StepLoader<E>)>) -> Result<Self, CoreError> {
        if steps.is_empty() {
            return Err(CoreError::EmptyTimeSeries);
        }
        let mut times = Vec::with_capacity(steps.len());
        let mut loaders = Vec::with_capacity(steps.len());
        for (idx, (t, loader)) in steps.into_iter().enumerate() {
            if !t.is_finite() {
                return Err(CoreError::InvalidTimeValue { step_index: idx });
            }
            times.push(t);
            loaders.push(loader);
        }
        Ok(Self {
            times,
            loaders,
            cache: Mutex::new(StepCache::new(DEFAULT_STEP_CACHE_CAPACITY)),
        })
    }

    /// Creates a single-step [`TimeSeries`] from an already-loaded [`UnstructuredGrid`].
    pub fn from_single_grid(time: f64, grid: UnstructuredGrid) -> Result<Self, CoreError> {
        Self::from_single_shared_grid(time, Arc::new(grid))
    }

    /// Creates a single-step [`TimeSeries`] from a shared [`Arc<UnstructuredGrid>`].
    pub fn from_single_shared_grid(
        time: f64,
        grid: Arc<UnstructuredGrid>,
    ) -> Result<Self, CoreError> {
        if !time.is_finite() {
            return Err(CoreError::InvalidTimeValue { step_index: 0 });
        }
        let cloned_for_loader = Arc::clone(&grid);
        let loader: StepLoader<E> = Arc::new(move || Ok((*cloned_for_loader).clone()));
        let mut cache = StepCache::new(DEFAULT_STEP_CACHE_CAPACITY);
        cache.insert(0, grid);
        Ok(Self {
            times: vec![time],
            loaders: vec![loader],
            cache: Mutex::new(cache),
        })
    }

    /// Sets the maximum number of steps retained simultaneously in the step cache.
    #[must_use]
    pub fn with_cache_capacity(self, capacity: usize) -> Self {
        if let Ok(mut cache) = self.cache.lock() {
            cache.capacity = capacity;
            while cache.entries.len() > capacity {
                cache.entries.pop_front();
            }
        }
        self
    }

    /// Returns the number of time steps in the series (always `>= 1`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `false` since a valid [`TimeSeries`] always has at least one step.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Returns the slice of simulation time values for all steps in order.
    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Returns the simulation time for `step_index`, or `None` if out of bounds.
    #[must_use]
    pub fn time_at(&self, step_index: usize) -> Option<f64> {
        self.times.get(step_index).copied()
    }

    /// Returns the `[min, max]` simulation time span across all steps in the series.
    #[must_use]
    pub fn time_bounds(&self) -> [f64; 2] {
        let mut min_t = f64::INFINITY;
        let mut max_t = f64::NEG_INFINITY;
        for &t in &self.times {
            min_t = min_t.min(t);
            max_t = max_t.max(t);
        }
        [min_t, max_t]
    }

    /// Finds the zero-based step index whose time value is closest to `target_time`.
    #[must_use]
    pub fn nearest_step_for_time(&self, target_time: f64) -> usize {
        let mut best_idx = 0;
        let mut best_diff = f64::INFINITY;
        for (idx, &t) in self.times.iter().enumerate() {
            let diff = (t - target_time).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = idx;
            }
        }
        best_idx
    }

    /// Lazily loads (or retrieves from the bounded cache) the [`UnstructuredGrid`] at `step_index`.
    ///
    /// Returns `Ok(None)` if `step_index >= self.len()`, or `Ok(Some(grid))` on success.
    pub fn try_step(&self, step_index: usize) -> Result<Option<Arc<UnstructuredGrid>>, E> {
        let Some(loader) = self.loaders.get(step_index) else {
            return Ok(None);
        };
        if let Ok(mut cache) = self.cache.lock()
            && let Some(cached) = cache.get(step_index)
        {
            return Ok(Some(cached));
        }

        let grid = Arc::new(loader()?);
        if let Ok(mut cache) = self.cache.lock() {
            cache.load_executions += 1;
            cache.insert(step_index, Arc::clone(&grid));
        }
        Ok(Some(grid))
    }

    /// Lazily loads (or retrieves from the bounded cache) the [`UnstructuredGrid`] at `step_index`,
    /// converting an out-of-bounds index via `From<CoreError>`.
    pub fn step(&self, step_index: usize) -> Result<Arc<UnstructuredGrid>, E>
    where
        E: From<CoreError>,
    {
        let num_steps = self.len();
        self.try_step(step_index)?.ok_or_else(|| {
            E::from(CoreError::TimeStepOutOfBounds {
                step_index,
                num_steps,
            })
        })
    }

    /// Maps the loader error type `E` to `E2` while preserving cached grids and counters.
    pub fn map_err<E2, F>(self, map_fn: F) -> TimeSeries<E2>
    where
        E: 'static,
        E2: 'static,
        F: Fn(E) -> E2 + Send + Sync + 'static,
    {
        let map_fn = Arc::new(map_fn);
        let loaders = self
            .loaders
            .into_iter()
            .map(|loader| {
                let map_fn = Arc::clone(&map_fn);
                let wrapped: StepLoader<E2> = Arc::new(move || loader().map_err(|e| map_fn(e)));
                wrapped
            })
            .collect();
        TimeSeries {
            times: self.times,
            loaders,
            cache: self.cache,
        }
    }

    /// Returns `true` if `step_index` is currently resident in the step cache.
    #[must_use]
    pub fn is_step_cached(&self, step_index: usize) -> bool {
        self.cache
            .lock()
            .map(|c| c.entries.iter().any(|(idx, _)| *idx == step_index))
            .unwrap_or(false)
    }

    /// Returns the number of steps currently resident in the step cache.
    #[must_use]
    pub fn cached_step_count(&self) -> usize {
        self.cache.lock().map(|c| c.entries.len()).unwrap_or(0)
    }

    /// Returns the total number of lazy step loader executions (cache misses) so far.
    #[must_use]
    pub fn load_executions(&self) -> usize {
        self.cache.lock().map(|c| c.load_executions).unwrap_or(0)
    }

    /// Returns the total number of step cache hits so far.
    #[must_use]
    pub fn cache_hits(&self) -> usize {
        self.cache.lock().map(|c| c.cache_hits).unwrap_or(0)
    }

    /// Evicts all currently cached [`UnstructuredGrid`] steps and resets counters.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.entries.clear();
            cache.load_executions = 0;
            cache.cache_hits = 0;
        }
    }
}
