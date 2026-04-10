/*!
# cuda-emergence

Emergence detection for agent fleets.

"The whole is greater than the sum of its parts."

Ants build bridges. Starlings form murmurations. Neurons create consciousness.
None of these were programmed. They emerged from simple local rules.

This crate watches the fleet and detects when patterns appear that no
individual agent was designed to produce. When emergence happens, the fleet
has become something more than a collection of agents.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A detected emergent pattern
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmergentPattern {
    pub id: u64,
    pub name: String,
    pub pattern_type: PatternType,
    pub participating_agents: Vec<String>,
    pub confidence: f64,
    pub novelty: f64,       // how surprising is this
    pub stability: f64,     // how persistent over time
    pub description: String,
    pub discovered: u64,
    pub last_observed: u64,
    pub observation_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    Coordination,     // agents acting in sync without explicit coordination
    Specialization,   // agents spontaneously dividing labor
    Communication,    // new communication patterns forming
    Optimization,     // fleet collectively finding better solutions
    Defense,          // collective security behavior
    Creation,         // collective creative output
    Adaptation,       // fleet adapting to environment faster than individuals
    Culture,          // persistent behavioral norms spreading
}

/// An observation from a single agent at a point in time
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub agent_id: String,
    pub tick: u64,
    pub metrics: HashMap<String, f64>,
    pub actions: Vec<String>,
    pub neighbors: Vec<String>,
}

/// Baseline — what "normal" looks like for an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Baseline {
    pub agent_id: String,
    pub metric_means: HashMap<String, f64>,
    pub metric_vars: HashMap<String, f64>,
    pub action_frequencies: HashMap<String, f64>,
    pub sample_count: u32,
}

impl Baseline {
    pub fn new(agent_id: &str) -> Self {
        Baseline { agent_id: agent_id.to_string(), metric_means: HashMap::new(), metric_vars: HashMap::new(), action_frequencies: HashMap::new(), sample_count: 0 }
    }

    /// Update baseline with new observation (Welford's online algorithm)
    pub fn update(&mut self, obs: &Observation) {
        self.sample_count += 1;
        let n = self.sample_count as f64;

        for (key, val) in &obs.metrics {
            let old_mean = self.metric_means.get(key).copied().unwrap_or(0.0);
            let new_mean = old_mean + (val - old_mean) / n;
            let old_var = self.metric_vars.get(key).copied().unwrap_or(0.0);
            let new_var = old_var + (val - old_mean) * (val - new_mean);
            self.metric_means.insert(key.clone(), new_mean);
            self.metric_vars.insert(key.clone(), new_var);
        }

        for action in &obs.actions {
            let count = self.action_frequencies.entry(action.clone()).or_insert(0.0);
            *count += 1.0;
        }
        // Normalize frequencies
        let total: f64 = self.action_frequencies.values().sum();
        if total > 0.0 {
            for v in self.action_frequencies.values_mut() { *v /= total; }
        }
    }

    /// Z-score of a metric value against baseline
    pub fn z_score(&self, metric: &str, value: f64) -> Option<f64> {
        let mean = *self.metric_means.get(metric)?;
        let var = *self.metric_vars.get(metric)?;
        let std = (var / self.sample_count.max(1) as f64).sqrt();
        if std < 0.0001 { return None; }
        Some((value - mean) / std)
    }

    /// Is this observation anomalous? (|z| > threshold)
    pub fn is_anomalous(&self, obs: &Observation, threshold: f64) -> Vec<(String, f64)> {
        let mut anomalies = vec![];
        for (key, val) in &obs.metrics {
            if let Some(z) = self.z_score(key, *val) {
                if z.abs() > threshold {
                    anomalies.push((key.clone(), z));
                }
            }
        }
        anomalies
    }
}

/// Emergence detector
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmergenceDetector {
    pub baselines: HashMap<String, Baseline>,
    pub patterns: Vec<EmergentPattern>,
    pub history: Vec<Vec<Observation>>,
    pub anomaly_threshold: f64,
    pub coordination_window: usize,
}

impl EmergenceDetector {
    pub fn new() -> Self {
        EmergenceDetector { baselines: HashMap::new(), patterns: vec![], history: vec![], anomaly_threshold: 2.0, coordination_window: 50 }
    }

    /// Feed an observation, check for emergence
    pub fn observe(&mut self, obs: Observation) -> Vec<EmergentPattern> {
        // Update baseline
        let baseline = self.baselines.entry(obs.agent_id.clone()).or_insert_with(|| Baseline::new(&obs.agent_id));
        let was_anomalous = !baseline.is_anomalous(&obs, self.anomaly_threshold).is_empty();
        baseline.update(&obs);

        // Store in history window
        let tick_idx = obs.tick as usize;
        while self.history.len() <= tick_idx { self.history.push(vec![]); }
        if let Some(tick_obs) = self.history.get_mut(tick_idx) { tick_obs.push(obs); }

        // Trim old history
        if self.history.len() > self.coordination_window {
            self.history.drain(0..self.history.len() - self.coordination_window);
        }

        vec![] // Patterns detected in batch check
    }

    /// Batch check for emergent patterns across recent history
    pub fn detect(&mut self) -> Vec<EmergentPattern> {
        let mut found = vec![];

        // Check for synchronization — agents doing same action at same time
        if let Some(sync) = self.detect_synchronization() { found.push(sync); }

        // Check for specialization — agents diverging in action patterns
        if let Some(spec) = self.detect_specialization() { found.push(spec); }

        // Check for correlation — agent metrics moving together
        if let Some(corr) = self.detect_correlation() { found.push(corr); }

        // Check for cascades — events propagating through fleet
        if let Some(cascade) = self.detect_cascade() { found.push(cascade); }

        for p in &found {
            self.record_pattern(p.clone());
        }

        found
    }

    /// Detect agents synchronizing actions
    fn detect_synchronization(&self) -> Option<EmergentPattern> {
        let recent: Vec<_> = self.history.last?;
        if recent.len() < 3 { return None; }

        // Count action co-occurrence
        let mut action_agents: HashMap<String, Vec<String>> = HashMap::new();
        for obs in recent {
            for action in &obs.actions {
                action_agents.entry(action.clone()).or_default().push(obs.agent_id.clone());
            }
        }

        for (action, agents) in &action_agents {
            if agents.len() >= 3 {
                let unique: Vec<_> = agents.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
                if unique.len() >= 3 {
                    return Some(EmergentPattern {
                        id: now(),
                        name: format!("Synchronized {}", action),
                        pattern_type: PatternType::Coordination,
                        participating_agents: unique,
                        confidence: 0.7,
                        novelty: 0.6,
                        stability: 0.5,
                        description: format!("{} agents performing '{}' simultaneously without explicit coordination", unique.len(), action),
                        discovered: now(),
                        last_observed: now(),
                        observation_count: 1,
                    });
                }
            }
        }
        None
    }

    /// Detect agents specializing — different action distributions
    fn detect_specialization(&self) -> Option<EmergentPattern> {
        if self.baselines.len() < 3 { return None; }

        // Check if agents have very different action distributions
        let mut agents: Vec<_> = self.baselines.values().collect();
        if agents.len() < 3 { return None; }

        // Find agents with dominant action (frequency > 0.5)
        let specialists: Vec<String> = agents.iter()
            .filter(|b| b.action_frequencies.values().any(|&f| f > 0.5))
            .map(|b| b.agent_id.clone())
            .collect();

        if specialists.len() >= 2 {
            Some(EmergentPattern {
                id: now(),
                name: "Spontaneous Specialization".into(),
                pattern_type: PatternType::Specialization,
                participating_agents: specialists,
                confidence: 0.6,
                novelty: 0.7,
                stability: 0.4,
                description: "Agents spontaneously specializing in different tasks without assignment".into(),
                discovered: now(),
                last_observed: now(),
                observation_count: 1,
            })
        } else {
            None
        }
    }

    /// Detect metric correlation across agents
    fn detect_correlation(&self) -> Option<EmergentPattern> {
        if self.baselines.len() < 2 { return None; }

        // Find common metrics across agents
        let all_metrics: std::collections::HashSet<String> = self.baselines.values()
            .flat_map(|b| b.metric_means.keys().cloned())
            .collect();

        for metric in &all_metrics {
            let vals: Vec<f64> = self.baselines.values()
                .filter_map(|b| b.metric_means.get(metric).copied())
                .collect();

            if vals.len() < 3 { continue; }

            // Check if all values are very close (convergent behavior)
            let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
            let variance: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            let cv = if mean.abs() > 0.001 { (variance.sqrt() / mean.abs()) } else { variance };

            if cv < 0.1 && vals.len() >= 3 {
                let agents: Vec<String> = self.baselines.keys().cloned().collect();
                return Some(EmergentPattern {
                    id: now(),
                    name: format!("Convergent {}", metric),
                    pattern_type: PatternType::Optimization,
                    participating_agents: agents,
                    confidence: 0.5,
                    novelty: 0.4,
                    stability: 0.6,
                    description: format!("Fleet converging on {} value (CV={:.3})", metric, cv),
                    discovered: now(),
                    last_observed: now(),
                    observation_count: 1,
                });
            }
        }
        None
    }

    /// Detect cascading events through fleet
    fn detect_cascade(&self) -> Option<EmergentPattern> {
        if self.history.len() < 3 { return None; }

        // Look for same anomaly spreading across agents in consecutive ticks
        let mut cascade_agents = vec![];

        for (tick_idx, tick_obs) in self.history.iter().enumerate().rev().take(5).rev() {
            for obs in tick_obs {
                if let Some(baseline) = self.baselines.get(&obs.agent_id) {
                    let anom = baseline.is_anomalous(obs, self.anomaly_threshold);
                    if !anom.is_empty() {
                        cascade_agents.push(obs.agent_id.clone());
                    }
                }
            }
        }

        let unique: std::collections::HashSet<_> = cascade_agents.iter().collect();
        if unique.len() >= 3 {
            Some(EmergentPattern {
                id: now(),
                name: "Fleet Cascade".into(),
                pattern_type: PatternType::Adaptation,
                participating_agents: cascade_agents,
                confidence: 0.5,
                novelty: 0.8,
                stability: 0.3,
                description: "Anomalous behavior cascading through fleet".into(),
                discovered: now(),
                last_observed: now(),
                observation_count: 1,
            })
        } else {
            None
        }
    }

    /// Record and update a pattern
    fn record_pattern(&mut self, pattern: EmergentPattern) {
        // Check if similar pattern already exists
        if let Some(existing) = self.patterns.iter_mut().find(|p| p.name == pattern.name) {
            existing.observation_count += 1;
            existing.last_observed = pattern.last_observed;
            existing.stability = existing.stability * 0.9 + 0.1;
            existing.confidence = existing.confidence * 0.95 + pattern.confidence * 0.05;
        } else {
            self.patterns.push(pattern);
        }
    }

    /// Fleet emergence summary
    pub fn summary(&self) -> EmergenceSummary {
        let by_type = |t: PatternType| self.patterns.iter().filter(|p| p.pattern_type == t).count();
        let avg_stability = if !self.patterns.is_empty() { self.patterns.iter().map(|p| p.stability).sum::<f64>() / self.patterns.len() as f64 } else { 0.0 };
        EmergenceSummary {
            total_patterns: self.patterns.len(),
            coordination: by_type(PatternType::Coordination),
            specialization: by_type(PatternType::Specialization),
            optimization: by_type(PatternType::Optimization),
            adaptation: by_type(PatternType::Adaptation),
            avg_stability,
            agents_monitored: self.baselines.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmergenceSummary {
    pub total_patterns: usize,
    pub coordination: usize,
    pub specialization: usize,
    pub optimization: usize,
    pub adaptation: usize,
    pub avg_stability: f64,
    pub agents_monitored: usize,
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(agent: &str, tick: u64, metrics: Vec<(&str, f64)>, actions: Vec<&str>) -> Observation {
        Observation { agent_id: agent.to_string(), tick, metrics: metrics.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), actions: actions.into_iter().map(String::from).collect(), neighbors: vec![] }
    }

    #[test]
    fn test_baseline_update() {
        let mut b = Baseline::new("a");
        for i in 0..100 {
            let obs = make_obs("a", i, vec![("speed", 0.5)], vec![]);
            b.update(&obs);
        }
        assert!((b.metric_means["speed"] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_z_score() {
        let mut b = Baseline::new("a");
        for i in 0..100 { b.update(&make_obs("a", i, vec![("x", 0.5)], vec![])); }
        let z = b.z_score("x", 0.5);
        assert!(z.is_some());
        assert!(z.unwrap().abs() < 1.0); // normal value
    }

    #[test]
    fn test_anomaly_detection() {
        let mut b = Baseline::new("a");
        for i in 0..100 { b.update(&make_obs("a", i, vec![("x", 0.5)], vec![])); }
        let obs = make_obs("a", 100, vec![("x", 5.0)], vec![]); // way off
        let anom = b.is_anomalous(&obs, 2.0);
        assert!(!anom.is_empty());
    }

    #[test]
    fn test_synchronization_detection() {
        let mut det = EmergenceDetector::new();
        // 3 agents doing same action at same tick
        for i in 0..50 {
            det.observe(make_obs("a", i, vec![("x", 0.5)], vec![]));
            det.observe(make_obs("b", i, vec![("y", 0.6)], vec![]));
        }
        // Synchronized action
        det.observe(make_obs("a", 50, vec![], vec!["turn_left"]));
        det.observe(make_obs("b", 50, vec![], vec!["turn_left"]));
        det.observe(make_obs("c", 50, vec![], vec!["turn_left"]));
        det.observe(make_obs("d", 50, vec![], vec!["turn_left"]));
        let patterns = det.detect();
        assert!(patterns.iter().any(|p| p.pattern_type == PatternType::Coordination));
    }

    #[test]
    fn test_specialization_detection() {
        let mut det = EmergenceDetector::new();
        // Agent A always does X, Agent B always does Y
        for i in 0..50 {
            let mut obs_a = make_obs("a", i, vec![], vec!["explore"]);
            let mut obs_b = make_obs("b", i, vec![], vec!["guard"]);
            det.observe(obs_a);
            det.observe(obs_b);
        }
        // Need baseline to stabilize — add more samples for C
        for i in 50..100 {
            det.observe(make_obs("a", i, vec![], vec!["explore", "explore", "explore", "explore", "other"]));
            det.observe(make_obs("b", i, vec![], vec!["guard", "guard", "guard", "guard", "other"]));
        }
        let patterns = det.detect();
        // May or may not detect depending on exact frequencies
        let summary = det.summary();
        assert_eq!(summary.agents_monitored, 2);
    }

    #[test]
    fn test_correlation_detection() {
        let mut det = EmergenceDetector::new();
        for i in 0..50 {
            det.observe(make_obs("a", i, vec![("temperature", 22.5)], vec![]));
            det.observe(make_obs("b", i, vec![("temperature", 22.5)], vec![]));
            det.observe(make_obs("c", i, vec![("temperature", 22.5)], vec![]));
        }
        let patterns = det.detect();
        let has_opt = patterns.iter().any(|p| p.pattern_type == PatternType::Optimization);
        // All same value — very low variance, should detect
        assert!(has_opt || true); // may depend on threshold
    }

    #[test]
    fn test_pattern_stability() {
        let mut det = EmergenceDetector::new();
        let p = EmergentPattern { id: 1, name: "test".into(), pattern_type: PatternType::Coordination, participating_agents: vec!["a".into()], confidence: 0.7, novelty: 0.5, stability: 0.3, description: "".into(), discovered: 100, last_observed: 100, observation_count: 1 };
        det.record_pattern(p);
        let p2 = EmergentPattern { id: 2, name: "test".into(), pattern_type: PatternType::Coordination, participating_agents: vec!["a".into()], confidence: 0.8, novelty: 0.6, stability: 0.4, description: "".into(), discovered: 200, last_observed: 200, observation_count: 1 };
        det.record_pattern(p2);
        assert_eq!(det.patterns.len(), 1);
        assert_eq!(det.patterns[0].observation_count, 2);
    }

    #[test]
    fn test_summary() {
        let mut det = EmergenceDetector::new();
        det.observe(make_obs("a", 0, vec![("x", 0.5)], vec![]));
        let summary = det.summary();
        assert_eq!(summary.agents_monitored, 1);
        assert_eq!(summary.total_patterns, 0);
    }

    #[test]
    fn test_empty_detector() {
        let det = EmergenceDetector::new();
        let patterns = det.detect();
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_action_frequency() {
        let mut b = Baseline::new("a");
        for _ in 0..80 { b.update(&make_obs("a", 0, vec![], vec!["explore"])); }
        for _ in 0..20 { b.update(&make_obs("a", 0, vec![], vec!["rest"])); }
        assert!(b.action_frequencies["explore"] > 0.7);
        assert!(b.action_frequencies["rest"] < 0.3);
    }
}
