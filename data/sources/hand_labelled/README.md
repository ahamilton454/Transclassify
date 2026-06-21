# hand_labelled — the trusted keystone (git-tracked)

Small, hand-authored, real-string eval source with labels we trust: real distribution, our gold,
varying taxonomies, immune to training-data leakage. The set you'd ultimately *decide* on.

Today it's ~12 placeholder examples (heavy on the tail, one description-flip pair) — we use
`llm_generated` for now. **Grow this** with real, anonymized transactions you label by hand, weighted
toward the cryptic long tail. It's the only source kept git-legible (edit the `*.jsonl` directly).
