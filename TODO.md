# Optimization Copilot — Remaining TODO

## Frontend Enhancements (Not Yet Implemented)

### Batch 38+ Feature Ideas (Researched but Not Built)
- [ ] **Proxy Measurement Decision Tree** (Overview) — When direct KPI measurement is slow/expensive, show which proxy metrics correlate best
- [ ] **Warm-Start Knowledge Gap** (Explore) — Assess how well transfer learning prior covers current search space
- [ ] **Acquisition Function Diagnostics** (Suggestions) — Visualize the acquisition function landscape to explain why suggestions were chosen
- [ ] **Overfitting-to-Noise Timeline** (History) — Track whether surrogate model is fitting noise vs. signal over time
- [ ] **Pareto Front Animation** (Overview) — Animated evolution of the Pareto front over iterations for multi-objective campaigns
- [ ] **Feature Engineering Suggestions** (Explore) — Recommend parameter transformations (log, polynomial) based on response surface shape
- [ ] **What-If Scenario Builder** (Suggestions) — Interactive slider-based prediction of outcomes
- [ ] **Campaign Comparison View** (History) — Side-by-side comparison of multiple campaigns

### UX Polish
- [ ] Mobile responsive layout for tablet use in labs
- [ ] Keyboard shortcuts guide/cheat sheet overlay
- [ ] Export all visualizations as PNG/SVG
- [ ] Print-friendly campaign report generation
- [ ] Accessibility audit (WCAG 2.1 AA)
- [ ] Loading skeleton states for all visualization cards
- [ ] Error boundaries per-visualization (graceful degradation)

### Performance
- [ ] Virtualize long lists (History tab with 1000+ trials)
- [ ] Memoize expensive computations (k-NN, clustering, correlation)
- [ ] Web Worker for heavy statistical computations
- [ ] Lazy-load tab content (only compute active tab)

## Backend Enhancements

### API
- [ ] WebSocket real-time updates (replace polling)
- [ ] Campaign export to JSON/CSV/PDF
- [ ] User authentication and team sharing
- [ ] Campaign archival and deletion
- [ ] Batch experiment submission endpoint

### Intelligence
- [ ] LLM-powered experiment rationale generation (Claude API integration)
- [ ] Automated campaign report generation
- [ ] Smart algorithm auto-selection based on problem fingerprint
- [ ] Natural language constraint expression

### Infrastructure
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Database persistence (currently in-memory)
- [ ] Production deployment guide

## Testing
- [ ] E2E test suite for all 4 workspace tabs
- [ ] Visual regression tests for micro-visualizations
- [ ] Performance benchmarks for large campaigns (1000+ trials)
- [ ] Cross-browser testing (Firefox, Safari)

## Documentation
- [x] Demo script (DEMO_SCRIPT.md)
- [x] Project statement (PROJECT_STATEMENT.md)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] User guide with screenshots
- [ ] Developer setup guide
- [ ] Architecture decision records (ADRs)
