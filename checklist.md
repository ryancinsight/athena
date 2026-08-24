# Checklist — athena

Owner-keyed execution steps. Edit only your own section.

## gap-audit-2026-08-20 (owner: atlas-gap-audit)

Static evidence-only audit of `f6608be` (detached HEAD, four peer-modified
files left untouched). No build, test, or lint command was run — every finding
below is grounded in a file path or exact command output.

- [x] Orient: `git log --oneline -8`, `git status -sb`, `git status
      --porcelain`. HEAD `f6608be`, detached, four modified files
      (`.github/workflows/book-pages.yml`, two `athena-leto` book examples,
      `docs/book/book.toml`) carrying an in-flight `mdbook test` wiring change.
- [x] Read declared scope: `README.md`, `CHANGELOG.md` Unreleased,
      `docs/adr/README.md` and all four ADRs, `docs/book/SUMMARY.md`,
      workspace `Cargo.toml`.
- [x] Measure: 4 packages; 4938 src LOC; 2622 test/example LOC; 63 `#[test]`
      functions; 10 book pages; 4 ADRs; 0 `todo!(`; 0 `unimplemented!(`; 0
      TODO/FIXME/HACK; 0 `#[allow(`; 5 `#[expect(`; 0 production `unwrap()`;
      1 file over 500 lines; 0 crates missing `#![deny(missing_docs)]`;
      8 `dyn` sites, all annotated example-binary error aggregation;
      2 `pub use ... as`, both namespace re-exports (`cpu`, `accelerator`),
      neither a compatibility shim; 0 benches; 0 property tests.
- [x] Question (a) solver coverage: declared set is CG, restarted GMRES,
      BiCGSTAB, LSQR plus Jacobi, incomplete-LU, and SOR preconditioning —
      every declared item is implemented with no stub. AMG, block, and
      polynomial preconditioners and MINRES/CGS/QMR/IDR are neither declared
      nor present. The real coverage gap is per-backend, not per-method:
      no preconditioner exists for the Hephaestus backend (AT-004).
- [x] Question (b) termination: solvers report terminal values on a `#[must_use]`
      `SolveReport`, not a bare cap — `Converged`, `InitialResidual`,
      `MaxIterations`, `Breakdown`, `NonPositiveCurvature`, `NonFinite`,
      `NormalEquations`, `Stagnated`, `Diverged`. The stagnation and divergence
      criteria are derived against `residual_noise_floor` = `sqrt(len) * EPS *
      ||b||` with its derivation published, but only GMRES detects them
      (AT-003). A typed error carrying residual history was considered and
      rejected in ADR 0004 with a recorded rationale; history is supplied by
      the `IterationObserver` seam without implicit allocation.
- [x] Question (c) backend parity: the Hephaestus backend implements the full
      `KrylovBackend` seam and runs CG, GMRES, and BiCGSTAB; LSQR is CPU-only
      because `RectangularOperator` has no accelerator implementation, which
      README declares. Preconditioning on the accelerator is `Identity` only
      (AT-004), and there is no CPU-versus-accelerator differential test —
      the device assertions use a hardcoded `1e-3` (AT-002) and the whole lane
      silently skips without an adapter (AT-001).
- [x] Question (d) convergence verification: manufactured solutions are
      recovered on 2-3 variable systems, generically over `f32` and `f64`, with
      value-semantic assertions; there is no convergence-rate, order, or
      finite-termination verification and no property testing (AT-005).
- [x] Cross-check README and Accepted-ADR claims against code. One claim
      contradicted: the accelerator-lane CI sentence. Fixed in `README.md` —
      no source reads the `CI` variable, so acquisition failure returns from
      every case identically on CI and locally.
- [x] Create `backlog.md` with 13 DoR-shaped items, priority-ordered.
- [x] Create this checklist.

### Handoff — recommended execution order

- [ ] AT-006 then AT-007 — one file, `.github/workflows/ci.yml`; the toolchain
      mismatch invalidates the compiler identity behind every green run, so it
      precedes any item whose acceptance is a gate result.
- [ ] AT-001 — make the accelerator lane's absence visible before treating any
      device result as evidence.
- [ ] AT-002 — CPU-versus-accelerator differential suite with derived bounds;
      depends on AT-001.
- [ ] AT-005 — convergence-property and property-test layer; supplies the
      parameterized SPD family AT-011 also needs.
- [ ] AT-003 — per-family stagnation and divergence derivations, recorded as an
      ADR 0004 revision.
- [ ] AT-004 — device Jacobi preconditioner; confirm the Hephaestus kernel
      exists before claiming, and treat a gap as upstream work.
- [ ] AT-009, AT-010, AT-008, AT-012, AT-013, AT-011 — conformance,
      documentation, and instrument items, independently claimable.

### Constraints observed

- No Rust source, `Cargo.toml`, or CI file was modified.
- No `git` mutation of any kind was run; the tree remains detached with the
  same four peer-modified files, plus this audit's `README.md`, `backlog.md`,
  and `checklist.md`.
