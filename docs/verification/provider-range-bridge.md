# Current/future provider compatibility bridge [patch]

Initial 0.3.1 bridge: widen only Eunomia>=0.10,<0.12 and Leto/LetoOps>=0.46,<0.48; retain current lower bounds,
canonical Git sources,feature selections and every Rust implementation/test.
No adapter,compatibility alias,path patch,branch/rev pin,CI change or gate/limit
reduction is proposed. Athena's regenerated lock also replaces inherited
Moirai0.9 pins with current0.10 defaults. Current normal locks select merged
Aequitas0.3.1 and Eunomia0.10; unreleased providers are not selected in them.

All four workspace packages reserve0.3.1; their internal requirements advance
with them so normalized packages cannot silently select old core/backend
artifacts. Publish core/backend/facade in dependency order only after applicable
normal gates and review. Existing current Leto0.46 remains admitted, while the
reviewed additive0.47 line can be consumed after coordinator integration.

## Evidence and regression controls

Three exact-baseline requirement probes reject Aequitas0.4 or Eunomia0.11 before
compilation. The widened exports pass real external typed-time/scalar-policy
consumers in four cases:current0.3.1/0.10,next-law0.4/0.10,next-scalar0.3.1/0.11,
and a probe-only combined0.4/0.11 range composition. Both value oracles execute
for primitive and wrapped single/double precision. Time advance,duration,
scaling,adaptive rejection and exact convergence thresholds are checked from
binary-exact inputs. These are computation and nominal-identity checks,not
successful-resolution assertions alone.

Future scalar source is independently reviewed Eunomia22daa6c67e3d0f16d346a312b6835ccbf351a708.
Future law source is independently reviewed Aequitas a6cb96fcc72fb312ac3d17341812d49dde015b28,
whose production is identical to cleared9a015229. The combined probe applies
only already-reviewed provider-range admission to that export; the final
canonical Aequitas bridge-first rebase still needs its own exact-head review.
Every path rewrite exists only in disposable probe exports outside this repo.
These probes do not establish publication,canonical final lock availability,
full backend future-cohort readiness or complete Python interoperability.

Current repository gates pass:136 native cases,five existing ignored allocation cases;three doctests. The configured allocation-only lane separately passes five cases with one pre-existing unselected case. Strict all-target/all-feature Clippy,
no-default targets,warning-denied Rustdoc,format and actual Rust1.95 all-target/
all-feature compilation pass. Unchanged supply-chain policy passes. Complete
actual locked baseline/head Rustdoc JSON passes223 explicit patch checks per
published package,with30 tool-defined skips; tools alone do not prove external
nominal dependency compatibility. No runtime compiler-floor bypass is used.

Athena's required accelerator tests initially fail to acquire Metal inside the
sandbox; the same unchanged suite passes with actual device access. No optional
device flag or disabled test is used to convert that failure to success.
Normal source-package verification and final hosted/independent review receipts
are recorded separately rather than inferred from local source tests.

## Coordination and rollout

The user approves the normal Eunomia0.11/Aequitas0.4/binding0.2/NumPy-PyO3.30
plan and ADR0020,bridge first,with mandatory tests/reviews. Eunomia owner01a13b11
retains Mnemosyne/Hermes bridges; coordinator01a14490 retains Leto/Hephaestus/
Apollo/Coeus. This change does not edit those branches. Full Athena future
backend coherence depends on their canonical bridge admissions; the isolated
core/time probes are not substituted for that final consumer gate.

Land backwards-compatible admissions on safe current providers before advancing
breaking provider defaults. Keep the combined NumPy0.30 change in still-unreleased
Eunomia0.11; if0.11 publishes first,a later breaking0.12 needs new admission beyond
this<0.12 ceiling. Rebase the law0.4/binding0.2 candidate after bridge140 and
regenerate/review its coherent final lock. No old-version pin or permanent
workaround replaces that rollout. This PR carries no merge/publication action.
Harmonia's separate numerical performance acceptance hold remains explicit.

## Current 0.3.2 canonical repair graph

Leto/LetoOps now require >=0.46.3,<0.48 and select merged/public 0.46.3 from
`5051cd9577650a5712aa6a7effcd8955c9d8bcc7`. Registry source, native tags and
fresh external consumers are verified. Hephaestus admits >=0.22.2,<0.25 and
selects current 0.24 default. All provider sources remain canonical; no product
path patch, revision pin, downgrade or Rust/feature/policy edit occurs. This
supersedes the initial bridge's admission of defective 0.46.2. The affected
Athena graph still needs independent review and lifecycle checks; it does not
inherit whole-cohort/absolute performance acceptance from native Leto review.
The future Eunomia/Aequitas/FFI cohort and all existing holds remain separate.
