# Tutorial refresh and boundary correction: acceptance evidence

Current acceptance is owned by [Contract 101 §12](../../contracts/101-docstrings-and-executable-examples.md#12-tutorial-refresh-checkpoint).
This record covers the refreshed notebooks 01–14 and the four approved library
corrections. Notebook 15, its fixtures, and its tests remain unchanged. No public
signature, dependency, benchmark implementation, or architecture guard was added.

## Latest tutorial extension

The 2026-09-25 extension adds explicit trial joins and generic/typed component
comparison to 08, a ragged frame round trip to 11, and lazy Position field assembly,
norm, and reduction to 12. Their saved outputs were refreshed after six fresh
kernel runs across the two supported working directories; browser review checked
the displays and schema controls. Notebooks 14 and 15 remain unchanged.

The alignment example exposed a stale parameter declaration after xarray dropped
an eager auxiliary. The existing spatial finalizer now reconciles optional names
through the core owner; eager merge and deferred equality policies remain intact.
The final full suite passed **6,491 tests**; the affected physical no-Numba matrix
passed **792 with 8 expected skips**. Strict Sphinx and static checks passed.
Full commands, content identity, environments, warnings, scope, and authoritative
acceptance are recorded in Contract 101 §12 above. The records below are historical.

## Resolved boundaries

| Boundary | Shared correction | Verified tutorial |
| --- | --- | --- |
| Indexed sequence packing | Segment-local positions before stacking/gathering; original sequence labels cannot realign packed payload. Batch joins remain label-based. Empty Dask results retain topology. | 06 rejoins indexed slices. |
| Query/event metadata | Consume inherited provenance-marked auxiliaries before namespace checks; preserve caller coordinates and deferred payload validation. Event packing shares one coordinate carrier without eager equality comparisons. | 05 reuses the boundary result's timestamp directly. |
| Spatial padding | Shared spatial preparation honors structural and operation-owned validity, neutralizes unreachable rows, and masks numerical outputs. Binary output prefix lengths are reconciled; coordinate-only dimensions do not expand constant rotations. Reached invalid quaternions still fail. | 11 expresses ragged observations through native-rate providers. |
| Generic component extraction | Functional/accessor extraction returns base AOs through existing schema, metadata-isolation, and resource owners. Domain decomposition retains typed components and graph behavior. | 13 plots a registered Pose component directly. |

The four original strict-xfail reproductions are now passing regressions.
Class-level cases cover supported eager/Dask execution, empty/ragged inputs,
multiple batch/query dimensions, ordering, metadata, both validation modes,
source immutability, and resource lifetime. Existing suites retain namespace
collision, native-index, deferred-validation, path-direction, optional-backend,
and physical-support coverage. Contract 008 maps the durable test IDs.

## Historical native-index and query-consumption correction

Structural and parameter masks now share positional construction through the
existing index owner, preserving complete RangeIndex/transform-backed groups.
Semantic broadcasting expands by shape and attaches each reference's index
container once. Rotation/Pose conversion, validation, reduction, and paired
assembly keep aligned payload variables separate from coordinate attachment.
Paired assembly replaces only the auxiliary carriers changed by deferred
coordinate validation. Component-only extraction preserves declared coverage
without assuming labels exist; unknown unindexed slicing still prunes validity.

Every parameter index consumption branch protects dimensions and native index
groups. Indexed prior-result sizes are not silently removed. Compatible source
size indexes survive preparation/finalization, while reserved source/query
indexes fail under the public parameter owner before numerical work.

Checkpoint source/test/support SHA-256 (52 Python files on base `25ffd2a`):
`6f392c97c67b1b4186dc023cb6bee0304de334819c2928c85ef2cc9d5e5931d8`.
That checkpoint passed:

- Maintained `tal`: `python -m pytest -q`, **6,335 passed, 47 warnings**.
  Warning categories match the historical inventory below. Final focused
  parameter/native/architecture, assembly, and broadcasting selections passed
  **779**, **385**, and **321** cases respectively before aggregate verification.
- Physical no-Numba `test`: **1,588 passed, 8 Numba-only skips**, selecting
  `test_native_index_numerical_topology.py`, query metadata, positional concat,
  component subtypes, spatial numerical validity/topology/alignment/lazy
  coordinates, parameter query/operation and AO structural tests, event windows,
  spatial path/Rotation/Pose/apply, and tutorial-boundary regressions. Both
  interpreters import this checkout; dependency versions remain those below.
- Eight fresh-kernel notebook executions: 05/08/11/14 from repository root and
  tutorial directory. All passed without cell stderr. Outputs remain local;
  notebook source/output files were not rewritten during this correction.
- Strict Sphinx `sphinx-build -E -q -n -W -b html docs docs/_build/html`,
  changed-file Ruff without rewriting, compilation, existing structural/import
  and ownership tests, shared budgets (**600/50/10/2**, 32 production files),
  Markdown links/fences, and staged/unstaged diff checks passed. A standalone
  `lint-imports` attempt found no root configuration; existing repository import
  tests are the executed boundary checks, not that unconfigured command.
- The obsolete parameter-mask direct-call requirement in `ORCH_ARCH_013` was
  removed without expanding its AST coverage. Shared value validation remains
  exercised by parameter runtime tests, and native-index preservation has
  public regressions. No new or accepted-phase guard remains from this work.
- Existing affected performance and isolated RSS verification completed.
  The initial graph-build failure remains in private artifacts; code changes
  removed repeated coordinate merges before the final measurements passed.
  No threshold, benchmark fixture, or measurement protocol changed. Detailed
  measurements and profiles remain outside the repository.
- Staged diff SHA-256 remains
  `72a54f6cd40c939fb7c68994f715e8d987b6ec6c908bf4ac52304375ab581f6e`;
  all 29 protected notebook-15 files equal their index content. Nothing was
  staged or committed.

The prior separate RangeIndex semantic-broadcast finding below is resolved by
this correction. Current acceptance is authoritative only in Contract 101 §12.

## Historical validity namespace and partial-mask follow-up

The two reviewed defects are corrected in the existing spatial numerical-validity
owner. Generated metadata preflight protects surviving dimensions and complete
index groups. Partial masks expand by operand shape and reuse its coordinate/index
container before the existing alignment owner runs. No numerical buffer is copied
or materialized to establish topology. Pose and kinematics remain thin consumers.

Contract 008 IDs `SPATIAL_VALIDITY_ALIGNMENT_001`–`006` add 90 public runtime cases:
size and temporal-mask namespace collisions, one/two batch axes, parameter and
inner/outer batch joins, differently labeled batches, empty/nonempty eager/Dask,
both validation modes, adjacent Pose/vector delegates, established native indexes,
source immutability, and independent SciPy expectations. Explicit parameter-key
alignment uses eager index labels with lazy payload/masks; its existing pandas
index-construction boundary materializes lazy labels. Batch-key coverage retains
fully lazy query chains. This correction adds no new materialization boundary.

Final source/test/support SHA-256 (40 Python files):
`b5f802c14f6add7db24f2d540bdb0188916fe28629602822da9882819ae012f8`, on base
`25ffd2a75c6ea14913d029c43d5423024c5cb3da`, using the convention below.
Verification used the unchanged environment versions listed below:

- Focused spatial/architecture/unchanged robot-capstone matrix: **699 passed**,
  one IPython configuration warning. Command: maintained `tal` Python
  `-m pytest tests/core/test_spatial_validity_alignment.py` plus the topology,
  lazy-coordinate, numerical-validity, Pose, apply, robot-capstone, and spatial
  architecture files, `-q --tb=short`.
- Full maintained `tal` `python -m pytest -q --tb=short`: **6,168 passed,
  47 warnings**, no failures/skips/xfails. Warning categories match the historical
  inventory below. Final private-helper docstring clarification changes no
  executable behavior; final source compilation/Ruff/budgets also passed.
- Physical no-Numba `test`: **1,282 passed, 8 skips**, selecting the earlier
  affected matrix below plus `test_spatial_validity_alignment.py`. Imports resolve
  to this checkout; Numba absence was checked before closeout.
- Notebooks 08, 11, and 14: six fresh-kernel executions across repository root
  and tutorial directory, all passed without cell stderr. Outputs remain local;
  unchanged figures and controls retain the earlier visual evidence.
- Strict Sphinx `-E -q -n -W -b html docs docs/_build/html`, Ruff without
  rewriting, compilation, existing architecture/import checks, Markdown links and
  fences, and both diff checks passed. Shared production budget maxima remain
  **599 / 50 / 9 / 2** across 22 files. No phase guard was added.
- Staged diff SHA-256 remains
  `72a54f6cd40c939fb7c68994f715e8d987b6ec6c908bf4ac52304375ab581f6e`;
  all 29 protected notebook-15 files equal their index content. No files were
  staged or committed.
- Affected performance verification is complete, using
  `NUMBA_NUM_THREADS=8 python -m benchmarks.bench_batched_fused_path_reference
  --check-134c` and the existing `--rss-public` runner. Detailed samples, gates,
  and memory evidence remain outside the repository. No threshold or benchmark
  implementation changed.

### Historical core topology finding (resolved by the native-index correction)

**P2: native RangeIndex semantic broadcast fails before spatial validity.** A
batched Rotation with a native `RangeIndex` and an unbatched Rotation, neither
with validity metadata, fail composition when the core realization owner adds
the missing batch axis. Passing the reference coordinate to xarray `expand_dims`
constructs a `PandasIndex`; exact alignment then rejects it against the original
`RangeIndex`. The owning function is
`tal/core/orchestration/topology.py::_expand_operand_semantic_dims`, unchanged
from HEAD. Existing-topology RangeIndex mask expansion passes the new regression.

Contracts 077/079/091 own semantic topology realization. This is a supported
boundary defect, not a proposed support expansion or a reason to weaken native
index guarantees. It was separate from the earlier reviewed spatial changes. The subsequent
native-index correction includes it through the same index-topology owner:
expand by shape, attach complete native index groups, and verify cross-domain
broadcast behavior (including eager/lazy, indexed/unindexed, mismatched labels,
source immutability, and zero coordinate-transform/payload execution).

## Historical content and environments

The final correction was verified on base revision
`25ffd2a75c6ea14913d029c43d5423024c5cb3da`. Its 39 changed Python
source/test/support files have SHA-256
`eb2b7f99980cfae81d5dbeb6630b0cbffcf3cf06f53a19b3564c92ade31e187e`.
Hash input is sorted repository-relative path, NUL, file bytes, NUL for each
unstaged changed or untracked nonignored Python file. The ordered manifest and
verification logs are local artifacts. No notebook was rewritten by this
correction, and the environment versions below remain unchanged.

The correction resolves the review findings around resolved batch validity,
either-operand temporal provenance, and inherited query metadata collisions,
plus the user-authorized lazy-coordinate assembly issue. Adjacent Pose
conversion and inverse paths now preserve the same validity provenance.
Unchanged eager size coordinates retain their source metadata; changed counts
receive generated-coordinate provenance. This preserves the unchanged robot
capstone's coordinate contract.

Final checks:

- Maintained `tal`: focused public regressions, Pose/apply coverage, unchanged
  robot-capstone tests, and spatial architecture checks: **686 passed**, one
  IPython configuration warning. Full `python -m pytest -q --tb=short`:
  **6,078 passed**, **47 warnings**, no failures/skips/xfails.
- Physical no-Numba `test`: **1,192 passed, 8 skipped**. The command selects
  `test_concat_positional_segments.py`, `test_query_metadata_chaining.py`,
  `test_component_extraction_subtypes.py`, `test_spatial_numerical_validity.py`,
  `test_spatial_validity_topology.py`, `test_spatial_lazy_coordinate_assembly.py`,
  `test_param_query_coordinates.py`, `test_event_around.py`,
  `test_spatial_path_solve.py`, `test_spatial_rotation.py`,
  `test_spatial_pose.py`, `test_spatial_apply.py`, and
  `test_tutorial_boundary_regressions.py`. Eight tests requiring Numba skip;
  physical absence and checkout imports were verified.
- Fresh maintained-`tal` kernels passed notebooks **05, 08, 11, and 14** from
  both supported directories (eight executions), without cell stderr. Outputs
  were kept locally. The previous full notebook execution and visual review
  below remain historical evidence for unchanged content.
- `sphinx-build -E -q -n -W -b html docs docs/_build/html` passed without
  warnings. Changed-file Ruff passed without rewriting; 39 changed Python
  files and notebook 01–14 code compiled. All 22 changed production files
  satisfy the shared budgets: maxima **599 / 50 / 9 / 2** for file lines,
  function lines, parameters, and control depth. Existing import/ownership
  checks passed in the full suite. Local Markdown links/fences and both diff
  checks passed.
- Affected performance verification used the maintained frozen reference
  runner, including the checked Dask route, executor/eager comparisons, graph
  growth, and isolated RSS protocol. Measurements and detailed results remain
  **outside the repository**. No thresholds or benchmark implementation changed.
- Temporary-test review retired `ARCH_SPATIAL_039`'s private import/call
  placement and removed the obsolete call-site assertion from
  `SPATIAL_ARCH_192`, retaining its numerical-owner boundary. Public Pose
  conversion/inverse/compose and numerical-validity regressions retain the
  behavior. No new architecture guard or phase marker was added.

An intermediate aggregate run exposed the unchanged-size metadata regression
and obsolete placement assertions. Those failures were corrected before the
final full run; they are not excluded from the acceptance result. Warning
categories match the historical inventory below. Remaining exclusions are
unchanged; deferred shared-coordinate mismatches fail upon materialization
through either their numerical payload or coordinate.

The staged diff hash below still matches, and all 29 protected notebook-15
files remain byte-identical to the index. New edits remain unstaged.

## Historical verification before the follow-up review

Base revision: `25ffd2a`. SHA-256 of changed Python sources/tests, notebook 01–14
code cells, and flight CSV fixtures:
`974a1d920951d019310a6c6d21678e9cc7ef82a9d28857ccc47e0cba7552d54b`.
The hash excludes notebook outputs and this acceptance record. The private
verification manifest records its ordered file list.

The staged binary diff remains byte-identical:
`72a54f6cd40c939fb7c68994f715e8d987b6ec6c908bf4ac52304375ab581f6e`.
All 29 protected notebook-15/robot-catching files retain their initial hashes.
New changes remain unstaged; no commit was made.

- Maintained `tal`: Python 3.13.9, NumPy 2.3.1, SciPy 1.16.0, pandas 2.2.3,
  xarray 2025.10.1, Dask 2025.5.1, Zarr 3.1.5, Numba 0.65.1,
  HoloViews 1.22.1, hvPlot 0.12.2, Bokeh 3.8.2, nbclient 0.10.2.
- Physical no-Numba `test`: Python 3.14.4, NumPy 2.4.4, SciPy 1.17.1,
  pandas 3.0.2, xarray 2026.4.0, Dask 2026.3.0. Numba is absent on disk.
- Imports and kernels resolve to this checkout using the repository working
  directory and explicit `PYTHONPATH`; maintained editable installations were
  not changed.

## Historical executed verification

- `tal` full repository `python -m pytest -q --tb=short`: **5,881 passed**, no
  failures/skips/xfails, **47 warnings**. This includes architecture, core,
  optional-domain, executable-documentation, and notebook-code coverage.
- `test` affected matrix: **628 passed**. The command selects
  `test_concat_positional_segments.py`, `test_query_metadata_chaining.py`,
  `test_component_extraction_subtypes.py`, `test_spatial_numerical_validity.py`,
  `test_param_query_coordinates.py`, `test_event_around.py`,
  `test_spatial_path_solve.py`, and `test_tutorial_boundary_regressions.py`.
- Notebooks **01–14 each executed in a fresh maintained-`tal` kernel from both
  repository root and `examples/tutorial`**. Final tutorial-directory outputs
  are saved. Notebook assertions and independent test references verify the
  corridor counts `[4, 5, 7]`, all 12 recordings, two unobserved crossings,
  analytic interpolation/calculus/transforms, lazy persistence, and cleanup.
- Browser review verified 05's event-aligned plot, 06's concat metadata,
  11's ragged typed displays, 13's registered-component plot, and 14's tables
  and corridor geometry. A local live Panel session executed notebook 13's
  actual code: selecting `flight_11` displayed its truncated recording;
  changing the independent explorer from line to scatter updated the plot.
  Static exports still require a live Python process for callback controls.
- Changed-file Ruff passed without rewriting. Changed Python and all notebook
  code compiled. Shared budgets passed for all 14 changed production files:
  maxima **597 file lines / 50 function lines / 9 parameters / depth 2**.
  Existing import/ownership/architecture checks passed in the full suite.
- `sphinx-build -E -q -n -W -b html docs docs/_build/html` passed. Local Markdown
  links/fences, fixture regeneration, and staged/unstaged diff checks passed.
  Sphinx checks its source tree, not every contract outside that tree.
- Existing affected spatial benchmarks were executed with unchanged fixtures,
  repetition protocols, and thresholds, including isolated eager RSS. Benchmark
  measurements, samples, resource artifacts, and detailed gate results are
  retained **outside the repository**. The frozen reference environment remains
  authoritative; physical no-Numba timing is supplementary.
- The obsolete extraction private-call-order guard was retired. Durable
  output-name/registry tests replace it. No new phase guard or AST interpreter
  was added; no resolved xfail marker remains in the four reproductions.

## Warnings and supported boundaries

Observed test warnings come from xarray merge-default changes, pyproj/NumPy
conversion deprecations, Zarr format-3 dtype/consolidation and rechunk notices,
and IPython's fallback from an unwritable home configuration directory.
Notebook 12 emits the Zarr consolidated-metadata portability notice. Local
nbconvert exports warn about HoloViews' empty initialization MIME bundle;
subsequent plots render. The live Bokeh session logged dropped-patch notices
while replacing plot models; the observed controls updated successfully.
No warning suppression was added.

An independent direct-xarray probe reproduces hvPlot 0.12.2 explorer's failure
when its defaults attempt numerical bounds over string component labels.
Generic component extraction is corrected; automatic plotting-field selection
inside hvPlot remains upstream-owned. Runtime tests verify explorer forwarding
with the existing backend double, and the notebook demonstrates real explorer
execution with explicit fields. No TAL-specific plotting workaround or type
branch was added.

Existing exclusions remain: non-prefix structural slices may clear prefix
validity, unsupported index types remain unsupported, and lazy condition-based
event discovery still requires the documented bounded/explicit-anchor policy.
The landing analysis concerns sampled geometric crossings of prescribed
trajectories, not continuous collision detection or physical landing safety.

## Historical tutorial-only checkpoint

Before these authorized library fixes, the tutorial refresh alone executed
notebooks 01–14 from both directories but left pytest unrun at the user's
request. Its source/fixture digest was
`79907da28da31b810053494241c7a268272917b4157a15059afdd7babcf7f1a9`.
Four supported-boundary findings prevented acceptance: indexed concat dropped
later values, inherited boundary metadata prevented window reuse, ragged
expression reached padding, and generic extraction attempted to rewrap an
incomplete Pose. Those findings are resolved above; earlier passing notebook
runs did not certify the failing routes.

Total nonblank/noncomment tutorial/support code (including unchanged notebook
15 and both generators): **2,799 before → 724 at that checkpoint** (620 notebook + 104 supporting
Python lines), approximately **74% smaller**. Test/reference code is separate
verification, not hidden notebook support. TAL construction, registration, and
analysis remain visible; the old `_helpers.py` stays removed.
