(api-concat)=
# Combine and Alignment

TAL combine operations assemble, align, merge, or reshape AOs while preserving
schema truthfulness. They are available as top-level core helpers and through
the `ao.combine` accessor.

## Batch and Sequence Combination

```python
from tal.core import BatchConcatOptions, SequenceConcatOptions, concat_batch, concat_sequence

batched = concat_batch([run_a, run_b], opts=BatchConcatOptions(batch_dim="run"))
longer = concat_sequence([segment_a, segment_b], opts=SequenceConcatOptions(overlap="error"))
```

Accessor form:

```python
batched = run_a.combine.concat_batch(run_b)
longer = segment_a.combine.concat_sequence(segment_b)
```

`concat_batch(...)` stacks compatible AOs along a batch axis.
`concat_sequence(...)` appends compatible AOs along declared sequence
semantics.

## Merge and Align

```python
from tal.core import AlignOptions, MergeOptions, align_many, merge

merged = merge([left, right], opts=MergeOptions(batch_join="inner"))
aligned_left, aligned_right = align_many([left, right], opts=AlignOptions(sequence_join="exact"))
```

Use `merge(...)` when variables or schema payloads should become one AO. Use
`align_many(...)` or `ao.combine.align(...)` when you want to inspect aligned
inputs before running numeric work.

## Core Assembly

Core assembly operations build or rewrite payload axes:

- `assemble_core(...)`
- `stack_core(...)`
- `block_core(...)`
- `concat_core(...)`
- `decompose_core(...)`
- `overlay_core(...)`

These are also used by `tal.linalg` wrappers when the payload should be treated
as vector or matrix structure.

## Validity and Fill Policy

Combine operations preserve declared validity where possible and update it when
the structure changes. Fill values, join policies, and overlap handling are
explicit option-object choices rather than hidden defaults.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/concat
   :nosignatures:

   tal.core.concat_batch
   tal.core.concat_sequence
   tal.core.merge
   tal.core.align_many
   tal.core.align_pair
   tal.core.assemble_core
   tal.core.stack_core
   tal.core.block_core
   tal.core.concat_core
   tal.core.decompose_core
   tal.core.overlay_core
   tal.core.BatchConcatOptions
   tal.core.SequenceConcatOptions
   tal.core.MergeOptions
   tal.core.AlignOptions
   tal.core.CoreConcatOptions
   tal.core.CoreDecomposeOptions
   tal.core.CoreOverlayOptions
   tal.core.combine_ops.accessor.CombineAccessor.concat_batch
   tal.core.combine_ops.accessor.CombineAccessor.concat_sequence
   tal.core.combine_ops.accessor.CombineAccessor.merge
   tal.core.combine_ops.accessor.CombineAccessor.align
```

## See Also

- {doc}`schema`
- {doc}`reducers`
- User guide: {doc}`../user-guide/numpy`
