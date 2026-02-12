# Indicator Package Architecture

Phase-2 cleanup reorganized the `indicator/` package into focused subpackages while keeping backward compatibility.

## Structure

- `indicator/engines/`
  - Core analysis engines and reusable domain modules.
- `indicator/integrations/`
  - Integration snippets showing how to wire engines into pipelines.
- `indicator/examples/`
  - Executable demos and usage examples.
- `indicator/apps/`
  - User-facing entrypoints (`analyze`, `runner`, `continuous_runner`).
- `indicator/continuous/`
  - Continuous analysis framework components.
- `indicator/display/`
  - Presentation and terminal formatting helpers.
- `indicator/tests/`
  - Unit tests for indicator modules.
- `indicator/docs/`
  - Package documentation moved from root.

## Backward Compatibility

Root-level module names are retained as shims (for example `indicator/indicators.py`, `indicator/runner.py`).
They re-export the new subpackage modules so existing imports and scripts continue to work.
The package entrypoint `indicator/__init__.py` now resolves exports lazily via `__getattr__`,
so `import indicator` does not eagerly import optional network dependencies.

## Placement Rules

1. Add reusable analytics logic to `indicator/engines/`.
2. Add integration glue and snippets to `indicator/integrations/`.
3. Add executable sample code to `indicator/examples/`.
4. Add CLI/app entrypoints to `indicator/apps/`.
5. Keep root-level `indicator/*.py` files as compatibility shims only.
