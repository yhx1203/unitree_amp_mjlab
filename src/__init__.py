from pathlib import Path
from types import SimpleNamespace


SRC_PATH: Path = Path(__file__).parent


def _normalize_warp_driver_version(version) -> tuple[int, int] | None:
  """Return a MuJoCo-Lab compatible CUDA driver version tuple."""
  if version is None:
    return None
  if isinstance(version, tuple):
    return tuple(version[:2])
  if isinstance(version, str):
    parts = version.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
      return (int(parts[0]), int(parts[1]))
    return None
  if isinstance(version, int):
    # cuDriverGetVersion style, e.g. 12090 -> (12, 9), 13000 -> (13, 0).
    return (version // 1000, (version % 1000) // 10)
  return None


def _patch_warp_context_runtime() -> None:
  """Compat shim for mjlab 1.2.0 with newer warp-lang.

  mjlab 1.2.0 reads ``wp.context.runtime.driver_version``. Newer Warp exposes
  the same information through ``wp.get_cuda_driver_version()`` instead.
  """
  try:
    import warp as wp
  except ModuleNotFoundError:
    return

  if hasattr(wp, "context"):
    return
  if not hasattr(wp, "get_cuda_driver_version"):
    return

  class _RuntimeCompat:
    @property
    def driver_version(self):
      return _normalize_warp_driver_version(wp.get_cuda_driver_version())

  wp.context = SimpleNamespace(runtime=_RuntimeCompat())  # type: ignore[attr-defined]


_patch_warp_context_runtime()
