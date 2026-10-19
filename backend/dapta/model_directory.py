"""Model/adapter directory for the Qwen worker.

This is deliberately small and honest about what it knows. The vLLM worker
(`vllm_backend.VllmRuntime`) loads exactly one base model per process; adapters
register and swap on top of that base model almost instantly, but switching the
base model itself means stopping and reloading the worker (minutes, not
milliseconds). Everything here is built around that one constraint:

- `BASE_MODELS` lists base models DAPTA knows about and is honest about which
  one is actually wired to a running worker today (`status="integrated"`) versus
  which are catalog entries DAPTA could point `QWEN_BASE_MODEL` at but has not
  provisioned, trained, or evaluated anything on (`status="catalog_only"`).
- `efficient_adapter` picks, among adapters that could serve a task, the one
  that avoids a base-model reload when possible, and is explicit rather than
  silent when a switch is unavoidable.

No benchmark numbers live here. Benchmark evidence for a specific adapter lives
on that adapter's own record (`metadata.benchmark_accuracy`, etc., set by
whoever actually ran the comparison) and is read from the job store, not
invented in this catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BaseModelSpec:
    model_id: str
    family: str
    params_b: float
    modality: str  # "text" or "text+image"
    approx_vram_gb_bf16: float  # rough weights-only estimate, not a measurement
    license: str
    status: str  # "integrated" | "catalog_only"
    note: str = ""


# Real, publicly listed Hugging Face model IDs. Only Qwen3-VL-8B-Instruct is
# actually wired to a running worker (QWEN_BASE_MODEL default in config.py /
# vllm_backend.py); the rest are real models DAPTA could switch to, not ones it
# has trained or measured anything on yet.
BASE_MODELS: tuple[BaseModelSpec, ...] = (
    BaseModelSpec(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        family="qwen3-vl",
        params_b=8.0,
        modality="text+image",
        approx_vram_gb_bf16=16.0,
        license="Apache-2.0",
        status="integrated",
        note="Default QWEN_BASE_MODEL. The only base model any worker actually serves today.",
    ),
    BaseModelSpec(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        family="qwen3-vl",
        params_b=2.0,
        modality="text+image",
        approx_vram_gb_bf16=4.0,
        license="Apache-2.0",
        status="catalog_only",
        note="Smaller/cheaper sibling of the integrated model; not provisioned on any worker yet.",
    ),
    BaseModelSpec(
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        family="qwen3-vl",
        params_b=30.0,
        modality="text+image",
        approx_vram_gb_bf16=60.0,
        license="Apache-2.0",
        status="catalog_only",
        note="Larger MoE sibling for harder tasks; needs more VRAM than the current GPU budget.",
    ),
    BaseModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        family="qwen2.5",
        params_b=7.0,
        modality="text",
        approx_vram_gb_bf16=14.0,
        license="Apache-2.0",
        status="catalog_only",
        note="Text-only fallback for skills with no image input, cheaper to serve than a VL model.",
    ),
)

BASE_MODELS_BY_ID: dict[str, BaseModelSpec] = {spec.model_id: spec for spec in BASE_MODELS}


@dataclass(frozen=True)
class AdapterSelection:
    adapter: Mapping[str, Any] | None
    requires_model_switch: bool
    switch_target_base_model: str | None
    reason: str


def efficient_adapter(
    candidates: Sequence[Mapping[str, Any]], loaded_base_model: str | None
) -> AdapterSelection:
    """Pick the cheapest candidate to actually run right now.

    `candidates` are adapter records (as returned by JobStore.get_adapter /
    list_adapters) that already match the requested task. This function does not
    rank by accuracy — that's the router's job upstream, using whatever evidence
    an adapter's own metadata carries. It only decides execution cost: an adapter
    on the currently-loaded base model is free (LoRA hot-swap); anything else
    needs an explicit, visible base-model reload.
    """
    if not candidates:
        return AdapterSelection(
            adapter=None,
            requires_model_switch=False,
            switch_target_base_model=None,
            reason="no candidate adapters",
        )

    if loaded_base_model:
        for adapter in candidates:
            if adapter.get("base_model") == loaded_base_model:
                return AdapterSelection(
                    adapter=adapter,
                    requires_model_switch=False,
                    switch_target_base_model=None,
                    reason=f"adapter already matches the loaded base model ({loaded_base_model})",
                )

    # Nothing matches what's loaded. Say so explicitly instead of picking one
    # and silently triggering a multi-minute reload.
    chosen = candidates[0]
    target = str(chosen.get("base_model") or "")
    return AdapterSelection(
        adapter=chosen,
        requires_model_switch=True,
        switch_target_base_model=target or None,
        reason=(
            f"no candidate runs on the loaded base model ({loaded_base_model or 'none loaded'}); "
            f"best candidate needs a reload onto {target or 'unknown'}"
        ),
    )


def directory_snapshot(adapters: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A read-only view combining the base model catalog with registered adapters.

    Used by a status endpoint or the ASI:One `list adapters` reply — not a
    source of truth; adapters still come from JobStore.
    """
    by_base_model: dict[str, list[str]] = {spec.model_id: [] for spec in BASE_MODELS}
    for adapter in adapters:
        base_model = str(adapter.get("base_model") or "")
        by_base_model.setdefault(base_model, []).append(str(adapter.get("adapter_id") or ""))

    return {
        "base_models": [
            {
                "model_id": spec.model_id,
                "family": spec.family,
                "params_b": spec.params_b,
                "modality": spec.modality,
                "approx_vram_gb_bf16": spec.approx_vram_gb_bf16,
                "license": spec.license,
                "status": spec.status,
                "note": spec.note,
                "adapter_ids": by_base_model.get(spec.model_id, []),
            }
            for spec in BASE_MODELS
        ],
        "adapter_count": len(adapters),
        "integrated_base_model": next(
            (spec.model_id for spec in BASE_MODELS if spec.status == "integrated"), None
        ),
    }
