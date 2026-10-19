from __future__ import annotations

import base64
import json
import hashlib
import gc
import io
import ipaddress
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .observability import initialize_sentry


class GenerateRequest(BaseModel):
    run_id: str
    workflow: str = Field(min_length=3, max_length=20_000)
    document_text: str = Field(min_length=1, max_length=500_000)
    adapter_id: str
    image_url: Optional[str] = Field(default=None, max_length=20_000)


NDA_ADAPTER_ID = "nda-intake/qwen3-vl-8b-v1"
RECEIPT_ADAPTER_ID = "receipt-to-expense/qwen3-vl-8b-cord-v1"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
NDA_SHORT_SYSTEM = (
    "Complete the NDA intake form using the learned workflow. Return JSON only. "
    "Use underscore-normalized strings; term must be a number_unit string, never a JSON number."
)
NDA_USER_INSTRUCTION = """Fill this NDA intake form.

Return exactly:
{
  "effective_date": "YYYY-MM-DD or null",
  "jurisdiction": "normalized_jurisdiction or null",
  "parties": ["normalized_party_names"],
  "term": "number_unit or null"
}

Replace spaces and colons inside extracted values with underscores. The term is a string such as
"3_years", not the number 3.

NDA:
"""
RECEIPT_PROMPT = (
    "Extract all information from this receipt image and return it as a JSON object with "
    "exactly this structure:\n"
    '{"items": [{"name": <string>, "count": <integer or null>, '
    '"unit_price": <number or null>, "price": <number or null>}], '
    '"subtotal": <number or null>, "discount": <number or null>, '
    '"service": <number or null>, "tax": <number or null>, "total": <number or null>}\n'
    "Rules: amounts must be plain numbers without currency symbols or thousands separators; "
    "use null for values not present on the receipt; output only the JSON object with no "
    "explanation and no markdown."
)


def generation_messages(request: GenerateRequest) -> list[dict[str, Any]]:
    """Build the inference prompt; the NDA path intentionally matches its frozen benchmark."""
    if request.adapter_id == NDA_ADAPTER_ID:
        return [
            {"role": "system", "content": NDA_SHORT_SYSTEM},
            {"role": "user", "content": NDA_USER_INSTRUCTION + request.document_text},
        ]
    if request.adapter_id == RECEIPT_ADAPTER_ID and request.image_url:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": request.image_url}},
                    {"type": "text", "text": RECEIPT_PROMPT},
                ],
            }
        ]
    return [
        {
            "role": "system",
            "content": "Complete the requested learned workflow. Return valid JSON only.",
        },
        {
            "role": "user",
            "content": f"Workflow: {request.workflow}\n\nDocument:\n{request.document_text}",
        },
    ]


def _validated_https_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("image attachment must use an HTTPS URL")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as error:
        raise ValueError("image attachment host could not be resolved") from error
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("image attachment host is not public")
    return url


class _SafeImageRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(
            req, fp, code, msg, headers, _validated_https_url(newurl)
        )


def load_receipt_image(url: str) -> Any:
    from PIL import Image, UnidentifiedImageError

    opener = urllib.request.build_opener(_SafeImageRedirectHandler())
    request = urllib.request.Request(
        _validated_https_url(url), headers={"User-Agent": "DAPTA receipt adapter/1.0"}
    )
    try:
        with opener.open(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_IMAGE_BYTES:
                raise ValueError("image attachment exceeds 10 MB")
            contents = response.read(MAX_IMAGE_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise ValueError("image attachment could not be downloaded") from error
    if len(contents) > MAX_IMAGE_BYTES:
        raise ValueError("image attachment exceeds 10 MB")
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise ValueError("attachment is not a safe supported image") from error
    if image.format not in {"JPEG", "PNG", "WEBP"}:
        image.close()
        raise ValueError("receipt image must be JPEG, PNG, or WebP")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        image.close()
        raise ValueError("receipt image has too many pixels")
    normalized = image.convert("RGB")
    image.close()
    return normalized


def jpeg_data_url(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image.close()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class ModelRuntime:
    """In-process transformers + PEFT generation (rollback path: DAPTA_QWEN_BACKEND=transformers)."""

    backend = "transformers"

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.adapter_repo = os.environ.get("QWEN_ADAPTER_REPO")
        self.loaded_adapters: dict[str, str] = {}
        self.lock = threading.Lock()

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        with self.lock:
            if self.model is not None:
                return
            model_id = os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
            self.processor = AutoProcessor.from_pretrained(model_id)
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation=os.environ.get("QWEN_ATTN_IMPLEMENTATION", "sdpa"),
            )
            self.model = base
            self.model.eval()
            self.loaded_adapters = {}
            default_adapter = os.environ.get("QWEN_ADAPTER_SUBFOLDER")
            if self.adapter_repo and default_adapter:
                self.select_adapter(default_adapter)

    def unload(self) -> None:
        import torch

        with self.lock:
            self.model = None
            self.processor = None
            self.loaded_adapters = {}
            gc.collect()
            torch.cuda.empty_cache()

    def select_adapter(self, adapter_id: str) -> None:
        if not self.adapter_repo:
            raise RuntimeError("QWEN_ADAPTER_REPO is not configured")
        if not re.fullmatch(r"[A-Za-z0-9._/-]{3,300}", adapter_id) or ".." in adapter_id:
            raise ValueError("invalid adapter id")
        alias = self.loaded_adapters.get(adapter_id)
        if alias:
            self.model.set_adapter(alias)
            return

        from peft import PeftModel

        alias = "adapter_" + hashlib.sha256(adapter_id.encode("utf-8")).hexdigest()[:12]
        if not isinstance(self.model, PeftModel):
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_repo,
                subfolder=adapter_id,
                adapter_name=alias,
            )
        else:
            self.model.load_adapter(
                self.adapter_repo,
                subfolder=adapter_id,
                adapter_name=alias,
            )
        self.model.set_adapter(alias)
        self.loaded_adapters[adapter_id] = alias
        self.model.eval()

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("model is not loaded")
        image = None
        try:
            with self.lock, torch.inference_mode():
                self.select_adapter(request.adapter_id)
                if request.image_url:
                    if request.adapter_id != RECEIPT_ADAPTER_ID:
                        raise ValueError("the selected adapter does not accept image input")
                    image = load_receipt_image(request.image_url)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": RECEIPT_PROMPT},
                            ],
                        }
                    ]
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text], images=[image], return_tensors="pt", padding=True
                    ).to("cuda")
                else:
                    messages = generation_messages(request)
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text], return_tensors="pt", padding=True
                    ).to("cuda")
                started = time.perf_counter()
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=int(os.environ.get("QWEN_MAX_NEW_TOKENS", "1200")),
                    do_sample=False,
                )
        finally:
            if image is not None:
                image.close()
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        output_text = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            output: Any = json.loads(output_text)
        except json.JSONDecodeError:
            output = output_text
        return {
            "output": output,
            "input_tokens": int(inputs.input_ids.shape[1]),
            "output_tokens": int(generated.shape[1]),
            "latency_ms": elapsed_ms,
            "ttft_ms": None,
            "backend": self.backend,
        }


def create_runtime() -> Any:
    """vLLM by default; the transformers runtime stays available as a rollback."""
    backend = os.environ.get("DAPTA_QWEN_BACKEND", "vllm").strip().lower()
    if backend == "vllm":
        from .vllm_backend import VllmRuntime

        return VllmRuntime(build_messages=generation_messages)
    if backend == "transformers":
        return ModelRuntime()
    raise RuntimeError(f"unknown DAPTA_QWEN_BACKEND: {backend}")


runtime = create_runtime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    runtime.load()
    yield


# Before the app is created, so Sentry's FastAPI integration continues the controller's trace
# from the incoming sentry-trace/baggage headers on /generate.
initialize_sentry(service="qwen-worker")
app = FastAPI(title="Dapta Qwen Worker", lifespan=lifespan)


def authorize(authorization: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("DAPTA_QWEN_TOKEN")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": runtime.backend,
        "model_loaded": runtime.model is not None,
        "loaded_adapters": list(runtime.loaded_adapters),
    }


@app.post("/generate", dependencies=[Depends(authorize)])
def generate(request: GenerateRequest) -> dict[str, Any]:
    try:
        if request.image_url:
            if request.adapter_id != RECEIPT_ADAPTER_ID:
                raise ValueError("the selected adapter does not accept image input")
            _validated_https_url(request.image_url)
            # vLLM receives the checked image inline. It does not fetch the caller's URL.
            if runtime.backend == "vllm":
                request = request.model_copy(update={
                    "image_url": jpeg_data_url(load_receipt_image(request.image_url)),
                })
        return runtime.generate(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/admin/unload", dependencies=[Depends(authorize)])
def unload() -> dict[str, Any]:
    runtime.unload()
    return {"status": "ok", "model_loaded": False}


@app.post("/admin/load", dependencies=[Depends(authorize)])
def load() -> dict[str, Any]:
    runtime.load()
    return {"status": "ok", "model_loaded": True}
