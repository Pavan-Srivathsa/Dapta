import "./dom.ts";

import assert from "node:assert/strict";
import { test } from "node:test";

import { act, createElement } from "react";
import { createRoot } from "react-dom/client";

import { api, SignInRequiredError, signIn } from "../src/api.ts";
import App from "../src/App.tsx";
import { container } from "./dom.ts";
// Same module the API client imports (the test loader maps src/base44Client.ts to this stub).
import { invocations, providerLogins, signInRedirects } from "./stubs/base44-client.mjs";

const unauthorized = async () => {
  throw Object.assign(new Error("Request failed with status code 401"),
    { response: { status: 401, data: { detail: "Sign in to use Dapta" } } });
};

test("a 401 raises SignInRequiredError and never navigates on its own", async () => {
  (globalThis as any).__base44Reply = unauthorized;
  await assert.rejects(api.adapters(), SignInRequiredError);
  assert.deepEqual(signInRedirects, [], "no automatic redirect: /login on this site serves the app again");
  (globalThis as any).__base44Reply = undefined;
});

test("Sign in uses Base44's OAuth route and returns to this page", () => {
  signIn();
  assert.deepEqual(providerLogins.at(-1), { provider: "google", fromUrl: window.location.href });
});

test("a signed-out visitor sees the sign-in screen after one load, with no reload loop", async () => {
  (globalThis as any).__base44Reply = unauthorized;
  const original = console.error;
  console.error = () => undefined;
  const root = createRoot(container());
  try {
    const before = invocations.length;
    await act(async () => { root.render(createElement(App)); });
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 1800)); });
    assert.match(document.body.textContent || "", /Sign in to use Dapta/);
    assert.equal(invocations.length - before, 5, "one initial load of five requests, then nothing");
    assert.deepEqual(signInRedirects, []);
  } finally {
    act(() => root.unmount());
    console.error = original;
    document.body.innerHTML = "";
    (globalThis as any).__base44Reply = undefined;
  }
});

test("other failures are reported without signing in", async () => {
  (globalThis as any).__base44Reply = async () => {
    throw Object.assign(new Error("Request failed with status code 502"),
      { response: { status: 502, data: { detail: "Dapta controller is unreachable" } } });
  };
  await assert.rejects(api.runs(), /controller is unreachable/);
  (globalThis as any).__base44Reply = undefined;
});
