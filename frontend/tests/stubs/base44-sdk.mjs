// Minimal stand-in for @base44/sdk as used by dapta-proxy; tests set globalThis.__base44User.
// Like the real SDK, auth.me() rejects with an HTTP 401 for a visitor who is not signed in.
export function createClientFromRequest() {
  const entity = { filter: async () => [], create: async () => ({}), update: async () => ({}) };
  return {
    auth: {
      me: async () => {
        if (globalThis.__base44User) return globalThis.__base44User;
        throw Object.assign(new Error("Request failed with status code 401"), { response: { status: 401 } });
      },
    },
    asServiceRole: { entities: { DaptaSnapshot: entity } },
  };
}
