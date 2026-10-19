// Stand-in for src/base44Client.ts: records dapta-proxy invocations and sign-in redirects;
// tests set the reply.
export const invocations = [];
export const signInRedirects = [];
export const providerLogins = [];
export const base44 = {
  functions: {
    async invoke(name, payload) {
      invocations.push({ name, payload });
      const reply = globalThis.__base44Reply;
      if (!reply) return { data: [] };
      return reply(name, payload);
    },
  },
  auth: {
    redirectToLogin(nextUrl) {
      signInRedirects.push(nextUrl);
    },
    loginWithProvider(provider, fromUrl) {
      providerLogins.push({ provider, fromUrl });
    },
  },
};
