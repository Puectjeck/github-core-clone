// main.ts — process entrypoint for the forge.
//
// The trial harness owns this process's lifecycle:
//   1. It starts us with MAKEFUN_ENV_URL pointing at its env server.
//   2. We start the app on an ephemeral port and announce that port
//      with POST /register.
//   3. The harness runs its suite against the announced port, then
//      flips GET /done to {"done": true}.
//   4. We observe done, shut the app down, and exit 0.
//
// Without MAKEFUN_ENV_URL there is no harness to serve, so we refuse
// to start (exit 1) instead of idling forever.

import { createApp } from "./app.ts";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// POST our port to the env server. Any HTTP response — including a
// 409 from a duplicate announcement — means the env heard us, so only
// transport-level failures (connection refused, reset, DNS) retry,
// with a short exponential backoff.
async function announce(base: string, port: number): Promise<void> {
  let delay = 100;
  for (;;) {
    try {
      await fetch(`${base}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ port }),
      });
      return;
    } catch {
      await sleep(delay);
      delay = Math.min(delay * 2, 1000);
    }
  }
}

// Poll GET /done until the env reports {"done": true}. While the env
// is answering we pause briefly (~100 ms) between polls; when a poll
// fails at the transport level we back off exponentially, capped at
// one second, and keep trying — the entry must never exit early.
async function waitForDone(base: string): Promise<void> {
  let failureDelay = 100;
  for (;;) {
    let done = false;
    let answered = false;
    let res: Response | null = null;
    try {
      res = await fetch(`${base}/done`);
    } catch {
      res = null;
    }
    if (res !== null) {
      // The env answered, even if the body is not the JSON we expect;
      // an unparseable body simply means "not done yet" and must not
      // be mistaken for a transport failure.
      answered = true;
      try {
        const body: unknown = await res.json();
        done =
          typeof body === "object" &&
          body !== null &&
          (body as { done?: unknown }).done === true;
      } catch {
        done = false;
      }
    }
    if (done) return;
    if (answered) {
      failureDelay = 100;
      await sleep(100);
    } else {
      await sleep(failureDelay);
      failureDelay = Math.min(failureDelay * 2, 1000);
    }
  }
}

async function run(): Promise<number> {
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").trim();
  if (envUrl === "") {
    console.error("main: MAKEFUN_ENV_URL is not set; refusing to start");
    return 1;
  }
  const base = envUrl.replace(/\/+$/, "");

  const server = createApp();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    console.error("main: app server did not bind a TCP port");
    server.close();
    return 1;
  }

  await announce(base, address.port);
  await waitForDone(base);

  // Shut down gracefully: stop accepting new connections and drop
  // idle keep-alive sockets at once — a lingering socket held open
  // by the env could otherwise keep this process alive past done.
  await new Promise<void>((resolve) => {
    server.close(() => resolve());
    server.closeIdleConnections();
  });
  return 0;
}

run().then(
  (code) => process.exit(code),
  (err: unknown) => {
    console.error("main: fatal error", err);
    process.exit(1);
  },
);
