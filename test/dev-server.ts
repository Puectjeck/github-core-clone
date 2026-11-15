// A throwaway launcher for local conformance runs: serves app.ts on a fixed port with a known
// token. Not part of the entry — the trial entrypoint is main.ts; this exists so a developer can
// point the kit's sdk/run.ts at a working tree without the env lifecycle.

import { createApp } from '../app.ts';

const port = Number(process.env.DEV_PORT ?? '8931');
createApp().listen(port, '127.0.0.1', () => {
  console.log(`dev forge listening on http://127.0.0.1:${port}`);
});
