// Everything the page knows about the robot comes through here.
//
// Two channels, deliberately different: camera panels are MJPEG <img> streams
// the browser decodes natively, and state is small JSON polled on a timer.
// Pushing frames through JSON would mean base64 in a React state update
// several times a second, which is exactly the thing that makes a page feel
// slow.

export const streamUrl = (name) => `/stream/${name}`;

// `timeoutMs` guards against a request that hangs rather than fails. A
// half-open TCP connection -- the service killed, the laptop suspended --
// produces exactly that, and an un-timed fetch would wait indefinitely.
export async function getStatus(timeoutMs = 4000) {
  const ctl = new AbortController();
  const bail = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch("/api/status", { signal: ctl.signal });
    if (!r.ok) throw new Error(`status ${r.status}`);
    return await r.json();
  } finally {
    clearTimeout(bail);
  }
}

export async function sendCommand(body) {
  const r = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`command ${r.status}`);
  return r.json();
}
