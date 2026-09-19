import { useCallback, useEffect, useRef, useState } from "react";
import { getStatus, sendCommand } from "./api";

// Poll interval. Fast enough that a button feels responsive, slow enough that
// it is not competing with the MJPEG streams for the browser's connections.
const POLL_MS = 350;
// While the service is unreachable, back off instead of polling at full rate:
// a dead service does not need 3 requests a second, and the browser caps how
// many connections it will open to one origin -- spending them on failing
// polls is what stops the camera streams from reconnecting.
const RETRY_MS = 1000;
const RETRY_MAX_MS = 5000;
// A fetch with no timeout can hang indefinitely. Without this the poll loop
// stops permanently the moment one request hangs: `tick` never returns, so it
// never schedules the next one, and the page looks frozen rather than
// disconnected.
const TIMEOUT_MS = 4000;

export function useRobot() {
  const [status, setStatus] = useState(null);
  const [online, setOnline] = useState(false);
  const timer = useRef(null);
  const backoff = useRef(RETRY_MS);

  useEffect(() => {
    let alive = true;

    const tick = async () => {
      let delay = POLL_MS;
      try {
        const s = await getStatus(TIMEOUT_MS);
        if (!alive) return;
        setStatus(s);
        setOnline(true);
        backoff.current = RETRY_MS;
      } catch {
        if (!alive) return;
        // Keep the last known status on screen rather than blanking the UI:
        // a momentary blip should not make the page forget what it knew.
        // The connection banner is what tells the user it is stale.
        setOnline(false);
        delay = backoff.current;
        backoff.current = Math.min(backoff.current * 2, RETRY_MAX_MS);
      }
      // Scheduled here, in one place, so the loop survives every outcome --
      // including a hung or rejected request.
      if (alive) timer.current = setTimeout(tick, delay);
    };

    tick();
    return () => {
      alive = false;
      clearTimeout(timer.current);
    };
  }, []);

  // Optimistically mark the arm busy so the pick buttons disable on click
  // rather than on the next poll, which would otherwise allow a double-send.
  const send = useCallback(async (body) => {
    if (body.action === "pick") {
      setStatus((s) => (s ? { ...s, busy: true } : s));
    }
    try {
      await sendCommand(body);
    } catch {
      // The next poll reports the real state. If the service is down, the
      // optimistic "busy" is cleared by that poll rather than sticking.
      setStatus((s) => (s ? { ...s, busy: false } : s));
    }
  }, []);

  return { status, online, send };
}
