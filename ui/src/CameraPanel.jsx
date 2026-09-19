import { useCallback, useEffect, useRef, useState } from "react";
import { streamUrl } from "./api";

// How long to wait before re-opening a stream that dropped, and the ceiling
// that backoff climbs to. A stream dies whenever the service restarts or
// pauses its camera pumps for a pick, which is routine -- so recovery has to
// be automatic, but it must not hammer a service that is still down.
const RETRY_MS = 1000;
const RETRY_MAX_MS = 10000;

/**
 * One camera view. Clicking it selects an object.
 *
 * The click is reported as a FRACTION of the panel, not as pixels: the image
 * is laid out responsively, so the browser's pixel coordinates mean nothing
 * to the server. The server maps the fraction back through that camera's own
 * calibration.
 *
 * The stream reconnects itself. An MJPEG <img> that loses its connection
 * stays broken forever -- the browser will not retry it -- so losing the
 * service once left dead panels until the page was reloaded by hand. The
 * cache-busting `t` parameter matters: without it the browser may serve the
 * dead response from cache instead of opening a new connection.
 */
export default function CameraPanel({ name, title, clickable, onSelect }) {
  const [attempt, setAttempt] = useState(0);
  const [live, setLive] = useState(false);
  const retry = useRef(null);
  const backoff = useRef(RETRY_MS);

  const scheduleRetry = useCallback(() => {
    setLive(false);
    clearTimeout(retry.current);
    retry.current = setTimeout(() => {
      backoff.current = Math.min(backoff.current * 2, RETRY_MAX_MS);
      setAttempt((n) => n + 1);
    }, backoff.current);
  }, []);

  // A frame arriving means the stream is healthy, so the next failure should
  // retry promptly rather than inheriting a long backoff from an earlier one.
  const handleLoad = useCallback(() => {
    backoff.current = RETRY_MS;
    setLive(true);
  }, []);

  useEffect(() => () => clearTimeout(retry.current), []);

  const handleClick = (e) => {
    if (!clickable || !live) return;
    const r = e.currentTarget.getBoundingClientRect();
    onSelect({
      panel: name,
      x: (e.clientX - r.left) / r.width,
      y: (e.clientY - r.top) / r.height,
    });
  };

  return (
    <figure className="panel">
      <figcaption>
        {title}
        {clickable && live && <span className="hint">click to select</span>}
        {!live && <span className="hint offline">reconnecting…</span>}
      </figcaption>
      <img
        key={attempt}
        src={`${streamUrl(name)}?t=${attempt}`}
        alt={title}
        onClick={handleClick}
        onLoad={handleLoad}
        onError={scheduleRetry}
        className={live ? "" : "stale"}
        style={{ cursor: clickable && live ? "crosshair" : "default" }}
      />
    </figure>
  );
}
