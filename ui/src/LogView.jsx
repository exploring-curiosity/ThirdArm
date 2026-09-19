import { useEffect, useRef } from "react";

/**
 * Output from the pick process and the server.
 *
 * Auto-scrolls only when the user is already at the bottom, so reading back
 * through history is not yanked away by the next line arriving.
 */
export default function LogView({ lines = [] }) {
  const ref = useRef(null);
  const stick = useRef(true);

  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    stick.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
  };

  return (
    <pre className="log" ref={ref} onScroll={onScroll}>
      {lines.join("\n")}
    </pre>
  );
}
