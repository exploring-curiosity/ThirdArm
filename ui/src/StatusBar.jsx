/** One sensor's live count, with its own colour so it ties to the overlays. */
function Stat({ label, value, sub, tone }) {
  return (
    <div className="stat">
      <span className="dot" style={{ background: tone }} />
      <span className="label">{label}</span>
      <span className="value">{value}</span>
      {sub && <span className="sub">{sub}</span>}
    </div>
  );
}

export default function StatusBar({ status, online }) {
  if (!online) {
    return (
      <header className="status offline">
        <strong>arm control</strong>
        <span className="warn">server not responding</span>
      </header>
    );
  }
  if (!status) {
    return (
      <header className="status">
        <strong>arm control</strong>
        <span className="sub">connecting…</span>
      </header>
    );
  }

  return (
    <header className="status">
      <strong>arm control</strong>
      <div className="stats">
        <Stat label="gripper" value={status.gripper} tone="#8a93a6" />
        <Stat label="blob" value={status.blob} tone="#ffffff" />
        <Stat
          label="cloud"
          value={status.seg}
          sub={`${status.seg_age}s`}
          tone="#78ff78"
        />
        <Stat label="overhead" value={status.overhead} tone="#ffaa00" />
        <Stat
          label="SAM"
          value={status.sam}
          sub={`${status.sam_age}s`}
          tone="#ffff00"
        />
      </div>
      <span className={status.busy ? "state busy" : "state idle"}>
        {status.busy ? "picking" : "idle"}
      </span>
      {status.selected && (
        <span className="selected">selected {status.selected}</span>
      )}
    </header>
  );
}
