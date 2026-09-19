const COLOURS = [
  { name: "blue", swatch: "#4aa8ff" },
  { name: "green", swatch: "#5dd47f" },
  { name: "orange", swatch: "#ff9f45" },
  { name: "yellow", swatch: "#ffe14a" },
];

/**
 * The action row.
 *
 * Pick buttons are disabled while a pick runs: track_pick owns the arm for
 * the duration, and a second one would fight it for the same hardware. Stop
 * stays enabled precisely because that is when it is needed.
 */
export default function Controls({ busy, selected, onCommand }) {
  return (
    <div className="controls">
      {COLOURS.map(({ name, swatch }) => (
        <button
          key={name}
          className="pick"
          style={{ background: swatch }}
          disabled={busy}
          onClick={() => onCommand({ action: "pick", colour: name })}
        >
          pick {name}
        </button>
      ))}

      <button
        className="sort"
        disabled={busy}
        onClick={() => onCommand({ action: "sort" })}
      >
        sort table
      </button>

      <button className="stop" onClick={() => onCommand({ action: "stop" })}>
        stop
      </button>

      <span className="spacer" />

      <button className="ghost" onClick={() => onCommand({ action: "home" })}>
        top-pose
      </button>
      <button className="ghost" onClick={() => onCommand({ action: "box" })}>
        find drop box
      </button>
      <button
        className="ghost"
        disabled={!selected}
        onClick={() => onCommand({ action: "clear" })}
      >
        clear selection
      </button>
    </div>
  );
}
