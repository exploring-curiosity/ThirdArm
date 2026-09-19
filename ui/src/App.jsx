import CameraPanel from "./CameraPanel";
import Controls from "./Controls";
import StatusBar from "./StatusBar";
import LogView from "./LogView";
import { useRobot } from "./useRobot";

export default function App() {
  const { status, online, send } = useRobot();

  return (
    <div className="app">
      <StatusBar status={status} online={online} />

      <Controls
        busy={status?.busy ?? false}
        selected={status?.selected ?? null}
        onCommand={send}
      />

      <div className="legend">
        <span><i style={{ background: "#ffffff" }} />blob + depth</span>
        <span><i style={{ background: "#78ff78" }} />point cloud</span>
        <span><i style={{ background: "#ffaa00" }} />overhead</span>
        <span><i style={{ background: "#ffff00" }} />SAM grasp axis</span>
        <span><i style={{ background: "#00dcff" }} />drop box</span>
        <span><i style={{ background: "#ff00ff" }} />selected</span>
      </div>

      <main className="views">
        <CameraPanel
          name="overhead"
          title="overhead"
          clickable
          onSelect={(p) => send({ action: "select", ...p })}
        />
        <CameraPanel
          name="wrist"
          title="wrist"
          clickable
          onSelect={(p) => send({ action: "select", ...p })}
        />
        <CameraPanel name="cloud" title="point cloud — top down, mm" />
      </main>

      <LogView lines={status?.log ?? []} />
    </div>
  );
}
