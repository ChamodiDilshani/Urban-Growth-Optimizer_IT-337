import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Swipe between the before and after image, with the model's overlays
 * (detected structures, protected zone) laid on top of both.
 */
export default function CompareViewer({ beforeUrl, afterUrl, changeOverlay, zoneOverlay, hasZone, width, height }) {
  const boxRef = useRef(null);
  const [pos, setPos] = useState(50);
  const [showChange, setShowChange] = useState(true);
  const [showZone, setShowZone] = useState(true);
  const dragging = useRef(false);

  const setFromX = useCallback((clientX) => {
    const r = boxRef.current?.getBoundingClientRect();
    if (!r) return;
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  }, []);

  useEffect(() => {
    const move = (e) => dragging.current && setFromX(e.clientX);
    const up = () => (dragging.current = false);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [setFromX]);

  const overlays = (
    <>
      {hasZone && showZone && <img src={zoneOverlay} alt="" className="pointer-events-none absolute inset-0 size-full" />}
      {showChange && <img src={changeOverlay} alt="" className="pointer-events-none absolute inset-0 size-full" />}
    </>
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted">Drag across the image to compare the two dates.</p>
        <div className="flex flex-wrap gap-4 text-sm">
          <Toggle checked={showChange} onChange={setShowChange} label="Detected structures" />
          {hasZone && <Toggle checked={showZone} onChange={setShowZone} label="Protected zone" />}
        </div>
      </div>

      <div
        ref={boxRef}
        className="relative w-full touch-none select-none overflow-hidden rounded-lg bg-black"
        style={{ aspectRatio: `${width} / ${height}` }}
        onPointerDown={(e) => {
          dragging.current = true;
          setFromX(e.clientX);
        }}
      >
        {/* after image underneath (right side) */}
        <img src={afterUrl} alt="After" className="absolute inset-0 size-full" draggable={false} />
        {overlays}

        {/* before image clipped to the left of the handle */}
        <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={beforeUrl} alt="Before" className="absolute inset-0 size-full" draggable={false} />
          {overlays}
        </div>

        <span className="pointer-events-none absolute left-3 top-3 rounded bg-black/70 px-2 py-1 text-xs font-semibold text-white">
          Before
        </span>
        <span className="pointer-events-none absolute right-3 top-3 rounded bg-black/70 px-2 py-1 text-xs font-semibold text-white">
          After
        </span>

        {/* handle */}
        <div className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(0,0,0,.4)]" style={{ left: `${pos}%` }}>
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={Math.round(pos)}
            onChange={(e) => setPos(Number(e.target.value))}
            aria-label="Comparison position"
            className="peer sr-only"
          />
          <div className="absolute top-1/2 left-1/2 grid size-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-full bg-white text-black shadow-md peer-focus-visible:outline-3 peer-focus-visible:outline-primary">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
              <path d="m9 6-6 6 6 6M15 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

function Toggle({ checked, onChange, label }) {
  return (
    <label className="flex cursor-pointer items-center gap-2">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="size-4 accent-primary" />
      {label}
    </label>
  );
}
