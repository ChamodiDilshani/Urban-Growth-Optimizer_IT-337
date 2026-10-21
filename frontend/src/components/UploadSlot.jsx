import { useEffect, useId, useRef, useState } from "react";

const ACCEPT = "image/png,image/jpeg,image/tiff,image/bmp,image/webp";

/**
 * One image input: click or drop a file, see a preview, replace or remove it.
 */
export default function UploadSlot({ step, title, hint, file, onChange, required, accent }) {
  const inputId = useId();
  const inputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [dims, setDims] = useState(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      setDims(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    const img = new Image();
    img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => setDims(null); // e.g. TIFF: browser can't preview, server can still read it
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const pick = (f) => {
    if (f && f.type && !f.type.startsWith("image/")) return;
    onChange(f || null);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline gap-3 md:min-h-16">
        <span
          className="grid size-7 shrink-0 place-items-center rounded-full text-sm font-bold text-white"
          style={{ background: accent }}
          aria-hidden="true"
        >
          {step}
        </span>
        <div>
          <label htmlFor={inputId} className="text-base font-semibold leading-tight">
            {title}
            {!required && <span className="ml-2 text-sm font-normal text-muted">optional</span>}
          </label>
          <p className="mt-0.5 text-sm leading-snug text-muted">{hint}</p>
        </div>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files?.[0]);
        }}
        className={[
          "relative aspect-[4/3] overflow-hidden md:aspect-square rounded-lg border-2 bg-sunken transition-colors",
          dragging ? "border-primary" : file ? "border-transparent" : "border-dashed border-line",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          onChange={(e) => {
            pick(e.target.files?.[0]);
            e.target.value = "";
          }}
        />

        {file ? (
          <>
            {dims ? (
              <img src={preview} alt={`${title} preview`} className="size-full object-cover" />
            ) : (
              <div className="grid size-full place-items-center p-6 text-center text-sm text-muted">
                {file.name}
                <br />
                Preview not available in the browser. The server will still read it.
              </div>
            )}
            <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-2 bg-black/65 px-3 py-2 text-xs text-white">
              <span className="truncate" title={file.name}>
                {file.name}
                {dims && <span className="tabular ml-2 opacity-75">{dims.w}×{dims.h}</span>}
              </span>
              <span className="flex shrink-0 gap-1">
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  className="rounded px-2 py-1 font-semibold hover:bg-white/20 focus-visible:outline-2 focus-visible:outline-white"
                >
                  Replace
                </button>
                <button
                  type="button"
                  onClick={() => onChange(null)}
                  className="rounded px-2 py-1 font-semibold hover:bg-white/20 focus-visible:outline-2 focus-visible:outline-white"
                >
                  Remove
                </button>
              </span>
            </div>
          </>
        ) : (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="flex size-full flex-col items-center justify-center gap-2 p-6 text-center text-muted hover:text-ink focus-visible:outline-2 focus-visible:outline-primary"
          >
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <path d="M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" strokeLinecap="round" />
            </svg>
            <span className="text-sm font-medium">Drop an image or click to choose</span>
            <span className="text-xs">PNG, JPG, TIFF</span>
          </button>
        )}
      </div>
    </div>
  );
}
