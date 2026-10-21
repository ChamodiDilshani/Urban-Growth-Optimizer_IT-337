import { useEffect, useRef, useState } from "react";
import { analyze, getHealth } from "../api.js";
import UploadSlot from "../components/UploadSlot.jsx";
import Results from "../components/Results.jsx";
import { baseName } from "../utils.js";

export default function EncroachmentCheck() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const [before, setBefore] = useState(null);
  const [zone, setZone] = useState(null);
  const [after, setAfter] = useState(null);

  const [tta, setTta] = useState(true);

  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [resultStem, setResultStem] = useState("result");
  const abortRef = useRef(null);
  const resultsRef = useRef(null);

  // Model status
  useEffect(() => {
    let alive = true;
    getHealth()
      .then((h) => alive && setHealth(h))
      .catch(() => alive && setHealthError("Cannot reach the analysis server. Start it with: uvicorn app.main:app --port 8000"));
    return () => {
      alive = false;
    };
  }, []);

  // Elapsed-time counter while the model runs
  useEffect(() => {
    if (!running) return;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, [running]);

  const ready = before && after && !running;
  const modelReady = health?.ready;

  async function run() {
    setError(null);
    setRunning(true);
    setElapsed(0);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      // No threshold is sent: the server always uses threshold.json (fixed, as in the notebook)
      const res = await analyze({ before, zone, after, tta }, ctrl.signal);
      setResult(res);
      setResultStem(baseName(after));
      requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message || "The check failed.");
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }

  function resetAll() {
    abortRef.current?.abort();
    setBefore(null);
    setZone(null);
    setAfter(null);
    setResult(null);
    setError(null);
  }

  const m = health?.test_metrics;

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-10 px-4 py-8 sm:px-6 lg:py-12">
      <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">Land encroachment check</h1>
          <p className="mt-2 text-lg leading-relaxed text-muted">
            Compare two satellite images of the same place and find new buildings that cross into a protected zone.
          </p>
        </div>
        <ModelStatus health={health} error={healthError} metrics={m} />
      </header>

      <main className="flex flex-col gap-10">
        <section aria-labelledby="inputs-heading" className="rounded-xl border border-line bg-surface p-4 sm:p-6">
          <h2 id="inputs-heading" className="sr-only">
            Input images
          </h2>
          <div className="grid gap-6 md:grid-cols-3">
            <UploadSlot
              step={1}
              title="Before image"
              hint="The older image (T1)."
              file={before}
              onChange={setBefore}
              required
              accent="var(--color-primary)"
            />
            <UploadSlot
              step={2}
              title="Zone image"
              hint="The before image with the protected area painted red."
              file={zone}
              onChange={setZone}
              accent="var(--color-zone)"
            />
            <UploadSlot
              step={3}
              title="After image"
              hint="The newer image (T2) of the same area."
              file={after}
              onChange={setAfter}
              required
              accent="var(--color-primary)"
            />
          </div>

          <ZoneHelp />

          <div className="mt-6 flex flex-col gap-5 border-t border-line pt-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:gap-10">
              <div className="text-sm">
                <p className="font-semibold">Detection threshold</p>
                <p className="tabular mt-1 text-base font-bold">
                  {health?.threshold != null ? health.threshold.toFixed(2) : "—"}
                </p>
                <p className="mt-0.5 text-xs text-muted">Fixed value chosen on the validation set.</p>
              </div>
              <label className="flex cursor-pointer items-start gap-2.5 text-sm">
                <input type="checkbox" checked={tta} onChange={(e) => setTta(e.target.checked)} className="mt-0.5 size-4 accent-primary" />
                <span>
                  <span className="font-semibold">Flip averaging</span>
                  <span className="block text-xs text-muted">Steadier results, about 4× slower.</span>
                </span>
              </label>
            </div>

            <div className="flex items-center gap-3">
              {(before || zone || after || result) && !running && (
                <button
                  type="button"
                  onClick={resetAll}
                  className="rounded-lg px-4 py-3 font-semibold text-muted hover:text-ink focus-visible:outline-2 focus-visible:outline-primary"
                >
                  Start over
                </button>
              )}
              {running ? (
                <button
                  type="button"
                  onClick={() => abortRef.current?.abort()}
                  className="rounded-lg border border-line px-5 py-3 font-semibold hover:bg-sunken focus-visible:outline-2 focus-visible:outline-primary"
                >
                  Cancel
                </button>
              ) : null}
              <button
                type="button"
                onClick={run}
                disabled={!ready || !modelReady}
                className="min-w-44 rounded-lg bg-primary px-6 py-3 font-bold text-primary-ink hover:brightness-110 focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-40"
              >
                {running ? `Checking… ${elapsed}s` : "Run check"}
              </button>
            </div>
          </div>

          <p className="mt-3 text-right text-sm text-muted" aria-live="polite">
            {!before || !after
              ? "Add a before and an after image to run the check."
              : !zone && !running
                ? "Without a zone image, only new construction is detected."
                : running
                  ? "Running the change-detection model. Large images on a CPU can take a minute."
                  : ""}
          </p>

          {error && (
            <div role="alert" className="mt-4 rounded-lg border border-alert/50 bg-alert/10 px-4 py-3 text-sm">
              <span className="font-semibold text-alert">The check didn't finish. </span>
              {error}
            </div>
          )}
        </section>

        <div ref={resultsRef} className="scroll-mt-6">
          {result && <Results result={result} stem={resultStem} />}
        </div>
      </main>


    </div>
  );
}

function ModelStatus({ health, error, metrics }) {
  if (error) {
    return (
      <p role="alert" className="max-w-sm rounded-lg border border-alert/50 bg-alert/10 px-4 py-3 text-sm">
        {error}
      </p>
    );
  }
  if (!health) return <p className="text-sm text-muted">Connecting to the model…</p>;
  if (!health.ready) {
    return (
      <p role="alert" className="max-w-sm rounded-lg border border-alert/50 bg-alert/10 px-4 py-3 text-sm">
        The model didn't load. {health.error}
      </p>
    );
  }
  return (
    <div className="text-sm md:text-right">
      <p className="flex items-center gap-2 font-semibold md:justify-end">
        <span className="size-2 rounded-full bg-ok" aria-hidden="true" />
        Model ready on {health.device.toUpperCase()}
      </p>
    </div>
  );
}

function ZoneHelp() {
  return (
    <details className="mt-6 rounded-lg bg-sunken px-4 py-3 text-sm">
      <summary className="cursor-pointer font-semibold">How to make the zone image</summary>
      <ol className="mt-2 list-decimal space-y-1 pl-5 leading-relaxed text-muted">
        <li>Open a copy of the before image in any editor (Paint, Photoshop, GIMP).</li>
        <li>Paint the protected area in red. A semi-transparent red fill works as long as it is clearly reddish.</li>
        <li>Save it at the same size as the before image, without cropping, and upload it here.</li>
      </ol>
      <p className="mt-2 text-muted">
        The server finds the zone by comparing this image with the before image, so the red area must be at least about 500 pixels
        and anything else should stay unchanged.
      </p>
    </details>
  );
}