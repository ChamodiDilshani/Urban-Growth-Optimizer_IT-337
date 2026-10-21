import CompareViewer from "./CompareViewer.jsx";
import { detectionsToCsv, downloadBlob, downloadDataUrl, fmtArea, fmtPct } from "../utils.js";

export default function Results({ result, stem }) {
  const { summary: s, detections, images, params, timing, warnings, geojson } = result;

  return (
    <section aria-labelledby="results-heading" className="flex flex-col gap-8">
      <Verdict s={s} />

      {warnings.length > 0 && (
        <div role="status" className="rounded-lg border border-zone/50 bg-zone/10 px-4 py-3 text-sm">
          <p className="font-semibold">Check before relying on this result</p>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 lg:grid-cols-5">
        <Stat label="New structures" value={s.n_objects} />
        <Stat label="New built area" value={fmtArea(s.total_m2)} />
        <Stat label="Inside the zone" value={s.has_zone ? s.n_flagged : "No zone"} tone={s.n_flagged ? "alert" : undefined} />
        <Stat label="Area inside the zone" value={s.has_zone ? fmtArea(s.breach_m2) : "No zone"} tone={s.n_flagged ? "alert" : undefined} />
        <Stat label="Zone covers" value={s.has_zone ? `${fmtPct(s.zone_pct)} of image` : "No zone"} />
      </dl>

      <div className="rounded-xl border border-line bg-surface p-4 sm:p-6">
        <h2 id="results-heading" className="mb-3 text-lg font-bold">
          Compare the two dates
        </h2>
        <CompareViewer
          beforeUrl={images.before}
          afterUrl={images.after}
          changeOverlay={images.change_overlay}
          zoneOverlay={images.zone_overlay}
          hasZone={s.has_zone}
          width={params.width}
          height={params.height}
        />
        <Legend hasZone={s.has_zone} />
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Figure
          title="Detected change"
          caption="Every new structure the model found on the after image, in yellow."
          src={images.change_result}
          onDownload={() => downloadDataUrl(images.change_result, `${stem}_1_change_result.png`)}
        />
        <Figure
          title="Compliance check"
          caption={
            s.has_zone
              ? "Orange is the protected zone. Red structures cross into it, green ones stay outside."
              : "No zone image was given, so every structure is marked compliant."
          }
          src={images.compliance_result}
          onDownload={() => downloadDataUrl(images.compliance_result, `${stem}_2_compliance_result.png`)}
        />
      </div>

      <div className="rounded-xl border border-line bg-surface">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3 sm:px-6">
          <h2 className="text-lg font-bold">Detected structures</h2>
          <div className="flex flex-wrap gap-2">
            <SmallButton onClick={() => downloadBlob(detectionsToCsv(detections), `${stem}_detections.csv`, "text/csv")}>
              Download CSV
            </SmallButton>
            <SmallButton
              onClick={() => downloadBlob(JSON.stringify(geojson, null, 2), `${stem}_detections.geojson`, "application/geo+json")}
            >
              Download GeoJSON
            </SmallButton>
          </div>
        </div>
        <DetectionsTable rows={detections} hasZone={s.has_zone} />
      </div>

      <p className="text-xs text-muted">
        Threshold {params.threshold.toFixed(2)}, flip averaging {params.tta ? "on" : "off"}, {params.width}×{params.height}px at{" "}
        {params.pixel_size_m} m per pixel. Model time {timing.model_s}s. Areas assume LEVIR-CD+ resolution; images at a different
        resolution will report areas off by the square of the scale difference. GeoJSON coordinates are in pixels.
      </p>
    </section>
  );
}

function Verdict({ s }) {
  if (!s.has_zone) {
    return (
      <div className="rounded-xl border-l-8 border-change bg-surface px-5 py-5 sm:px-7">
        <p className="text-2xl font-extrabold leading-tight sm:text-3xl">
          {s.n_objects === 0
            ? "No new construction found between the two dates."
            : `${s.n_objects} new ${s.n_objects === 1 ? "structure" : "structures"} found, covering ${fmtArea(s.total_m2)}.`}
        </p>
        <p className="mt-2 text-muted">Add a zone image to check them against a protected area.</p>
      </div>
    );
  }
  if (s.n_flagged > 0) {
    return (
      <div className="rounded-xl border-l-8 border-alert bg-surface px-5 py-5 sm:px-7">
        <p className="text-2xl font-extrabold leading-tight text-alert sm:text-3xl">
          {s.n_flagged} {s.n_flagged === 1 ? "structure is" : "structures are"} built inside the protected zone.
        </p>
        <p className="mt-2 text-muted">
          {fmtArea(s.breach_m2)} of new construction falls inside the zone, out of {fmtArea(s.total_m2)} detected in total.
        </p>
      </div>
    );
  }
  return (
    <div className="rounded-xl border-l-8 border-ok bg-surface px-5 py-5 sm:px-7">
      <p className="text-2xl font-extrabold leading-tight text-ok sm:text-3xl">No new construction inside the protected zone.</p>
      <p className="mt-2 text-muted">
        {s.n_objects === 0
          ? "The model found no new structures anywhere in the image."
          : `${s.n_objects} new ${s.n_objects === 1 ? "structure was" : "structures were"} found, all outside the zone.`}
      </p>
    </div>
  );
}

function Stat({ label, value, tone }) {
  return (
    <div className="bg-surface px-4 py-3">
      <dt className="text-sm text-muted">{label}</dt>
      <dd className={`tabular mt-1 text-xl font-bold ${tone === "alert" ? "text-alert" : ""}`}>{value}</dd>
    </div>
  );
}

function Legend({ hasZone }) {
  const items = hasZone
    ? [
        ["var(--color-alert)", "Inside the zone"],
        ["var(--color-ok)", "Outside the zone"],
        ["var(--color-zone)", "Protected zone"],
      ]
    : [["var(--color-change)", "New structure"]];
  return (
    <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-muted">
      {items.map(([c, l]) => (
        <li key={l} className="flex items-center gap-2">
          <span className="size-3 rounded-sm" style={{ background: c }} aria-hidden="true" />
          {l}
        </li>
      ))}
    </ul>
  );
}

function Figure({ title, caption, src, onDownload }) {
  return (
    <figure className="flex flex-col overflow-hidden rounded-xl border border-line bg-surface">
      <a href={src} target="_blank" rel="noreferrer" className="block bg-black" title="Open full size">
        <img src={src} alt={title} className="w-full" />
      </a>
      <figcaption className="flex flex-1 items-start justify-between gap-4 p-4">
        <div>
          <p className="font-semibold">{title}</p>
          <p className="mt-0.5 text-sm text-muted">{caption}</p>
        </div>
        <SmallButton onClick={onDownload}>Download PNG</SmallButton>
      </figcaption>
    </figure>
  );
}

function DetectionsTable({ rows, hasZone }) {
  if (rows.length === 0) {
    return <p className="px-4 py-6 text-muted sm:px-6">No structures above the minimum size were detected.</p>;
  }
  const sorted = [...rows].sort((a, b) => Number(b.flag) - Number(a.flag) || b.inside_m2 - a.inside_m2 || a.id - b.id);
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full min-w-[640px] text-left text-sm">
        <thead className="text-muted">
          <tr className="border-b border-line">
            <Th>Structure</Th>
            <Th>Status</Th>
            <Th right>Area</Th>
            <Th right>Inside zone</Th>
            <Th right>Share inside</Th>
            <Th right>Centre (x, y px)</Th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.id} className="border-b border-line last:border-0">
              <td className="px-4 py-2.5 font-semibold sm:px-6">#{r.id}</td>
              <td className="px-4 py-2.5">
                <Status row={r} hasZone={hasZone} />
              </td>
              <td className="px-4 py-2.5 text-right">{r.area_m2.toLocaleString()} m²</td>
              <td className="px-4 py-2.5 text-right">{r.inside_m2.toLocaleString()} m²</td>
              <td className="px-4 py-2.5 text-right">{r.inside_pct.toFixed(1)}%</td>
              <td className="px-4 py-2.5 text-right text-muted sm:pr-6">
                {r.cx}, {r.cy}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Status({ row, hasZone }) {
  if (!hasZone) return <span className="text-muted">Not checked</span>;
  if (!row.flag)
    return (
      <span className="inline-flex items-center gap-1.5 font-medium text-ok">
        <span className="size-2 rounded-full bg-ok" aria-hidden="true" /> Compliant
      </span>
    );
  const major = row.status.includes("major");
  return (
    <span className="inline-flex items-center gap-1.5 font-semibold text-alert">
      <span className="size-2 rounded-full bg-alert" aria-hidden="true" />
      {major ? "Encroachment, major" : "Encroachment, partial"}
    </span>
  );
}

function Th({ children, right }) {
  return <th className={`px-4 py-2.5 font-medium first:sm:pl-6 last:sm:pr-6 ${right ? "text-right" : ""}`}>{children}</th>;
}

function SmallButton({ children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="shrink-0 rounded-md border border-line px-3 py-1.5 text-sm font-semibold hover:bg-sunken focus-visible:outline-2 focus-visible:outline-primary"
    >
      {children}
    </button>
  );
}
