export const fmtArea = (m2) =>
  m2 >= 10000 ? `${(m2 / 10000).toFixed(2)} ha` : `${Math.round(m2).toLocaleString()} m²`;

export const fmtPct = (v) => `${v.toFixed(1)}%`;

export function downloadBlob(content, filename, type) {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export function detectionsToCsv(rows) {
  const cols = ["id", "cx", "cy", "area_m2", "inside_m2", "inside_pct", "status", "flag"];
  const esc = (v) => (typeof v === "string" && /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
}

export const baseName = (file) => (file?.name || "image").replace(/\.[^.]+$/, "");
