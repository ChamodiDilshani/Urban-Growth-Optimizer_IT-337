const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function parse(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error page */
  }
  if (!res.ok) {
    const detail = body?.detail;
    const msg = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : null;
    throw new Error(msg || `Server responded with ${res.status}.`);
  }
  return body;
}

export async function getHealth() {
  const res = await fetch(`${BASE}/api/health`);
  return parse(res);
}

export async function analyze({ before, zone, after, threshold, tta }, signal) {
  const form = new FormData();
  form.append("before", before);
  form.append("after", after);
  if (zone) form.append("zone", zone);
  if (threshold != null) form.append("threshold", String(threshold));
  form.append("tta", tta ? "true" : "false");
  const res = await fetch(`${BASE}/api/analyze`, { method: "POST", body: form, signal });
  return parse(res);
}
