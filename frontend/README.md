# Manhole & Drain Inspector

The React frontend provides image upload and analysis, phone photo capture,
and a dashboard for submitted reports. Image detection runs in the browser.

## Setup

From this directory, install the dependencies and start the Vite development
server:

```powershell
npm ci
npm run dev
```

The Vite development server proxies `/api` requests to
`http://127.0.0.1:8000`, where the backend API should be running.

The image-analysis page loads the ONNX model from
`public/model/model.onnx`. Make sure the model is available at that path before
using image analysis.

Supabase reporting uses the `VITE_SUPABASE_URL` and
`VITE_SUPABASE_ANON_KEY` environment variables. Set them in a local `.env`
file when Supabase reporting is configured. Do not commit `.env` files or
service-role keys.

## Checks

```powershell
npm run lint
npm run build
```
