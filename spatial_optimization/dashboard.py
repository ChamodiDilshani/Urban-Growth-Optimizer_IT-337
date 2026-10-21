"""
Interactive Web Dashboard for Spatial Optimization Results.

Launches a local HTTP server with an interactive HTML dashboard
displaying benchmark results, plots, and AHP analysis.

Usage:
    python dashboard.py
    Then open http://localhost:8050 in your browser.
"""

import http.server
import socketserver
import json
import base64
import os
import webbrowser
import threading

PORT = 8050
OUTPUT_DIR = "outputs"


def image_to_base64(path: str) -> str:
    """Convert an image file to base64 string for HTML embedding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_dashboard_html() -> str:
    """Build the complete HTML dashboard."""

    # Load benchmark results
    with open(os.path.join(OUTPUT_DIR, "benchmark_results.json"), "r") as f:
        results = json.load(f)

    ga = results[0]
    pso = results[1]

    # Encode images as base64
    convergence_b64 = image_to_base64(os.path.join(OUTPUT_DIR, "convergence_curves.png"))
    heatmap_b64 = image_to_base64(os.path.join(OUTPUT_DIR, "suitability_heatmap.png"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spatial Optimization Dashboard - Matale District, Sri Lanka</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
        }}

        .header {{
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding: 20px 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 1.8rem;
            background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }}

        .header p {{
            color: #aaa;
            font-size: 0.95rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
        }}

        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            background: rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }}

        .card h3 {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 8px;
        }}

        .card .value {{
            font-size: 2.2rem;
            font-weight: 700;
        }}

        .card .sub {{
            font-size: 0.85rem;
            color: #999;
            margin-top: 4px;
        }}

        .winner {{ color: #6bcb77; }}
        .ga-color {{ color: #ff6b6b; }}
        .pso-color {{ color: #4d96ff; }}
        .ahp-color {{ color: #ffd93d; }}

        .section {{
            background: rgba(255,255,255,0.04);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.06);
        }}

        .section h2 {{
            font-size: 1.3rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .plot-container {{
            text-align: center;
        }}

        .plot-container img {{
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}

        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}

        th {{
            color: #aaa;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        td {{
            font-size: 1rem;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .badge-pass {{
            background: rgba(107, 203, 119, 0.2);
            color: #6bcb77;
            border: 1px solid rgba(107, 203, 119, 0.3);
        }}

        .badge-winner {{
            background: rgba(77, 150, 255, 0.2);
            color: #4d96ff;
            border: 1px solid rgba(77, 150, 255, 0.3);
        }}

        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}

        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}

        .ahp-bar {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}

        .ahp-bar .label {{
            width: 200px;
            font-size: 0.9rem;
        }}

        .ahp-bar .bar-bg {{
            flex: 1;
            height: 28px;
            background: rgba(255,255,255,0.08);
            border-radius: 14px;
            overflow: hidden;
            margin: 0 12px;
        }}

        .ahp-bar .bar-fill {{
            height: 100%;
            border-radius: 14px;
            background: linear-gradient(90deg, #ffd93d, #ff6b6b);
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #1a1a2e;
            transition: width 1s ease;
        }}

        .json-block {{
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 16px;
            font-family: 'Consolas', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            color: #6bcb77;
            border: 1px solid rgba(107, 203, 119, 0.15);
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #555;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>Multi-Criteria Spatial Optimization Dashboard</h1>
    <p>Infrastructure Site Selection | Matale District, Sri Lanka | AHP + GA/PSO</p>
</div>

<div class="container">

    <!-- Summary Cards -->
    <div class="cards">
        <div class="card">
            <h3>AHP Consistency</h3>
            <div class="value ahp-color">CR = 0.0433</div>
            <div class="sub"><span class="badge badge-pass">PASS</span> Threshold &lt; 0.10</div>
        </div>
        <div class="card">
            <h3>GA Best Fitness</h3>
            <div class="value ga-color">{ga['suitability_score']:.4f}</div>
            <div class="sub">Site: ({ga['optimal_x']:.1f}, {ga['optimal_y']:.1f}) | {ga['elevation_z']:.0f}m | {ga['execution_time_seconds']:.3f}s</div>
        </div>
        <div class="card">
            <h3>PSO Best Fitness</h3>
            <div class="value pso-color">{pso['suitability_score']:.4f}</div>
            <div class="sub">Site: ({pso['optimal_x']:.1f}, {pso['optimal_y']:.1f}) | {pso['elevation_z']:.0f}m | {pso['execution_time_seconds']:.3f}s</div>
        </div>
        <div class="card">
            <h3>Winner</h3>
            <div class="value winner">PSO</div>
            <div class="sub"><span class="badge badge-winner">+{(pso['suitability_score'] - ga['suitability_score']):.4f}</span> higher fitness</div>
        </div>
    </div>

    <!-- AHP Weights -->
    <div class="section">
        <h2>AHP Criteria Weights (Eigenvector Method)</h2>
        <div class="ahp-bar">
            <span class="label">Road Proximity</span>
            <div class="bar-bg"><div class="bar-fill" style="width: 56.5%">0.5650</div></div>
        </div>
        <div class="ahp-bar">
            <span class="label">Slope Safety</span>
            <div class="bar-bg"><div class="bar-fill" style="width: 26.2%">0.2622</div></div>
        </div>
        <div class="ahp-bar">
            <span class="label">Future Growth Velocity</span>
            <div class="bar-bg"><div class="bar-fill" style="width: 11.8%">0.1175</div></div>
        </div>
        <div class="ahp-bar">
            <span class="label">Existing Amenity Buffer</span>
            <div class="bar-bg"><div class="bar-fill" style="width: 5.5%">0.0553</div></div>
        </div>
    </div>

    <!-- Benchmark Comparison Table -->
    <div class="section">
        <h2>Benchmark Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th style="color:#ff6b6b">Genetic Algorithm</th>
                    <th style="color:#4d96ff">Particle Swarm Optimization</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Best Fitness Score</td>
                    <td>{ga['suitability_score']:.6f}</td>
                    <td><strong>{pso['suitability_score']:.6f}</strong></td>
                </tr>
                <tr>
                    <td>Optimal X</td>
                    <td>{ga['optimal_x']:.4f}</td>
                    <td>{pso['optimal_x']:.4f}</td>
                </tr>
                <tr>
                    <td>Optimal Y</td>
                    <td>{ga['optimal_y']:.4f}</td>
                    <td>{pso['optimal_y']:.4f}</td>
                </tr>
                <tr>
                    <td>Elevation Z (m)</td>
                    <td>{ga['elevation_z']:.2f}</td>
                    <td>{pso['elevation_z']:.2f}</td>
                </tr>
                <tr>
                    <td>Execution Time (s)</td>
                    <td>{ga['execution_time_seconds']:.4f}</td>
                    <td><strong>{pso['execution_time_seconds']:.4f}</strong></td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- Plots -->
    <div class="two-col">
        <div class="section">
            <h2>Convergence Curves (GA vs PSO)</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{convergence_b64}" alt="Convergence Curves">
            </div>
        </div>
        <div class="section">
            <h2>Suitability Heatmap + Optimal Sites</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{heatmap_b64}" alt="Suitability Heatmap">
            </div>
        </div>
    </div>

    <!-- JSON Output -->
    <div class="section">
        <h2>JSON Output</h2>
        <div class="json-block">
            <pre>{json.dumps(results, indent=2)}</pre>
        </div>
    </div>

</div>

<div class="footer">
    Component 4: Multi-Criteria Spatial Optimization | AHP + GA/PSO | Matale District
</div>

</body>
</html>"""
    return html


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves the dashboard HTML."""

    dashboard_html: str = ""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self.dashboard_html.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress server logs


def main():
    """Build dashboard and start HTTP server."""
    print("Building dashboard...")
    html = build_dashboard_html()
    DashboardHandler.dashboard_html = html

    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"\n  Dashboard running at: {url}")
        print("  Press Ctrl+C to stop.\n")

        # Auto-open browser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        httpd.serve_forever()


if __name__ == "__main__":
    main()
