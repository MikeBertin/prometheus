"""
PROMETHEUS — capture.py

Captures the repo's social assets from the real running page:
  web/og.png   — 1200x630 OG card, rendered from web/og.html
  web/demo.gif — the live terminal actually generating (frames -> ffmpeg)

Usage:  .venv/bin/python scripts/capture.py   (needs the local server on :8123
        and ffmpeg on PATH; `make web` first so the wasm build is current)
"""
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

BASE = "http://localhost:8123"
ROOT = Path(__file__).resolve().parent.parent
FRAME_MS = 180          # capture cadence for the gif
MAX_FRAMES = 60
HOLD_FRAMES = 6         # freeze on the finished speech before the loop restarts


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()

        # ── OG card: fixed-size static render ────────────────────────────
        page = await browser.new_page(viewport={"width": 1200, "height": 630})
        await page.goto(f"{BASE}/og.html")
        await page.wait_for_timeout(400)          # let fonts/gradients settle
        await page.screenshot(path=str(ROOT / "web" / "og.png"))
        print("wrote web/og.png")
        await page.close()

        # ── demo.gif: drive a real generation, film the terminal ─────────
        page = await browser.new_page(viewport={"width": 1080, "height": 800})
        await page.goto(BASE)
        # ready = the load-time benchmark finished and the button re-enabled
        await page.wait_for_selector("#go:not([disabled])", timeout=60_000)
        # let the auto-fired first generation finish so we film a clean run
        await page.wait_for_function(
            "document.getElementById('go').textContent.includes('generate')",
            timeout=30_000)

        await page.fill("#prompt", "ROMEO:")
        crt = page.locator(".crt")
        frames_dir = Path(tempfile.mkdtemp(prefix="prom_frames_"))
        await page.click("#go")

        n = 0
        done_at = None
        while n < MAX_FRAMES:
            await crt.screenshot(path=str(frames_dir / f"f_{n:03d}.png"))
            n += 1
            finished = await page.evaluate(
                "document.getElementById('go').textContent.includes('generate')")
            if finished and done_at is None:
                done_at = n
            if done_at is not None and n - done_at >= HOLD_FRAMES:
                break
            await page.wait_for_timeout(FRAME_MS)
        print(f"captured {n} frames")
        await browser.close()

    # assemble with a tight palette; scale to 720 wide like the sibling repos
    fps = round(1000 / FRAME_MS)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps), "-i", str(frames_dir / "f_%03d.png"),
        "-vf", (f"fps={fps},scale=720:-1:flags=lanczos,"
                "split[s0][s1];[s0]palettegen=max_colors=96[p];"
                "[s1][p]paletteuse=dither=bayer:bayer_scale=5"),
        str(ROOT / "web" / "demo.gif"),
    ], check=True)
    size = (ROOT / "web" / "demo.gif").stat().st_size
    print(f"wrote web/demo.gif ({size/1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
