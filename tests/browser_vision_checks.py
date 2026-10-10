"""Browser checks for cloud recognition controls with all API calls mocked.

Run: .venv/bin/python tests/browser_vision_checks.py
Requires Playwright and Chromium (optional CHROMIUM_PATH override). No OpenAI key or paid API request is used;
these checks test UI behavior and timer regressions, not recognition accuracy.
"""
from __future__ import annotations
import io, json, os, shutil, threading, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright, expect
ROOT = Path(__file__).resolve().parents[1]
LABELS = ['beef_rolls', 'crab_sticks', 'napa_cabbage', 'lotus_root', 'soft_tofu', 'wide_noodles', 'udon_noodles', 'shiitake_mushrooms', 'beef_meatballs']
NAMES = ['肥牛卷', '蟹棒', '大白菜', '藕片', '嫩豆腐', '宽粉', '乌冬面', '香菇', '牛肉丸']
DURATIONS = [15, 90, 150, 300, 300, 480, 180, 300, 480]
CONFIG = {'ingredients': [{'id': id, 'name': name, 'en': id, 'duration_seconds': seconds} for id, name, seconds in zip(LABELS, NAMES, DURATIONS)], 'session_id': 'ui-check'}

class QuietHandler(SimpleHTTPRequestHandler):

    def log_message(self, *args):
        pass
def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = {'provider': 'openai', 'cloud': True, 'model': 'gpt-5.4', 'state': 'ready', 'label': 'GPT-5.4', 'message': 'Cloud vision available', 'min_interval_ms': 3000, 'request_limit': 100, 'attempts_remaining': 100, 'usage': {'attempts': 0, 'completed': 0, 'estimated_usd': 0}}
    calls = []
    fail_next = [False]
    unknown_next = [False]
    background_next = [False]
    errors = []

    def handle(route):
        request = route.request
        path = request.url.split(server_url)[-1]
        payload = {}
        if path == '/api/config':
            payload = CONFIG
        elif path == '/api/dataset':
            payload = {'counts': {}, 'total': 0}
        elif path == '/api/model':
            payload = model
        elif path == '/api/model/provider':
            new = request.post_data_json
            provider = new['provider']
            model.update(provider=provider, cloud=provider == 'openai', model=new.get('model', 'local'), state='ready', label=new.get('model', 'Local test model'))
            payload = model
        elif path == '/api/model/reload':
            payload = model
        elif path == '/api/recognize':
            calls.append(time.monotonic())
            model['usage']['attempts'] += 1
            model['attempts_remaining'] = 100 - model['usage']['attempts']
            if fail_next[0]:
                fail_next[0] = False
                route.fulfill(status=503, json={'detail': 'Test upstream error'})
                return
            model['usage']['completed'] += 1
            model['usage']['estimated_usd'] += 0.0025
            if model['cloud']:
                payload = {'ingredient_id': None if background_next[0] else 'beef_rolls', 'uncertain': unknown_next[0] or background_next[0], 'confidence': None, 'scores': [], 'reason': 'No supported ingredient is visible.' if background_next[0] else 'The visible food is ambiguous.' if unknown_next[0] else 'Thin rolled slices of beef are visible.', 'inference_ms': 12, 'usage': model['usage'], 'provider': 'openai', 'model': model['model']}
                unknown_next[0] = False
                background_next[0] = False
            else:
                payload = {'ingredient_id': 'beef_rolls', 'uncertain': False, 'confidence': 0.94, 'scores': [{'id': 'beef_rolls', 'score': 0.94}], 'inference_ms': 12}
        route.fulfill(json=payload)
    server_url = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH') or shutil.which('chromium') or shutil.which('chromium-browser') or shutil.which('google-chrome') or (str(Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')) if Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome').exists() else None), headless=True, args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
            context = browser.new_context(permissions=['camera'], viewport={'width': 1440, 'height': 1100})
            page = context.new_page()
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.route('**/api/**', handle)
            page.goto(server_url + '/static/index.html')
            expect(page.locator('#recognition-model')).to_have_value('openai:gpt-5.4')
            expect(page.locator('#auto-recognize')).not_to_be_checked()
            expect(page.locator('#recognition-caption')).to_contain_text('API billed separately')
            assert len(calls) == 0
            image = Image.new('RGB', (1600, 1200), (160, 60, 80))
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            page.locator('#image-upload').set_input_files({'name': 'ingredient.png', 'mimeType': 'image/png', 'buffer': buf.getvalue()})
            expect(page.locator('#source-badge')).to_have_text('UPLOADED IMAGE')
            page.wait_for_timeout(1100)
            assert len(calls) == 0, 'Cloud upload generated a paid request without explicit recognition'
            page.locator('#recognize-once').click()
            expect(page.locator('#prediction-name')).to_have_text('肥牛卷')
            expect(page.locator('#prediction-detail')).to_contain_text('Thin rolled slices')
            expect(page.locator('#confidence-badge')).to_have_text('RECOGNIZED')
            expect(page.locator('#frame-result-score')).to_have_text('')
            expect(page.locator('#score-bars')).to_be_hidden()
            expect(page.locator('#model-usage')).to_contain_text('1 / 100 requests attempted')
            expect(page.locator('#model-usage')).to_contain_text('$0.0025')
            expect(page.locator('#recognize-once')).to_be_disabled()
            page.wait_for_timeout(3100)
            expect(page.locator('#recognize-once')).to_be_enabled()
            page.locator('[data-ingredient-id="crab_sticks"]').click()
            page.locator('#recognize-once').click()
            expect(page.locator('#selected-name')).to_have_text('蟹棒')
            page.locator('#recognition-model').select_option('openai:gpt-5.4-mini')
            expect(page.locator('#model-label')).to_have_text('gpt-5.4-mini')
            expect(page.locator('#auto-recognize')).not_to_be_checked()
            page.wait_for_timeout(3100)
            assert len(calls) == 2, 'Switching cloud models generated an inference'
            unknown_next[0] = True
            page.locator('#recognize-once').click()
            expect(page.locator('#confidence-badge')).to_have_text('AMBIGUOUS')
            expect(page.locator('#selected-name')).to_have_text('蟹棒')
            page.wait_for_timeout(3100)
            background_next[0] = True
            page.locator('#recognize-once').click()
            expect(page.locator('#confidence-badge')).to_have_text('UNKNOWN')
            expect(page.locator('#prediction-detail')).to_contain_text('No supported ingredient')
            expect(page.locator('#selected-name')).to_have_text('蟹棒')
            page.locator('#reload-model').click()
            page.wait_for_timeout(500)
            assert len(calls) == 4, 'Reload generated a cloud inference'
            page.locator('#start-camera').click()
            expect(page.locator('#source-badge')).to_have_text('CAMERA LIVE', timeout=10000)
            page.wait_for_timeout(5300)
            assert len(calls) == 4, 'Camera start generated a cloud inference without live opt-in'
            page.locator('.recognition-controls .toggle').click()
            page.wait_for_timeout(4500)
            assert len(calls) == 4, 'Live interval shorter than advertised five seconds'
            expect(page.locator('#prediction-name')).to_have_text('肥牛卷', timeout=2500)
            assert len(calls) == 5
            page.wait_for_timeout(5200)
            assert len(calls) == 6
            assert calls[-1] - calls[-2] >= 4.9
            fail_next[0] = True
            expect(page.locator('#auto-recognize')).not_to_be_checked(timeout=6500)
            expect(page.locator('#global-message')).to_contain_text('Live recognition paused')
            failed_count = len(calls)
            page.wait_for_timeout(5200)
            assert len(calls) == failed_count, 'Cloud failure triggered another billable retry'
            page.locator('#recognition-model').select_option('local')
            expect(page.locator('#auto-recognize')).to_be_checked()
            expect(page.locator('#confidence-badge')).to_have_text('0.940 MATCH SCORE', timeout=5000)
            expect(page.locator('#score-bars')).to_be_visible()
            page.locator('#recognition-model').select_option('openai:gpt-5.4')
            expect(page.locator('#auto-recognize')).not_to_be_checked()
            cloud_count = len(calls)
            page.wait_for_timeout(1100)
            assert len(calls) == cloud_count, 'Switch to cloud retained live recognition'
            page.locator('#stop-camera').click()
            for id, duration in zip(LABELS, DURATIONS):
                page.locator(f'[data-ingredient-id="{id}"]').click()
                expect(page.locator('#duration-input')).to_have_value(str(duration))
                page.locator('#start-timer').click()
            expect(page.locator('.timer-item')).to_have_count(9)
            saved = page.evaluate("JSON.parse(localStorage.getItem('hotpot-lab-v1'))")
            for id, duration in zip(LABELS, DURATIONS):
                timer = saved['timers'][id]
                assert timer['duration'] == duration and timer['deadline'] - timer['started_at'] == duration * 1000
            page.reload()
            expect(page.locator('.timer-item')).to_have_count(9)
            expect(page.locator('#auto-recognize')).not_to_be_checked()
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert errors == [], errors
            browser.close()
        print('Cloud UI checks passed: manual-only cloud startup/upload/reload/switch; exact model choices; reasons without invented confidence;3s manual cooldown;5s opt-in live;pause on cloud error;local behavior;all9 timers persist;mobile layout. API calls mocked; recognition accuracy not tested.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
