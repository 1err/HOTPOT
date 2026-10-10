"""Exercise the real SDK against an in-memory HTTP transport, without API calls."""

import json

import httpx
import pytest
from PIL import Image

openai = pytest.importorskip("openai")

from hotpot.ingredients import SAMPLE_LABELS
from hotpot.openai_recognition import OpenAIRecognizer


def test_responses_sdk_serializes_vision_request_and_decodes_structured_result():
    requests = []

    def respond(request):
        assert request.url == "https://api.openai.com/v1/responses"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] == "gpt-5.4"
        assert payload["store"] is False
        assert payload["reasoning"]["effort"] == "none"
        assert payload["text"]["format"]["schema"]["properties"]["label"]["enum"] == list(SAMPLE_LABELS)
        picture = payload["input"][0]["content"][1]
        assert picture["type"] == "input_image"
        assert picture["detail"] == "high"
        assert picture["image_url"].startswith("data:image/jpeg;base64,")
        return httpx.Response(200, json={
            "id": "resp_mock", "object": "response", "created_at": 0, "status": "completed",
            "model": "gpt-5.4", "error": None, "incomplete_details": None,
            "output": [{"id": "msg_mock", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "annotations": [],
                                     "text": json.dumps({"label": "lotus_root", "uncertain": False,
                                                         "reason": "Round slice with radial holes."})}]}],
            "usage": {"input_tokens": 1100, "output_tokens": 40, "total_tokens": 1140,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with openai.OpenAI(api_key="dummy-test-key", http_client=http_client, max_retries=0) as client:
            recognizer = OpenAIRecognizer(client=client, min_interval_ms=0)
            result = recognizer.predict(Image.new("RGB", (768, 768), "white"))
    assert len(requests) == 1
    assert result["ingredient_id"] == "lotus_root"
    assert result["confidence"] is None and result["scores"] == []
    assert result["usage"]["estimated_usd"] == pytest.approx(0.00335)
