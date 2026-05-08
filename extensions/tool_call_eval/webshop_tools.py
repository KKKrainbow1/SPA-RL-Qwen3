"""OpenAI-style tool schemas for the WebShop action space.

Two tools mirror the two underlying actions of `WebAgentTextEnv`:
  - search[keyword]  →  search(keyword=...)
  - click[value]     →  click(value=...)

These schemas are passed to the vLLM OpenAI-compatible server's chat
completions endpoint via the `tools` parameter, which Qwen3 has been
trained to consume via its native `<tool_call>` output format.
"""

WEBSHOP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the WebShop catalog for products matching the given keywords. "
                "Only available on the search page (initial page or after clicking 'Back to Search'). "
                "Use specific, descriptive keywords matching the user's instruction; multiple words "
                "improve recall."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": (
                            "Search keywords describing the desired product. "
                            "Should incorporate key attributes from the instruction "
                            "(product type, color, size, material, etc.)."
                        ),
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click on a clickable element visible in the current observation. "
                "Valid `value` examples include: product ASIN IDs (e.g. 'B09C337K8S'), "
                "page navigation ('Next >', '< Prev', 'Back to Search'), info tabs "
                "('Description', 'Features', 'Reviews', 'Attributes'), product options "
                "('color: black', 'size: 18 inch'), or 'Buy Now' to complete a purchase. "
                "The value must exactly match an element shown in the observation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The exact text/value of the element to click.",
                    }
                },
                "required": ["value"],
            },
        },
    },
]
