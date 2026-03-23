"""Quick test of the Anthropic API key."""
import os
import anthropic
from anthropic import Anthropic

key = os.environ.get("ANTHROPIC_API_KEY", "")
print(f"Key: {key}")
print(f"Key length: {len(key)}")

# exit(1)

# try:
#     client = anthropic.Anthropic(api_key=key)
#     resp = client.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=20,
#         messages=[{"role": "user", "content": "Say hello"}],
#     )
#     print(f"Success: {resp.content[0].text}")
# except Exception as e:
#     print(f"Error: {e}")


try: 
    client = Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",  # or claude-opus-4-6, claude-haiku-4-5-20251001
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude"}
        ],
    )

    print(message.content[0].text)
except Exception as e:
    print(f"Error: {e}")