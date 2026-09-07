import asyncio
from anthropic import AsyncAnthropic

async def main():
    client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(response.content[0].text)

asyncio.run(main())
