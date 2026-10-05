"""
Example: Using OpenAI-compatible API providers (OpenRouter, Together, etc.)

This demonstrates how to use the modular LLM backend with any OpenAI-compatible API.
"""

import asyncio
import json

from dotenv import load_dotenv
from playwright.async_api import Page, async_playwright

from hcaptcha_challenger import AgentConfig, AgentV, CaptchaResponse
from hcaptcha_challenger.utils import SiteKey

load_dotenv()


async def challenge_with_openrouter(page: Page) -> AgentV:
    """Automates the process of solving an hCaptcha challenge using OpenRouter.

    Configure via .env:
        LLM_PROVIDER=openai
        LLM_API_KEY=sk-or-v1-...
        LLM_BASE_URL=https://openrouter.ai/api/v1
        LLM_MODEL=anthropic/claude-3.5-sonnet
    """

    agent_config = AgentConfig()
    agent = AgentV(page=page, agent_config=agent_config)

    await agent.robotic_arm.click_checkbox()
    await agent.wait_for_challenge()

    return agent


async def challenge_with_local_llm(page: Page) -> AgentV:
    """Example using a local LLM (e.g., Ollama with vision support).

    Configure via .env:
        LLM_PROVIDER=openai
        LLM_API_KEY=ollama
        LLM_BASE_URL=http://localhost:11434/v1
        LLM_MODEL=llama3.2-vision
    """

    agent_config = AgentConfig()
    agent = AgentV(page=page, agent_config=agent_config)

    await agent.robotic_arm.click_checkbox()
    await agent.wait_for_challenge()

    return agent


async def challenge_with_openai(page: Page) -> AgentV:
    """Example using OpenAI directly.

    Configure via .env:
        LLM_PROVIDER=openai
        LLM_API_KEY=sk-...
        LLM_MODEL=gpt-4o
    """

    agent_config = AgentConfig()
    agent = AgentV(page=page, agent_config=agent_config)

    await agent.robotic_arm.click_checkbox()
    await agent.wait_for_challenge()

    return agent


# noinspection DuplicatedCode
async def main():
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="tmp/.cache/user_data",
            headless=False,
            locale="en-US",
        )

        page = await context.new_page()
        await page.goto(SiteKey.as_site_link(SiteKey.epic))

        # Configure your provider in .env file, then:
        agent: AgentV = await challenge_with_openrouter(page)

        if agent.cr_list:
            cr: CaptchaResponse = agent.cr_list[-1]
            print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
