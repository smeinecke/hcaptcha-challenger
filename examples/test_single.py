import asyncio
from datetime import datetime
from pathlib import Path

from playwright._impl._errors import TargetClosedError
from playwright.async_api import async_playwright

from hcaptcha_challenger import AgentConfig, AgentV
from hcaptcha_challenger.helper import inject_mouse_visualizer_global
from hcaptcha_challenger.utils import SiteKey


async def main():
    record_dir = Path("tmp/.cache/record") / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    record_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="tmp/.cache/user_data",
            headless=False,
            locale="en-US",
            record_video_dir=str(record_dir),
            record_video_size={"width": 1440, "height": 960},
        )
        try:
            page = await context.new_page()

            # Navigate to easy demo site
            await page.goto(SiteKey.as_site_link(SiteKey.user_easy))
            await inject_mouse_visualizer_global(page)

            # Initialize agent
            agent_config = AgentConfig()
            agent = AgentV(page=page, agent_config=agent_config)

            # Click checkbox to trigger challenge
            await agent.robotic_arm.click_checkbox()

            # Wait for challenge and solve it
            result = await agent.wait_for_challenge()

            print(f"\nResult: {result}")

            if agent.cr_list:
                print("Success! Captcha response obtained.")

            # Keep browser open briefly so the live run can be observed.
            await asyncio.sleep(5)
        finally:
            try:
                await context.close()
            except TargetClosedError:
                pass

    videos = sorted(record_dir.rglob("*.webm")) + sorted(record_dir.rglob("*.mp4"))
    print(f"Video directory: {record_dir.resolve()}")
    if videos:
        for video in videos:
            print(f"Video file: {video.resolve()}")
    else:
        print("No video files were produced.")


if __name__ == "__main__":
    asyncio.run(main())
