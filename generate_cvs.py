import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from utils.cv import generate_cv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Read env vars as ints (with sensible defaults)
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
CV_TO_GENERATE = int(os.getenv("CV_TO_GENERATE", "30"))


async def generate_one(semaphore: asyncio.Semaphore, idx: int) -> Path:
    """Generate a single CV PDF, respecting the concurrency semaphore."""
    async with semaphore:
        pdf_path = await generate_cv()
        return pdf_path


async def main():
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        asyncio.create_task(generate_one(semaphore, i))
        for i in range(1, CV_TO_GENERATE + 1)
    ]

    ok_paths = []
    errors = []

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating CVs"):
        try:
            result = await future
            ok_paths.append(result)
        except Exception as e:
            errors.append(e)

    out_dir = ok_paths[0].parent if ok_paths else Path.cwd()

    logging.info("✅ Generated %d/%d CVs in %s",
                 len(ok_paths), CV_TO_GENERATE, out_dir.resolve())

    if errors:
        logging.warning("⚠️ %d task(s) failed. First error: %r",
                        len(errors), errors[0])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
