import os
import base64
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

async def image_generation(prompt: str) -> str | None:
    # 1) Use GPT-5.2 (ChatGPT 5.2 Thinking-style) to refine the image prompt
    prompt_refine = await client.responses.create( # pyright: ignore[reportAttributeAccessIssue]
        model="gpt-5.2",
        input=(
            "Rewrite the following into a high-quality image-generation prompt. "
            "Be specific about subject, setting, composition, lighting, style, and mood. "
            "Avoid mentioning watermarks/logos. Return ONLY the final prompt text.\n\n"
            f"USER_PROMPT:\n{prompt}"
        ),
        reasoning={"effort": "high"},  # increase “thinking” depth :contentReference[oaicite:3]{index=3}
    )
    refined_prompt = (prompt_refine.output_text or "").strip()
    if not refined_prompt:
        return None

    # 2) Generate the image with OpenAI Image API (GPT Image family)
    img = await client.images.generate(
        model="gpt-image-1.5",
        prompt=refined_prompt,
        n=1,
        size="1024x1024",
    )

    if not img or not img.data:
        return None

    # GPT Image models return base64 (b64_json), not a URL :contentReference[oaicite:4]{index=4}
    b64 = img.data[0].b64_json
    if not b64:
        return None

    # Return a data URL (handy for browsers / frontends)
    return f"data:image/png;base64,{b64}"
