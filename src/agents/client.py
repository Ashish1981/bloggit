from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

import os
from dotenv import load_dotenv

load_dotenv()

model_client = OpenAIChatCompletionClient(
    # ChatGPT “5.2 Thinking” (API: GPT-5.2)
    model="gpt-5.2",  # :contentReference[oaicite:1]{index=1}

    # Use your OpenAI key (not OpenRouter)
    api_key=os.environ["OPENAI_API_KEY"],  # :contentReference[oaicite:2]{index=2}

    # Optional: increase reasoning depth (Thinking-style)
    reasoning_effort="high",  # allowed: none|minimal|low|medium|high|xhigh :contentReference[oaicite:3]{index=3} # type: ignore

    # base_url omitted -> defaults to OpenAI-hosted endpoint :contentReference[oaicite:4]{index=4}
    # Keep model_info to avoid “model_info required” errors on some AutoGen versions
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "structured_output": True,
        "family": ModelFamily.GPT_5,  # :contentReference[oaicite:5]{index=5} # type: ignore
    },
)
