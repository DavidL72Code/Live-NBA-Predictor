"""LLM analyst: takes GameContext → structured win-probability assessment.

Uses Gemini function calling with mode=ANY to force a single structured
``submit_analysis`` call, guaranteeing validated JSON output.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from nba_winprob.analyst.context import GameContext

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert NBA analyst with deep knowledge of in-game strategy, "
    "player tendencies, and situational basketball. You are watching a live game "
    "and have access to a raw 20-feature XGBoost win-probability estimate plus the "
    "specific game context supplied in the prompt.\n\n"
    "Your role is to give an analytical judgment on the HOME team's true win "
    "probability, but use only facts explicitly present in the prompt. Do not "
    "invent injuries, trades, signings, lineups, depth, coaching tendencies, "
    "historical records, or matchup information. If a factor is not supplied, "
    "say it is unknown and do not use it to move the probability. Team venue "
    "splits and Elo are team-specific; never generalize another team's home "
    "performance.\n\n"
    "CRITICAL — probability direction:\n"
    "  analyst_probability is ALWAYS the HOME team's chance of winning.\n"
    "  If you believe the HOME team will win, set it ABOVE 0.5.\n"
    "  If you believe the AWAY team will win, set it BELOW 0.5.\n"
    "  Before submitting, do a self-check: read your own reasoning. If your "
    "  reasoning concludes the AWAY team has the edge, your analyst_probability "
    "  MUST be below 0.5. If your reasoning concludes the HOME team has the edge, "
    "  it MUST be above 0.5. A probability above 0.5 with reasoning that favors "
    "  the away team is a contradiction — fix the number, not the reasoning.\n\n"
    "You must call submit_analysis with your complete assessment. "
    "Adjust the probability if context warrants — don't just echo the model. "
    "Be specific and concise."
)

ANALYST_TOOL_SCHEMA = {
    "name": "submit_analysis",
    "description": (
        "Submit your structured win-probability analysis for the current game state. "
        "Call this exactly once with your complete assessment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_probability": {
                "type": "number",
                "description": "The XGBoost model's home-team win probability (0–1), as provided.",
            },
            "analyst_probability": {
                "type": "number",
                "description": (
                    "HOME team win probability (0–1). >0.5 means home team wins; <0.5 means away team wins. "
                    "Must be consistent with your reasoning. Clamp to [0.01, 0.99]."
                ),
            },
            "direction": {
                "type": "string",
                "enum": ["higher", "lower", "accurate"],
                "description": "Whether your estimate is higher, lower, or consistent with the model.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Your confidence in the adjusted estimate.",
            },
            "headline": {
                "type": "string",
                "description": "One broadcast-style sentence capturing the game situation (≤ 25 words).",
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2–5 specific factors driving your probability adjustment.",
            },
            "reasoning": {
                "type": "string",
                "description": "2–3 sentences explaining why you agree or disagree with the model.",
            },
        },
        "required": [
            "model_probability",
            "analyst_probability",
            "direction",
            "confidence",
            "headline",
            "key_factors",
            "reasoning",
        ],
    },
}


class AnalystOutput(BaseModel):
    model_probability: float
    analyst_probability: float
    direction: str
    confidence: str
    headline: str
    key_factors: list[str]
    reasoning: str


class LLMAnalyst:
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite") -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._types = types

    def analyze(self, context: GameContext) -> AnalystOutput:
        """Run the analyst on a GameContext and return a structured output."""
        from google.genai import types

        prompt = context.to_prompt_text()
        logger.debug("analyst prompt:\n%s", prompt)

        tool = types.Tool(function_declarations=[ANALYST_TOOL_SCHEMA])
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["submit_analysis"],
                )
            ),
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )

        # Extract the function call from the first candidate
        fn_call = None
        for part in response.candidates[0].content.parts:
            if part.function_call and part.function_call.name == "submit_analysis":
                fn_call = part.function_call
                break

        if fn_call is None:
            raise RuntimeError(
                f"Gemini did not call submit_analysis "
                f"(finish_reason={response.candidates[0].finish_reason})"
            )

        args = dict(fn_call.args)
        # Ensure key_factors is a plain list (Gemini may return a MapComposite)
        if "key_factors" in args and not isinstance(args["key_factors"], list):
            args["key_factors"] = list(args["key_factors"])

        return AnalystOutput(**args)
