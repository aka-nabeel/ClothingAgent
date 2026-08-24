"""Response formatting and language guardrails for Fitzy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .state import LanguageMode
from ..llm.client import LLMClient, LLMResponseError
from ..llm.prompts import build_response_system_prompt

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


class ResponseGuard:
    """Validate and, when necessary, regenerate unsafe language output."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def generate(
        self,
        *,
        language: LanguageMode,
        user_message: str,
        runtime_context: Mapping[str, Any],
    ) -> str:
        """Generate a customer-facing response and reject Devanagari output."""

        try:
            context_text = str(dict(runtime_context))
            prompt = build_response_system_prompt(language=language.value)
            response = await self._llm.generate_text(
                system_prompt=prompt,
                user_message=(
                    f"Customer message:\n{user_message}\n\n"
                    f"Authoritative runtime context and tool results:\n{context_text}"
                ),
            )

            # Currency post-processing: replace accidental $ with PKR
            response = re.sub(r"\$(\s*\d+)", r"PKR \1", response)

            if DEVANAGARI_RE.search(response):
                retry_prompt = (
                    f"{prompt}\n\nHARD SAFETY RULE: Your previous response contained Devanagari. "
                    f"Generate the same useful answer again using {'Urdu script' if language == LanguageMode.URDU_SCRIPT else 'Roman Urdu' if language == LanguageMode.ROMAN_URDU else 'English'} only."
                )
                response = await self._llm.generate_text(
                    system_prompt=retry_prompt,
                    user_message=(
                        f"Customer message:\n{user_message}\n\n"
                        f"Authoritative runtime context and tool results:\n{context_text}"
                    ),
                )
                response = re.sub(r"\$(\s*\d+)", r"PKR \1", response)

            if DEVANAGARI_RE.search(response):
                raise LLMResponseError("Unsafe Devanagari/Hindi output detected after regeneration")

            return response.strip()
        except Exception:
            return self._fallback_response(language, user_message, runtime_context)

    def _fallback_response(self, language: LanguageMode, user_message: str, runtime_context: Mapping[str, Any]) -> str:
        prods = runtime_context.get("displayed_products", [])
        if language == LanguageMode.ROMAN_URDU:
            if prods:
                lines = ["Yeh aap ke liye humare store se available options hain:\n"]
                for p in prods:
                    name = p.get("product_name") or p.get("name", "Product")
                    price = p.get("final_price") or p.get("price", "")
                    lines.append(f"- **{name}** (PKR {price})")
                return "\n".join(lines)
            order = runtime_context.get("placed_order")
            if order:
                num = order.get("order_number", "")
                return f"Aap ka order #{num} successfully place ho gaya hai! Northstar se shopping karne ka shukriya."
            return "Main aap ki shopping mein madad kar sakta hoon. Aap kya dekhna chahenge?"
        elif language == LanguageMode.URDU_SCRIPT:
            if prods:
                lines = ["یہ ہمارے اسٹور میں دستیاب آپشنز ہیں:\n"]
                for p in prods:
                    name = p.get("product_name") or p.get("name", "Product")
                    price = p.get("final_price") or p.get("price", "")
                    lines.append(f"- **{name}** (PKR {price})")
                return "\n".join(lines)
            order = runtime_context.get("placed_order")
            if order:
                num = order.get("order_number", "")
                return f"آپ کا آرڈر #{num} کامیابی سے مکمل ہو گیا ہے۔ نارتھ اسٹار کا انتخاب کرنے کا شکریہ!"
            return "میں آپ کی خریداری میں مدد کر سکتا ہوں۔ آپ کیا تلاش کر رہے ہیں؟"
        else:
            if prods:
                lines = ["Here are the available options found in our store catalog:\n"]
                for p in prods:
                    name = p.get("product_name") or p.get("name", "Product")
                    price = p.get("final_price") or p.get("price", "")
                    lines.append(f"- **{name}** (PKR {price})")
                return "\n".join(lines)
            order = runtime_context.get("placed_order")
            if order:
                num = order.get("order_number", "")
                return f"Your order #{num} has been successfully placed! Thank you for shopping with Northstar."
            return "I am here to assist with your shopping. What are you looking for today?"
