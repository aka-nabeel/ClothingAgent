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

            # Language leakage check for URDU_SCRIPT mode:
            is_urdu_leak = (
                language == LanguageMode.URDU_SCRIPT
                and not re.search(r"[\u0600-\u06FF]", response)
            )

            if DEVANAGARI_RE.search(response) or is_urdu_leak:
                retry_prompt = (
                    f"{prompt}\n\nHARD SAFETY RULE: The customer's session language is strictly set to URDU SCRIPT (اردو). "
                    f"Your previous response was NOT in Urdu script. You MUST write your entire response 100% in clear, polite Urdu script (اردو) right now!"
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

            if language == LanguageMode.URDU_SCRIPT and not re.search(r"[\u0600-\u06FF]", response):
                return self._fallback_response(language, user_message, runtime_context)

            sanitized = self._sanitize_metadata_if_not_requested(user_message, response.strip())
            return sanitized
        except Exception:
            return self._fallback_response(language, user_message, runtime_context)

    @staticmethod
    def _sanitize_metadata_if_not_requested(user_message: str, response: str) -> str:
        """Enforce business rule: Strip prose metadata dumps unless customer explicitly asked for product details."""
        msg_lower = user_message.lower()
        explicit_detail_keywords = {
            "detail", "details", "fabric", "material", "specification", "specifications",
            "more info", "tell me about", "tafsilat", "تفصیلات", "معلومات", "kya fabric hai"
        }
        if any(w in msg_lower for w in explicit_detail_keywords):
            return response

        lines = []
        for line in response.split("\n"):
            if re.search(r"PKR\s*[\d,]+", line) and ("Cotton" in line or "Linen" in line or "fit" in line or "In-stock" in line or "discount" in line or "welcome" in line):
                match = re.search(r"(\d+\.\s*|- \s*)?\*?\*?([^*]+?)\*?\*?\s*-\s*.*? (PKR\s*[\d,]+)", line)
                if match:
                    prefix = match.group(1) or "- "
                    title = match.group(2).strip(" -*")
                    price = match.group(3)
                    lines.append(f"{prefix}{title} - {price}")
                    continue
            lines.append(line)
        return "\n".join(lines)

    def _fallback_response(self, language: LanguageMode, user_message: str, runtime_context: Mapping[str, Any]) -> str:
        prods = runtime_context.get("displayed_products", [])
        if language == LanguageMode.ROMAN_URDU:
            if prods:
                lines = ["Okay, yeh humare paas top options hain:\n"]
                for i, p in enumerate(prods, 1):
                    name = p.get("product_name") or p.get("name", "Product")
                    price = p.get("final_price") or p.get("price", "")
                    lines.append(f"{i}. {name} - PKR {price}")
                lines.append("\nAap in mein se kis product ko cart mein add karna chahte hain ya mazeed options dekhna chahte hain?")
                return "\n".join(lines)
            order = runtime_context.get("placed_order")
            if order:
                num = order.get("order_number", "")
                return f"Aap ka order #{num} successfully place ho gaya hai! Northstar se shopping karne ka shukriya."
            return (
                "Northstar mein humare paas menswear ki ek behtareen aur vast collection available hai:\n\n"
                "- Shirts (Formal, Casual, Linen, Oxford)\n"
                "- T-Shirts (Crew Neck, Graphic, Polo, Compression)\n"
                "- Pants & Trousers (Chinos, Jeans, Formal Trousers, Cargo)\n"
                "- Traditional Wear (Kurta, Shalwar Kameez)\n"
                "- Outerwear (Jackets, Hoodies, Bombers)\n\n"
                "Aap kis tarah ke kapray ya subcategory dekhna chahte hain taakay main aap ki behtar madad kar sakoon?"
            )
        elif language == LanguageMode.URDU_SCRIPT:
            if prods:
                lines = ["یہ ہمارے پاس چند بہترین آپشنز ہیں:\n"]
                for i, p in enumerate(prods, 1):
                    name = p.get("product_name") or p.get("name", "Product")
                    price = p.get("final_price") or p.get("price", "")
                    lines.append(f"{i}. {name} - PKR {price}")
                lines.append("\nآپ ان میں سے کس پروڈکٹ کو کارٹ میں شامل کرنا چاہتے ہیں یا مزید آپشنز دیکھنا چاہتے ہیں؟")
                return "\n".join(lines)
            order = runtime_context.get("placed_order")
            if order:
                num = order.get("order_number", "")
                return f"آپ کا آرڈر #{num} کامیابی سے مکمل ہو گیا ہے۔ نارتھ اسٹار کا انتخاب کرنے کا شکریہ!"
            return (
                "نارتھ اسٹار میں ہمارے پاس مردانہ ملبوسات کا ایک وسیع اور بہترین کلیکشن موجود ہے:\n\n"
                "- شرٹس (فارمل، کیژول، لتن، آکسفورڈ)\n"
                "- ٹی شرٹس (کرو نیک، گرافک، پولو، کمپریشن)\n"
                "- پینٹس اور ٹراؤزرز (چینو، جینز، فارمل، کارگو)\n"
                "- روایتی ملبوسات (کرتا، شلوار قمیض)\n"
                "- آؤٹر ویئر (جیکٹس، ہوڈیز، بمبر)\n\n"
                "آپ کس سب کیٹیگری میں سے پروڈکٹس دیکھنا چاہتے ہیں تاکہ میں آپ کی بہترین رہنمائی کر سکوں؟"
            )
        else:
            if prods:
                lines = ["Okay, here are a few top picks for you:\n"]
                for i, p in enumerate(prods, 1):
                    name = p.get("product_name") or p.get("name", "Product")
                    price = p.get("final_price") or p.get("price", "")
                    lines.append(f"{i}. {name} - PKR {price}")
                lines.append("\nWhich product catches your attention and would you like to add to cart, or would you like more options?")
                return "\n".join(lines)
            order = runtime_context.get("placed_order")
            if order:
                num = order.get("order_number", "")
                return f"Your order #{num} has been successfully placed! Thank you for shopping with Northstar."
            return (
                "Northstar carries a vast collection of premium menswear across several key categories:\n\n"
                "- Shirts (Formal, Casual, Linen, Oxford)\n"
                "- T-Shirts (Crew Neck, Graphic, Polo, Compression)\n"
                "- Pants & Trousers (Chinos, Jeans, Formal Trousers, Cargo)\n"
                "- Traditional Wear (Kurta, Shalwar Kameez)\n"
                "- Outerwear (Jackets, Hoodies, Bombers)\n\n"
                "Please let me know which category or product type you would like to explore today so I can assist you!"
            )
