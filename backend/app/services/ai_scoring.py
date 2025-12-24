"""Dual-model AI scoring service using Claude and OpenAI"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.models.schemas import AIEvaluation, FullAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ModelScore:
    """Individual model scoring result"""

    score: float
    recommendation: str
    risk_level: str
    reasoning: str
    model_name: str
    trading_tags: list[str]


SCORING_PROMPT = """You are a professional crypto trading analyst evaluating a trader for copy-trading suitability.

Analyze the following trader metrics and provide:
1. A score from 0-100 (higher = better for copy-trading)
2. A recommendation: one of "strong_follow", "follow", "neutral", "avoid", "strong_avoid"
3. A risk_level: one of "low", "medium", "high", "extreme"
4. A brief reasoning in Chinese (1-2 sentences)
5. Trading style tags (2-4 tags) in Chinese, choose from these categories:
   - Capital size: 小资金(<$10K), 中等资金($10K-$100K), 大资金(>$100K)
   - Trade frequency: 高频(>5 trades/day), 中频(1-5 trades/day), 低频(<1 trade/day)
   - Holding style: 短线(<1h avg), 中线(1h-24h avg), 长线(>24h avg)
   - Direction bias: 做多为主(L/S>1.5), 做空为主(L/S<0.67), 多空均衡
   - Risk style: 激进(high DD), 稳健(low DD), 马丁(high trade count + losses)
   - Asset focus: 主流币(BTC/ETH heavy), 山寨币(altcoin heavy)

TRADER METRICS:
- Address: {address}
- Account Value: ${account_value} (CURRENT value, may differ from period start)
- 7-Day ROE: {roe_7d}%
- 30-Day ROE: {roe_30d}%
- 90-Day ROE: {roe_90d}%
- Win Rate: {win_rate}%
- Profit Factor: {profit_factor} (capped at 100 for display)
- Max Drawdown: {max_drawdown}%
- Total Trades: {total_trades}
- Trade Frequency: {trade_frequency} trades/day
- Avg Holding Time: {avg_holding_time} hours
- Long/Short Ratio: {long_short_ratio}

BACKTEST RESULTS (simulated with $10,000 fixed capital, proportional scaling):
- 7-Day: ROE={bt_7d_roe}%, Max DD={bt_7d_dd}%
- 30-Day: ROE={bt_30d_roe}%, Max DD={bt_30d_dd}%
- 90-Day: ROE={bt_90d_roe}%, Max DD={bt_90d_dd}%

IMPORTANT NOTES ON METRICS:
1. ROE vs Backtest ROE difference is EXPECTED and NOT a data error:
   - "ROE" = Actual PnL / Current Account Value (trader's real performance)
   - "Backtest ROE" = Simulated copy-trading result with fixed $10K capital
   - Large differences occur when trader's account value changed significantly during the period
   - Example: If trader started with $10K, made $50K profit, now has $60K account:
     - Actual ROE would be calculated as $50K / $60K = 83%
     - But true period return was $50K / $10K = 500%
   - Backtest shows what YOU would get by copy-trading with $10K
2. Profit Factor capped at 100 (values above indicate near-zero losses, not data error)
3. Win Rate of 100% with few trades may indicate insufficient sample size

SCORING GUIDELINES:
- Score 80+: Consistently profitable, low drawdown (<15%), high win rate (>55%), stable performance
- Score 60-79: Generally profitable, moderate risk, some volatility
- Score 40-59: Mixed results, higher risk, proceed with caution
- Score 20-39: Poor performance or high risk indicators
- Score 0-19: Severe losses, extreme drawdown, or insufficient data

Respond ONLY with valid JSON in this exact format:
{{"score": <number>, "recommendation": "<string>", "risk_level": "<string>", "reasoning": "<Chinese text>", "trading_tags": ["tag1", "tag2", "tag3"]}}"""


def _fmt(val, default: str = "N/A", fmt_str: str = "{}") -> str:
    """Format a value or return default"""
    if val is None:
        return default
    try:
        return fmt_str.format(val)
    except (ValueError, TypeError):
        return default


class AIScoringService:
    """Dual-model AI scoring using Claude and OpenAI"""

    def __init__(self):
        self.settings = get_settings()
        self.timeout = self.settings.ai_scoring_timeout

    def _format_prompt(self, result: FullAnalysisResult) -> str:
        """Format the scoring prompt with trader data"""
        avg_holding_hours = None
        if result.avg_holding_time is not None:
            avg_holding_hours = result.avg_holding_time / 3600

        return SCORING_PROMPT.format(
            address=result.address,
            account_value=_fmt(result.account_value, "N/A", "{:,.2f}"),
            roe_7d=_fmt(result.roe_7d, "N/A", "{:.1f}"),
            roe_30d=_fmt(result.roe_30d, "N/A", "{:.1f}"),
            roe_90d=_fmt(result.roe_90d, "N/A", "{:.1f}"),
            win_rate=_fmt(result.win_rate, "N/A", "{:.1f}"),
            profit_factor=_fmt(result.profit_factor, "N/A", "{:.2f}"),
            max_drawdown=_fmt(result.max_drawdown_pct, "N/A", "{:.1f}"),
            total_trades=result.total_trades or "N/A",
            trade_frequency=_fmt(result.trade_frequency, "N/A", "{:.1f}"),
            avg_holding_time=_fmt(avg_holding_hours, "N/A", "{:.1f}"),
            long_short_ratio=_fmt(result.long_short_ratio, "N/A", "{:.2f}"),
            bt_7d_roe=_fmt(result.backtest_7d.roe if result.backtest_7d else None, "N/A", "{:.1f}"),
            bt_7d_dd=_fmt(
                result.backtest_7d.max_drawdown_pct if result.backtest_7d else None, "N/A", "{:.1f}"
            ),
            bt_30d_roe=_fmt(
                result.backtest_30d.roe if result.backtest_30d else None, "N/A", "{:.1f}"
            ),
            bt_30d_dd=_fmt(
                result.backtest_30d.max_drawdown_pct if result.backtest_30d else None,
                "N/A",
                "{:.1f}",
            ),
            bt_90d_roe=_fmt(
                result.backtest_90d.roe if result.backtest_90d else None, "N/A", "{:.1f}"
            ),
            bt_90d_dd=_fmt(
                result.backtest_90d.max_drawdown_pct if result.backtest_90d else None,
                "N/A",
                "{:.1f}",
            ),
        )

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from model response, handling markdown code blocks"""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            content = "\n".join(lines)
        return json.loads(content)

    async def _call_claude(self, prompt: str) -> ModelScore | None:
        """Call Claude API via Messages API"""
        if not self.settings.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set, skipping Claude")
            return None

        url = f"{self.settings.anthropic_base_url}/v1/messages"
        headers = {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.settings.claude_model,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

            content = data["content"][0]["text"]
            result = self._parse_json_response(content)

            return ModelScore(
                score=float(result["score"]),
                recommendation=result["recommendation"],
                risk_level=result["risk_level"],
                reasoning=result["reasoning"],
                model_name="claude",
                trading_tags=result.get("trading_tags", []),
            )
        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            return None

    async def _call_claude_haiku(self, prompt: str) -> ModelScore | None:
        """Call Claude Haiku API for second opinion"""
        if not self.settings.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set, skipping Claude Haiku")
            return None

        url = f"{self.settings.anthropic_base_url}/v1/messages"
        headers = {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.settings.claude_haiku_model,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

            content = data["content"][0]["text"]
            result = self._parse_json_response(content)

            return ModelScore(
                score=float(result["score"]),
                recommendation=result["recommendation"],
                risk_level=result["risk_level"],
                reasoning=result["reasoning"],
                model_name="haiku",
                trading_tags=result.get("trading_tags", []),
            )
        except Exception as e:
            logger.error("Claude Haiku API call failed: %s", e)
            return None

    async def _call_openai(self, prompt: str) -> ModelScore | None:
        """Call OpenAI API (GPT-4o / Codex)"""
        if not self.settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set, skipping OpenAI")
            return None

        url = f"{self.settings.openai_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

            content = data["choices"][0]["message"]["content"]
            result = self._parse_json_response(content)

            return ModelScore(
                score=float(result["score"]),
                recommendation=result["recommendation"],
                risk_level=result["risk_level"],
                reasoning=result["reasoning"],
                model_name="codex",
                trading_tags=result.get("trading_tags", []),
            )
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            return None

    async def evaluate(self, result: FullAnalysisResult) -> AIEvaluation:
        """
        Evaluate trader using dual models, averaging scores.
        If use_dual_claude is True, uses Claude Sonnet + Haiku.
        Otherwise uses Claude + OpenAI.
        Falls back to single model if one fails, or rule-based if both fail.
        """
        if not self.settings.ai_scoring_enabled:
            logger.info("AI scoring disabled, using fallback")
            return self._fallback_evaluation(result)

        prompt = self._format_prompt(result)

        # Choose which models to use
        if self.settings.use_dual_claude:
            # Dual Claude mode: Sonnet + Haiku
            results = await asyncio.gather(
                self._call_claude(prompt), self._call_claude_haiku(prompt), return_exceptions=True
            )
            model1_name = "claude"
            model2_name = "haiku"
        else:
            # Claude + OpenAI mode
            results = await asyncio.gather(
                self._call_claude(prompt), self._call_openai(prompt), return_exceptions=True
            )
            model1_name = "claude"
            model2_name = "codex"

        model1_result = results[0] if not isinstance(results[0], Exception) else None
        model2_result = results[1] if not isinstance(results[1], Exception) else None

        if isinstance(results[0], Exception):
            logger.error("%s call raised exception: %s", model1_name, results[0])
        if isinstance(results[1], Exception):
            logger.error("%s call raised exception: %s", model2_name, results[1])

        scores: list[ModelScore] = []
        if model1_result:
            scores.append(model1_result)
        if model2_result:
            scores.append(model2_result)

        if not scores:
            logger.warning("Both AI models failed, using fallback")
            return self._fallback_evaluation(result)

        # Calculate averages
        avg_score = sum(s.score for s in scores) / len(scores)

        # Use first available for recommendation/risk
        recommendation = scores[0].recommendation
        risk_level = scores[0].risk_level

        # Merge reasoning from both models
        reasonings = [f"[{s.model_name.upper()}] {s.reasoning}" for s in scores]
        merged_reasoning = " | ".join(reasonings)

        # Merge trading tags (dedupe, keep order from first model)
        merged_tags = []
        seen_tags = set()
        for s in scores:
            for tag in s.trading_tags:
                if tag not in seen_tags:
                    merged_tags.append(tag)
                    seen_tags.add(tag)

        # Determine which scores to store
        claude_score = None
        codex_score = None
        for s in scores:
            if s.model_name == "claude":
                claude_score = s.score
            elif s.model_name in ("codex", "haiku"):
                codex_score = s.score

        return AIEvaluation(
            score=round(avg_score, 1),
            recommendation=recommendation,
            risk_level=risk_level,
            reasoning=merged_reasoning,
            trading_tags=merged_tags[:5],
            claude_score=claude_score,
            codex_score=codex_score,
            models_used=[s.model_name for s in scores],
        )

    def _fallback_evaluation(self, result: FullAnalysisResult) -> AIEvaluation:
        """Rule-based fallback evaluation"""
        score = 50.0
        factors = []

        if result.roe_30d is not None:
            if result.roe_30d > 50:
                score += 15
                factors.append("30天ROE优异(>50%)")
            elif result.roe_30d > 20:
                score += 10
                factors.append("30天ROE良好(>20%)")
            elif result.roe_30d > 0:
                score += 5
                factors.append("30天ROE为正")
            elif result.roe_30d < -20:
                score -= 15
                factors.append("30天ROE严重亏损(<-20%)")
            else:
                score -= 5
                factors.append("30天ROE为负")

        if result.win_rate is not None:
            if result.win_rate > 60:
                score += 10
                factors.append("胜率高(>60%)")
            elif result.win_rate > 50:
                score += 5
                factors.append("胜率正常(>50%)")
            elif result.win_rate < 40:
                score -= 10
                factors.append("胜率偏低(<40%)")

        if result.profit_factor is not None:
            if result.profit_factor > 2:
                score += 10
                factors.append("盈亏比优异(>2)")
            elif result.profit_factor > 1.5:
                score += 5
                factors.append("盈亏比良好(>1.5)")
            elif result.profit_factor < 1:
                score -= 10
                factors.append("盈亏比差(<1)")

        if result.max_drawdown_pct is not None:
            dd = abs(result.max_drawdown_pct)
            if dd < 10:
                score += 10
                factors.append("回撤控制极好(<10%)")
            elif dd < 20:
                score += 5
                factors.append("回撤控制良好(<20%)")
            elif dd > 50:
                score -= 15
                factors.append("回撤过大(>50%)")
            elif dd > 30:
                score -= 10
                factors.append("回撤偏大(>30%)")

        if result.backtest_30d:
            if result.backtest_30d.roe > 10:
                score += 5
                factors.append("回测30天盈利(>10%)")
            elif result.backtest_30d.roe < -10:
                score -= 5
                factors.append("回测30天亏损(<-10%)")

        if result.total_trades is not None:
            if result.total_trades < 10:
                score -= 10
                factors.append("交易次数过少，数据不足")

        score = max(0, min(100, score))

        if score >= 75:
            recommendation = "strong_follow"
        elif score >= 60:
            recommendation = "follow"
        elif score >= 40:
            recommendation = "neutral"
        elif score >= 25:
            recommendation = "avoid"
        else:
            recommendation = "strong_avoid"

        dd = abs(result.max_drawdown_pct or 0)
        if dd > 50 or score < 25:
            risk_level = "extreme"
        elif dd > 30 or score < 40:
            risk_level = "high"
        elif dd > 15 or score < 60:
            risk_level = "medium"
        else:
            risk_level = "low"

        reasoning = "[FALLBACK] " + (
            "; ".join(factors) if factors else "数据不足，无法进行详细评估"
        )

        # Generate trading tags based on rules
        tags = []
        if result.account_value is not None:
            if result.account_value < 10000:
                tags.append("小资金")
            elif result.account_value < 100000:
                tags.append("中等资金")
            else:
                tags.append("大资金")
        if result.trade_frequency is not None:
            if result.trade_frequency > 5:
                tags.append("高频")
            elif result.trade_frequency >= 1:
                tags.append("中频")
            else:
                tags.append("低频")
        if result.avg_holding_time is not None:
            hours = result.avg_holding_time / 3600
            if hours < 1:
                tags.append("短线")
            elif hours < 24:
                tags.append("中线")
            else:
                tags.append("长线")
        if result.long_short_ratio is not None:
            if result.long_short_ratio > 1.5:
                tags.append("做多为主")
            elif result.long_short_ratio < 0.67:
                tags.append("做空为主")
            else:
                tags.append("多空均衡")
        if dd > 30:
            tags.append("激进")
        elif dd < 15:
            tags.append("稳健")

        return AIEvaluation(
            score=round(score, 1),
            recommendation=recommendation,
            risk_level=risk_level,
            reasoning=reasoning,
            trading_tags=tags[:5],
            claude_score=None,
            codex_score=None,
            models_used=["fallback"],
        )


_ai_service: AIScoringService | None = None


def get_ai_scoring_service() -> AIScoringService:
    global _ai_service
    if _ai_service is None:
        _ai_service = AIScoringService()
    return _ai_service
