"""
GOATlens Telegram Bot
Usage: /analyze AAPL
"""

import asyncio
import io
import logging
import os
import re

import httpx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

matplotlib.use("Agg")  # Non-interactive backend — required for server-side rendering

# ── MarkdownV2 helper ──────────────────────────────────────────────────────────

def esc(text: str) -> str:
    """Escape all MarkdownV2 reserved characters in dynamic text."""
    reserved = r'\_*[]()~`>#+-=|{}.!'
    return re.sub(f"([{re.escape(reserved)}])", r'\\\1', str(text))

# Load .env from backend directory
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Verdict styling ────────────────────────────────────────────────────────────

VERDICT_EMOJI = {
    "strong_buy":  "🚀",
    "buy":         "✅",
    "hold":        "⏸️",
    "sell":        "⚠️",
    "strong_sell": "🔴",
}

VERDICT_LABEL = {
    "strong_buy":  "STRONG BUY",
    "buy":         "BUY",
    "hold":        "HOLD",
    "sell":        "SELL",
    "strong_sell": "STRONG SELL",
}

AGENT_EMOJI = {
    "Warren Buffett":  "🎩",
    "Peter Lynch":     "📈",
    "Benjamin Graham": "📚",
    "Charlie Munger":  "🧠",
    "Ray Dalio":       "🌊",
    "Michael Burry":   "🔍",
}

MOAT_EMOJI = {
    "strengthening": "🏰⬆️",
    "stable":        "🏰",
    "weakening":     "🏰⬇️",
    "collapsed":     "🏚️",
}

# ── Chart helpers ──────────────────────────────────────────────────────────────

def build_agent_scores_chart(agent_results: list) -> io.BytesIO:
    """Bar chart of each agent's score (-100 to +100)."""
    names  = [r["agent"].split()[-1] for r in agent_results]
    scores = [r["score"] for r in agent_results]
    colors = ["#2ecc71" if s > 20 else "#e74c3c" if s < -20 else "#f39c12"
              for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(names, scores, color=colors, height=0.55, zorder=3)
    ax.axvline(0, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(-110, 110)
    ax.set_xlabel("Score", color="#cccccc", fontsize=10)
    ax.set_title("Agent Scores", color="#ffffff", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(axis="x", color="#333355", linestyle="--", alpha=0.5, zorder=0)

    for bar, score in zip(bars, scores):
        ax.text(
            score + (3 if score >= 0 else -3),
            bar.get_y() + bar.get_height() / 2,
            f"{score:+.0f}",
            va="center", ha="left" if score >= 0 else "right",
            color="#ffffff", fontsize=9, fontweight="bold"
        )

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Buy zone  (>20)"),
        mpatches.Patch(color="#f39c12", label="Hold zone (-20 to 20)"),
        mpatches.Patch(color="#e74c3c", label="Sell zone (<-20)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#333355",
              labelcolor="#cccccc", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def build_price_history_chart(price_data: dict, ticker: str) -> io.BytesIO:
    """Line chart of 1-year price history."""
    dates  = price_data.get("dates", [])
    closes = price_data.get("closes", [])

    if not dates or not closes:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.plot(range(len(closes)), closes, color="#3498db", linewidth=1.5, zorder=3)
    ax.fill_between(range(len(closes)), closes,
                    alpha=0.15, color="#3498db", zorder=2)

    step = max(1, len(dates) // 6)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels(
        [dates[i][:10] for i in range(0, len(dates), step)],
        rotation=30, ha="right", fontsize=7, color="#cccccc"
    )

    ax.set_ylabel("Price (USD)", color="#cccccc", fontsize=10)
    ax.set_title(f"{ticker} - 1 Year Price History",
                 color="#ffffff", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(color="#333355", linestyle="--", alpha=0.4, zorder=0)

    min_idx = int(np.argmin(closes))
    max_idx = int(np.argmax(closes))
    ax.annotate(f"${closes[min_idx]:.0f}",
                xy=(min_idx, closes[min_idx]),
                xytext=(min_idx, closes[min_idx] * 0.97),
                color="#e74c3c", fontsize=8, ha="center")
    ax.annotate(f"${closes[max_idx]:.0f}",
                xy=(max_idx, closes[max_idx]),
                xytext=(max_idx, closes[max_idx] * 1.02),
                color="#2ecc71", fontsize=8, ha="center")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def build_earnings_chart(earnings_data: list, ticker: str) -> io.BytesIO:
    """Bar chart of earnings surprises (beat/miss %)."""
    if not earnings_data:
        return None

    recent    = earnings_data[-8:]
    quarters  = [e.get("date", "")[:7] for e in recent]
    surprises = []
    for e in recent:
        actual   = e.get("actual_eps") or e.get("epsActual", 0) or 0
        estimate = e.get("estimated_eps") or e.get("epsEstimated", 0) or 0
        if estimate and estimate != 0:
            surprises.append(((actual - estimate) / abs(estimate)) * 100)
        else:
            surprises.append(0)

    colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in surprises]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.bar(range(len(quarters)), surprises, color=colors, zorder=3)
    ax.axhline(0, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(quarters)))
    ax.set_xticklabels(quarters, rotation=30, ha="right",
                       fontsize=8, color="#cccccc")
    ax.set_ylabel("EPS Surprise %", color="#cccccc", fontsize=10)
    ax.set_title(f"{ticker} - Earnings Surprises (Last 8 Quarters)",
                 color="#ffffff", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(axis="y", color="#333355", linestyle="--", alpha=0.4, zorder=0)

    for i, s in enumerate(surprises):
        ax.text(i, s + (0.3 if s >= 0 else -0.8),
                f"{s:+.1f}%", ha="center", va="bottom",
                color="#ffffff", fontsize=7, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Message formatter ──────────────────────────────────────────────────────────

def format_analysis_message(data: dict) -> str:
    """
    Build the MarkdownV2 analysis message.
    All dynamic API content is passed through esc() to auto-escape
    reserved characters like . ( ) - ! etc.
    """
    ticker        = data.get("ticker", "")
    company       = data.get("company_name", ticker)
    sector        = data.get("sector", "N/A")
    verdict       = data.get("consensus_verdict", "hold")
    agreement     = data.get("agreement_score", 0)
    moat_trend    = data.get("moat_trend", "")
    next_earnings = data.get("next_earnings_date", "N/A")

    v_emoji = VERDICT_EMOJI.get(verdict, "❓")
    v_label = VERDICT_LABEL.get(verdict, verdict.upper())
    m_emoji = MOAT_EMOJI.get(moat_trend, "")

    lines = [
        f"🐐 *{esc(company)}* \\(`{esc(ticker)}`\\)",
        f"🏢 Sector: {esc(sector)}",
        "",
        f"{v_emoji} *Consensus: {esc(v_label)}*",
        f"🤝 Agreement: {esc(f'{agreement:.0%}')}",
    ]

    if moat_trend:
        lines.append(f"{m_emoji} Moat: {esc(moat_trend.capitalize())}")
    if next_earnings and next_earnings != "N/A":
        lines.append(f"📅 Next Earnings: {esc(str(next_earnings))}")

    # Agent verdicts
    lines += ["", "━━━━━━━━━━━━━━━━━━━━━", "*Agent Verdicts*"]
    for r in data.get("agent_results", []):
        agent     = r.get("agent", "")
        emoji     = AGENT_EMOJI.get(agent, "•")
        a_verdict = r.get("verdict", "hold")
        score     = r.get("score", 0)
        a_label   = VERDICT_LABEL.get(a_verdict, a_verdict.upper())
        lines.append(
            f"{emoji} *{esc(agent)}*: {esc(a_label)} \\({esc(f'{score:+.0f}')}\\)"
        )

    # Consensus points
    consensus_pts = data.get("consensus_points", [])
    if consensus_pts:
        lines += ["", "━━━━━━━━━━━━━━━━━━━━━", "✅ *Where They Agree*"]
        for pt in consensus_pts[:2]:
            lines.append(f"• {esc(pt)}")

    # Divergence points
    divergence_pts = data.get("divergence_points", [])
    if divergence_pts:
        lines += ["", "⚡ *Where They Disagree*"]
        for pt in divergence_pts[:2]:
            lines.append(f"• {esc(pt)}")

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━",
        "_For educational purposes only\\. Not financial advice\\._"
    ]

    return "\n".join(lines)


# ── Command handlers ───────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐐 *Welcome to GOATlens\\!*\n\n"
        "Analyze stocks through the eyes of legendary investors\\.\n\n"
        "*Usage:*\n"
        "`/analyze AAPL` — Full analysis\n"
        "`/analyze MSFT 2019 2024` — Custom anchor years\n\n"
        "*Investors:* Buffett 🎩 Lynch 📈 Graham 📚 Munger 🧠 Dalio 🌊",
        parse_mode="MarkdownV2"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *GOATlens Help*\n\n"
        "*Commands:*\n"
        "`/analyze TICKER` — Analyze a stock\n"
        "`/analyze TICKER YEAR1 YEAR2` — With custom anchor years\n\n"
        "*Examples:*\n"
        "`/analyze AAPL`\n"
        "`/analyze TSLA 2019 2024`\n"
        "`/analyze NVDA 2014 2019 2024`\n\n"
        "*Verdicts:* 🚀 Strong Buy ✅ Buy ⏸️ Hold ⚠️ Sell 🔴 Strong Sell",
        parse_mode="MarkdownV2"
    )


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args

    if not args:
        await update.message.reply_text(
            "Please provide a ticker\\. Example: `/analyze AAPL`",
            parse_mode="MarkdownV2"
        )
        return

    ticker = args[0].upper().strip()

    # Parse optional anchor years
    anchor_years = [2019, 2024]
    if len(args) >= 3:
        try:
            anchor_years = [int(y) for y in args[1:]]
        except ValueError:
            pass

    # Immediate acknowledgement
    thinking_msg = await update.message.reply_text(
        f"🔍 Analyzing *{esc(ticker)}* \\.\\.\\. This takes about 15 seconds\\.",
        parse_mode="MarkdownV2"
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            analyze_task  = client.post(
                f"{API_BASE_URL}/api/analyze",
                json={"ticker": ticker, "anchor_years": anchor_years}
            )
            price_task    = client.get(f"{API_BASE_URL}/api/price-history/{ticker}")
            earnings_task = client.get(f"{API_BASE_URL}/api/earnings/{ticker}")

            analyze_resp, price_resp, earnings_resp = await asyncio.gather(
                analyze_task, price_task, earnings_task,
                return_exceptions=True
            )

        # ── Parse responses ────────────────────────────────────────────────────
        if isinstance(analyze_resp, Exception):
            raise analyze_resp

        if analyze_resp.status_code != 200:
            await thinking_msg.delete()
            await update.message.reply_text(
                f"❌ Analysis failed for `{esc(ticker)}`\\. "
                f"Check the ticker and try again\\.",
                parse_mode="MarkdownV2"
            )
            return

        analysis_data = analyze_resp.json()
        price_data    = (price_resp.json()
                         if not isinstance(price_resp, Exception)
                         and price_resp.status_code == 200 else {})
        earnings_data = (earnings_resp.json()
                         if not isinstance(earnings_resp, Exception)
                         and earnings_resp.status_code == 200 else [])

        await thinking_msg.delete()

        # ── Main analysis text ─────────────────────────────────────────────────
        message_text = format_analysis_message(analysis_data)
        await update.message.reply_text(message_text, parse_mode="MarkdownV2")

        # ── Charts ─────────────────────────────────────────────────────────────
        agent_results = analysis_data.get("agent_results", [])

        if agent_results:
            scores_buf = build_agent_scores_chart(agent_results)
            await update.message.reply_photo(
                photo=scores_buf,
                caption=f"📊 {ticker} — Agent Score Breakdown"
            )

        if price_data:
            price_buf = build_price_history_chart(price_data, ticker)
            if price_buf:
                await update.message.reply_photo(
                    photo=price_buf,
                    caption=f"📈 {ticker} — 1 Year Price History"
                )

        earnings_list = (earnings_data if isinstance(earnings_data, list)
                         else earnings_data.get("earnings", []))
        if earnings_list:
            earnings_buf = build_earnings_chart(earnings_list, ticker)
            if earnings_buf:
                await update.message.reply_photo(
                    photo=earnings_buf,
                    caption=f"💰 {ticker} — Earnings Surprises"
                )

    except httpx.ConnectError:
        await thinking_msg.delete()
        await update.message.reply_text(
            "❌ Cannot connect to GOATlens server\\. "
            "Make sure `python main\\.py` is running\\.",
            parse_mode="MarkdownV2"
        )
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
        try:
            await thinking_msg.delete()
        except Exception:
            pass
        await update.message.reply_text(
            f"❌ Unexpected error analyzing `{esc(ticker)}`\\. Please try again\\.",
            parse_mode="MarkdownV2"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    token = TELEGRAM_BOT_TOKEN
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("help",    help_command))
    app.add_handler(CommandHandler("analyze", analyze))

    logger.info("🐐 GOATlens bot is running... Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()