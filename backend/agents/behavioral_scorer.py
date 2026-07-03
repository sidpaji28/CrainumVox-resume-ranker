"""
Rule-based scoring of redrob platform signals.
These capture candidate engagement quality and hiring likelihood —
things a keyword matcher completely ignores.
"""
from datetime import datetime, timezone
from typing import Dict, Tuple


def score_behavioral_signals(signals: dict, jd_parsed: dict) -> Tuple[float, str, list, list]:
    """
    Returns (score 0-100, rationale, red_flags, green_flags)
    """
    score = 0.0
    components = []
    red_flags = []
    green_flags = []

    # ── 1. Profile completeness (max 10 pts) ──────────────────────────────
    completeness = signals.get("profile_completeness_score", 0)
    comp_pts = (completeness / 100) * 10
    score += comp_pts
    components.append(f"Profile completeness {completeness:.0f}% → {comp_pts:.1f}/10")
    if completeness < 60:
        red_flags.append(f"Low profile completeness ({completeness:.0f}%) — candidate may not be serious")

    # ── 2. Verification (max 10 pts) ──────────────────────────────────────
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)
    ver_pts = (5 if verified_email else 0) + (5 if verified_phone else 0)
    score += ver_pts
    if not verified_email:
        red_flags.append("Email not verified")
    if not verified_phone:
        red_flags.append("Phone not verified")
    if verified_email and verified_phone:
        green_flags.append("Both email and phone verified")

    # ── 3. Recruiter response rate (max 15 pts) ───────────────────────────
    rrr = signals.get("recruiter_response_rate", -1)
    if rrr >= 0:
        rrr_pts = rrr * 15
        score += rrr_pts
        if rrr < 0.20:
            red_flags.append(f"Very low recruiter response rate ({rrr:.0%}) — hard to reach")
        elif rrr > 0.50:
            green_flags.append(f"High recruiter response rate ({rrr:.0%})")
        components.append(f"Response rate {rrr:.0%} → {rrr_pts:.1f}/15")

    # ── 4. Interview completion rate (max 15 pts) ─────────────────────────
    icr = signals.get("interview_completion_rate", -1)
    if icr >= 0:
        icr_pts = icr * 15
        score += icr_pts
        if icr < 0.50:
            red_flags.append(f"Low interview completion rate ({icr:.0%}) — drops out of processes")
        elif icr > 0.70:
            green_flags.append(f"Strong interview follow-through ({icr:.0%})")
        components.append(f"Interview completion {icr:.0%} → {icr_pts:.1f}/15")

    # ── 5. Offer acceptance rate (max 10 pts) ─────────────────────────────
    oar = signals.get("offer_acceptance_rate", -1)
    if oar >= 0:
        oar_pts = oar * 10
        score += oar_pts
        if oar < 0.30:
            red_flags.append(f"Low offer acceptance rate ({oar:.0%}) — may be using offers as leverage")
        elif oar > 0.60:
            green_flags.append(f"Solid offer acceptance rate ({oar:.0%})")

    # ── 6. Recent activity (max 10 pts) ───────────────────────────────────
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_ago = (now - last_dt).days
            if days_ago <= 30:
                activity_pts = 10
                green_flags.append(f"Active on platform within last {days_ago} days")
            elif days_ago <= 90:
                activity_pts = 6
            elif days_ago <= 180:
                activity_pts = 3
                red_flags.append(f"Last active {days_ago} days ago — may no longer be searching")
            else:
                activity_pts = 0
                red_flags.append(f"Inactive for {days_ago} days — likely not job hunting")
            score += activity_pts
        except Exception:
            pass

    # ── 7. Open to work flag (max 5 pts) ──────────────────────────────────
    if signals.get("open_to_work_flag"):
        score += 5
        green_flags.append("Actively open to work")
    else:
        red_flags.append("Not marked as open to work")

    # ── 8. GitHub activity (max 10 pts, only for technical roles) ─────────
    github = signals.get("github_activity_score", -1)
    domain = jd_parsed.get("domain", "").lower()
    is_technical = any(k in domain for k in ["engineer", "ml", "data", "backend", "developer", "software"])
    if is_technical and github >= 0:
        github_pts = min(github / 10, 1.0) * 10  # score is 0-10 presumably
        score += github_pts
        if github < 3:
            red_flags.append(f"Low GitHub activity score ({github}) for a technical role")
        elif github > 7:
            green_flags.append(f"Strong GitHub activity ({github}/10)")

    # ── 9. Notice period (max 5 pts) ──────────────────────────────────────
    notice = signals.get("notice_period_days", -1)
    if notice >= 0:
        if notice <= 15:
            score += 5
            green_flags.append(f"Short notice period ({notice} days)")
        elif notice <= 30:
            score += 4
        elif notice <= 60:
            score += 2
        else:
            score += 0
            red_flags.append(f"Long notice period ({notice} days)")

    # Cap at 100
    score = min(score, 100.0)

    rationale = " | ".join(components) if components else "Signals evaluated"
    return round(score, 1), rationale, red_flags, green_flags
