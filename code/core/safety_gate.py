"""Phase 3 — deterministic POST-MODEL safety gate (winner pattern #2).

Runs AFTER the model, in pure Python, so the model can never talk its way
out of a safety decision. Design rules (RESEARCH.md §6, winner #10 scalpel):

- HARD BLOCK (scam -> mute/scam): manipulation attempts (prompt injection),
  OTP/verification/code asks with account-blocking pressure, prize-claim
  pressure, lookalike links with pressure content, unverified businesses
  sending from a domain that does not match their official one with
  pressure/verification content, QR-payment demands, and first-contact
  senders asking for sensitive/verification details.
- USER-PREFERENCE rules (flip only on explicit, measured signals):
  business opt-out -> mute; repeated-ignored sender (dismissed/muted
  history, low open rate) with low-value content -> mute; quiet hours ->
  notify downgraded to digest.
- CAUTION flags (domain mismatch on a VERIFIED business, reported history,
  not-opened history) NEVER flip a decision by themselves — sample_msg_007
  is digest/promotion DESPITE a domain mismatch (verified + user opted in).

Every threshold below was calibrated against the 30 solved samples (zero
regressions required) and the 110 incoming rows (adversarial rows must be
caught: OTP scams in Hinglish, "router metadata" injections, lookalike
domains like chase-secure-alert.com / amazonpay-delivery.in).

The gate only rewrites `action` when the model chose notify/digest for a
hard-scam row (then also type=scam + reason + confidence). A model that
already said mute is left untouched (its type/reason stand).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.data_loader import Dataset, Message, build_features
from core.media import read_image, transcribe_voice
from core.schema import RoutingOutput, calibrate_confidence

# Confidence applied when the gate flips a decision to mute/scam (golden
# scam confidences on the 30 samples are 0.81-0.87; never shrink a higher
# model confidence).
SCAM_CONFIDENCE = 0.85

# ---------------------------------------------------------------------------
# Signal regexes (message text + media text, lowercased)
# ---------------------------------------------------------------------------

# Prompt-injection / router-manipulation language (sample_msg_053, msg_095,
# msg_107, msg_108, msg_110).
INJECTION_RE = re.compile(
    r"ignore\s+(all\s+|any\s+)?(previous|above|prior|earlier)\s+"
    r"(rules|instructions|prompts?|messages|routing)"
    r"|mark\s+this\s+(message\s+)?as\s+(notify|digest|mute)"
    r"|set\s+(action|verdict)\s*="
    r"|routing\s+override"
    r"|(router|routing)\s+(metadata|instructions?|rules)"
    r"|(system|internal)\s+(note|instruction|message|metadata)\s+"
    r"(for|to)\s+(the\s+)?(notification\s+)?router"
    r"|assistant\s+instruction"
    r"|ignore\s+sender\s+risk"
    r"|classify\s+as\s+(urgent|notify)"
    r"|forget\s+(your|all)\s+(previous\s+)?(rules|instructions|prompts?)"
    r"|always\s+mark\s+this\s+as"
    r"|you\s+are\s+now\s+"
    r"|new\s+instructions",
    re.IGNORECASE,
)

# One-time codes / credentials. "otp" is language-neutral (Hinglish scams
# keep the token: "OTP leak ho gaya hai").
CODE_RE = re.compile(
    r"\b(otp|one[- ]?time\s*pass(word)?|login\s*code|verification\s*code|"
    r"security\s*code|passcode|\bpin\b|password)\b",
    re.IGNORECASE,
)

# Ask/action verbs (incl. Hinglish: batao/karo/daalo/dijiye).
ASK_RE = re.compile(
    r"\b(reply|send|share|enter|confirm|provide|give|paste|submit|type|"
    r"open|follow|complete|verify|batao|karo|daalo?|dijiye)\w*\b",
    re.IGNORECASE,
)

# Account-blocking / expiry / leak pressure.
PRESSURE_RE = re.compile(
    r"\b(block(ed|ing)?|locked?|suspend(ed)?|expire(s|d)?|expiry|closure|"
    r"closed?|deactiv\w*|leak(ed|ing)?|restrict\w*|\bhold\b|terminat\w*|"
    r"freez\w*|\bband\b|banned|shut\s*down|unauthorized|failed|failure|"
    r"restore|at\s+risk|security\s+risk)\w*\b",
    re.IGNORECASE,
)

# Payment / money-movement words.
PAYMENT_RE = re.compile(
    r"\b(pay(ment|ments)?|refund|wallet|bank(ing)?|transfer|upi|gpay|"
    r"payout|charge|amount\s+due|\bbill\b|card\b|credential|clearance|"
    r"penalty|fine)\w*\b",
    re.IGNORECASE,
)

# Prize / lottery claim pressure. DOTALL: prize messages can span lines.
PRIZE_RE = re.compile(
    r"\b(congrats|congratulations|you\s+won|winner|lucky\s+(number|customer)|"
    r"selected\s+for\s+(reward|prize)|claim\s+(your|the)?\s*(prize|reward|"
    r"voucher|gift)|jackpot|lottery)\b.*\b(claim|today|now|before|expires|"
    r"expire|hurry)\b",
    re.IGNORECASE | re.DOTALL,
)

# QR / scan payment demands (DOTALL: multi-line fake admin notices).
QR_PAY_RE = re.compile(
    r"\b(qr|scan)\b.*\b(pay|payment|penalty|fine|clearance)\b|"
    r"\b(pay|payment|penalty|fine|clearance)\b.*\b(qr|scan)\b",
    re.IGNORECASE | re.DOTALL,
)

# Content words that turn a lookalike domain / unverified business into a
# hard scam signal (ask/pressure/payment already covered above).
DOMAIN_CONTENT_RE = re.compile(
    r"\b(verify|verification|security|pending|confirm|renew|kyc|"
    r"unauthorized|risk)\w*\b",
    re.IGNORECASE,
)

# Negation guard: legitimate senders explicitly disclaiming code asks
# (msg_093: "no payment or OTP is required for this delivery"; sample 048:
# "the brand says they never ask for OTP") and vision-model descriptions
# that deny risk ("No visible links, QR code, OTP request, or overt scam
# instruction" — img_003/img_023/img_025). Suppresses the code-ask rule
# only; injection/prize/domain signals still fire. Evaluated PER SOURCE
# (message text vs media text) so a denial in one cannot mask a real ask
# in the other.
NEGATION_RE = re.compile(
    r"(never|won'?t|will\s+not|do\s+not|don'?t|does\s+not|is\s+not|are\s+not)"
    r"(\s+\w+){0,4}\s+(ask|require|request|share|need)\w*"
    r"|no\s+(\S+\s+){0,12}(otp|qr\s*code|code|password|pin|payment|links?|instructions?)\b"
    r"|without\s+(any\s+)?(otp|code|password|pin|payment)\b"
    r"|does\s+not\s+(contain|have|include|ask)\w*"
    r"|not\s+required",
    re.IGNORECASE,
)

# Whisper transcribes "OTP" as "OTT" on vn_008 ("Share the OTT you received
# so we can complete verification"). Only scammy in the ask/verification
# combination — never on its own ("OTT subscription expires" is legit).
OTT_SCAM_RE = re.compile(
    r"\bott\b.{0,60}\b(share|send|received)\b.{0,60}\b(verification|verify|code|blocked?)\b",
    re.IGNORECASE | re.DOTALL,
)

# Lookalike-domain tokens. Short tokens (<= 3 chars: pay, kyc, sim) only
# count at label/path boundaries so legit domains (paypal.com, paytm.com,
# t-mobile.com) stay clean; longer tokens count anywhere in a label/path
# (account-login.in, chase-secure-alert.com, amazonpay-delivery.in,
# bit.ly/verify-quick, sbireward.in). None of the 110 official sender
# domains contains one of the long tokens.
_URL_TOKENS = (
    "account", "login", "verify", "secure", "pay", "help", "alert",
    "auth", "wallet", "kyc", "sim", "refund", "reward", "payout",
    "bill", "block", "check", "gift", "gold", "draw", "result",
    "delivery", "desk", "renew", "update", "confirm",
)
_SHORT_TOKENS = frozenset(t for t in _URL_TOKENS if len(t) <= 3)
URL_RE = re.compile(
    r"(?:https?://|www\.)?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.[a-z]{2,}"
    r"(?:/[^\s,.;:)]*)?",
    re.IGNORECASE,
)


def _urls_in(text: str) -> list[str]:
    return [m.group(0) for m in URL_RE.finditer(text.lower())]


def _lookalike_urls(text: str) -> list[str]:
    """URLs whose domain/path contains lookalike-domain tokens."""
    flagged: list[str] = []
    for url in _urls_in(text):
        for label in re.split(r"[/.-]", url):
            if not label:
                continue
            if any(t in label for t in _URL_TOKENS if len(t) >= 4):
                flagged.append(url)
                break
            if any(
                re.search(rf"(?:^|[-.]){re.escape(t)}(?=[^a-z0-9]|$)", label)
                for t in _SHORT_TOKENS
            ):
                flagged.append(url)
                break
    return flagged



# First-contact sensitive asks (samples: OTP/password/codes, payments,
# credentials). "passport"/"id" deliberately excluded (msg_089/msg_096 are
# benign first contacts).
SENSITIVE_RE = re.compile(
    r"\b(aadhaar|pan\s*card|ssn|cvv|card\s+number|account\s+number|"
    r"credentials|otp|password|pin|upi|gpay|bank\s+details)\w*\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Risk signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskSignals:
    scam: bool = False
    scam_signal: str = ""  # human-readable description of the hard signal
    injection: bool = False
    opt_out: bool = False
    repeated_ignored: bool = False
    quiet_hours: bool = False
    # Caution flags (never flip; recorded for observability).
    domain_mismatch: bool = False
    sender_reported_history: bool = False


def _trusted_business(features) -> bool:
    """A verified business sending from its official domain (or with no
    sender-domain claim) is exempt from URL-based and first-contact flags:
    msg_092 (Amex statement), msg_093 (FedEx), msg_094 (Nykaa) are benign
    first contacts with payment/ask words but matching domains."""
    b = features.business
    return (
        b is not None
        and b.verified
        and (b.official_domain == "" or b.official_domain == b.domain_used_by_sender)
    )

def _media_text(ds: Dataset, msg: Message) -> str | None:
    """Media read/transcript for the gate (disk-cached — cheap on re-read)."""
    if msg.media_type == "image" and msg.media_id:
        try:
            return read_image(ds, msg.media_id)
        except Exception:  # noqa: BLE001 — gate must never raise
            return None
    if msg.media_type == "voice" and msg.media_id:
        try:
            return transcribe_voice(ds, msg.media_id)
        except Exception:  # noqa: BLE001 — gate must never raise
            return None
    return None


def _scam_signal(text: str, ds: Dataset, msg: Message, features) -> tuple[bool, str]:
    """Deterministic scam detection. Returns (is_scam, signal_description)."""
    low = text.lower()

    if INJECTION_RE.search(low):
        return True, "message attempts to instruct the router (prompt-injection pattern)"

    negated = bool(NEGATION_RE.search(low))
    code = bool(CODE_RE.search(low)) or bool(OTT_SCAM_RE.search(low))
    ask = bool(ASK_RE.search(low))
    pressure = bool(PRESSURE_RE.search(low))
    payment = bool(PAYMENT_RE.search(low))

    # OTP/code ask + account pressure (or an explicit ask for the code).
    if code and not negated and (ask or pressure):
        return True, "OTP/verification/code request with account pressure"

    if PRIZE_RE.search(low):
        return True, "prize/reward claim with expiry pressure"

    # Lookalike/suspicious link + pressure/verification/payment content.
    # Trusted businesses (verified + matching domain) are exempt: their
    # messages can legitimately contain "secure"/"verify" URLs.
    if not _trusted_business(features) and _lookalike_urls(low) and (
        pressure or ask or payment
    ):
        return True, "suspicious lookalike link with pressure or payment content"

    # QR / scan payment demand with pressure (fake admin notices).
    if QR_PAY_RE.search(low) and pressure:
        return True, "QR/scan payment demand with block pressure"

    # Unverified business sending from a domain that does not match its
    # official one, with pressure/verification/payment content.
    if features.business is not None:
        b = features.business
        if not b.verified and b.official_domain != b.domain_used_by_sender and (
            pressure or ask or payment or bool(DOMAIN_CONTENT_RE.search(low))
        ):
            return True, (
                f"unverified business sending from lookalike domain "
                f"{b.domain_used_by_sender!r} with pressure content"
            )

    # First-contact sender + sensitive ask (trusted verified businesses with
    # a matching domain are exempt: msg_092/msg_093/msg_094 are benign).
    if features.sender_stats.n == 0 and not _trusted_business(features):
        if SENSITIVE_RE.search(low) and ask:
            return True, "first-contact sender asking for sensitive details"
    return False, ""


def analyze_signals(
    ds: Dataset,
    msg: Message,
    features=None,
    media_text: str | None = None,
    fetch_media: bool = False,
) -> RiskSignals:
    """Deterministic risk signals for one message (no model calls).

    `features` may be precomputed (build_features); `media_text` may be
    supplied by the pipeline (already fetched for the agent). When neither
    is given and fetch_media=True, cached media reads are used.
    """
    if features is None:
        features = build_features(ds, msg)
    if media_text is None and fetch_media:
        media_text = _media_text(ds, msg)

    # Sources are scanned SEPARATELY: message text is raw user content;
    # image reads are vision-model descriptions that routinely DENY risk
    # ("No visible links, QR code, OTP request..."). A denial in one source
    # must never mask a real ask in the other.
    scam, scam_signal = False, ""
    injection = False
    for source in (msg.message_text, media_text or ""):
        if not source.strip():
            continue
        injection = injection or bool(INJECTION_RE.search(source.lower()))
        if not scam:
            s, sig = _scam_signal(source, ds, msg, features)
            if s:
                scam, scam_signal = True, sig

    opt_out = False
    if features.ubh is not None and features.ubh.promotions_opted_out_at is not None:
        opt_out = True

    repeated_ignored = False
    s = features.sender_stats
    if s.n >= 2 and s.dismissed >= 2 and (s.muted_after or 0) >= 1:
        # Low open rate keeps the rule off engaged users (u_032 opens 8/8
        # -> digest/promotion 044 stays; u_033 opens 0/5 -> mute 045).
        if s.opened_rate <= 0.25:
            repeated_ignored = True

    return RiskSignals(
        scam=scam,
        scam_signal=scam_signal,
        injection=injection,
        opt_out=opt_out,
        repeated_ignored=repeated_ignored,
        quiet_hours=features.in_quiet_hours,
        domain_mismatch=(
            features.business is not None
            and features.business.verified
            and features.business.official_domain != features.business.domain_used_by_sender
        ),
        sender_reported_history=s.reported > 0,
    )


# ---------------------------------------------------------------------------
# Gate application (post-model)
# ---------------------------------------------------------------------------


def apply_gate(
    ds: Dataset,
    msg: Message,
    output: RoutingOutput,
    features=None,
    media_text: str | None = None,
    fetch_media: bool = False,
) -> RoutingOutput:
    """Enforce the deterministic rules on a model output. Returns the final
    row. Rules fire in order of severity; only the first applicable flip is
    applied (a scam row is mute regardless of opt-out etc.)."""
    if features is None:
        features = build_features(ds, msg)
    sig = analyze_signals(
        ds, msg, features=features, media_text=media_text, fetch_media=fetch_media
    )

    # 1. Hard block: manipulation / scam -> mute + scam.
    if sig.scam:
        if output.action != "mute":
            # Re-anchor the confidence to the mute action's calibrated mean
            # (the pre-flip value was calibrated for the model's action),
            # then enforce the scam floor (Phase 4 review finding 4).
            conf = max(calibrate_confidence("mute", output.confidence), SCAM_CONFIDENCE)
            return RoutingOutput(
                message_id=output.message_id,
                action="mute",
                message_type="scam",
                reason=(
                    f"Safety gate: {sig.scam_signal}. "
                    "Muted regardless of sender or urgency claims."
                ),
                confidence=conf,
                evidence_message_ids=output.evidence_message_ids,
            )
        return output  # already mute — keep the model's type/reason/evidence

    # 2. User opted out of this business's promotions -> mute.
    if sig.opt_out and output.action != "mute":
        mtype = (
            output.message_type
            if output.message_type != "unknown"
            else "promotion"
        )
        return RoutingOutput(
            message_id=output.message_id,
            action="mute",
            message_type=mtype,
            reason=(
                "Safety gate: user opted out of promotions from this business "
                f"({features.ubh.promotions_opted_out_at:%Y-%m-%d})."
            ),
            # Re-anchored to the final action's calibrated mean (finding 4).
            confidence=calibrate_confidence("mute", output.confidence),
            evidence_message_ids=output.evidence_message_ids,
        )

    # 3. Repeated-ignored sender with low-value content -> mute.
    if sig.repeated_ignored and output.action != "mute":
        s = features.sender_stats
        return RoutingOutput(
            message_id=output.message_id,
            action="mute",
            message_type=output.message_type,
            reason=(
                f"Safety gate: {s.dismissed}/{s.n} prior messages from this "
                "sender were dismissed and muted by this user."
            ),
            # Re-anchored to the final action's calibrated mean (finding 4).
            confidence=calibrate_confidence("mute", output.confidence),
            evidence_message_ids=output.evidence_message_ids,
        )

    # 4. Quiet hours: important but not now -> digest (never mute).
    if sig.quiet_hours and output.action == "notify":
        return RoutingOutput(
            message_id=output.message_id,
            action="digest",
            message_type=output.message_type,
            reason=(
                "Safety gate: arrived during the user's quiet hours "
                f"({features.user.dnd_raw}); deferred to digest."
            ),
            # Re-anchored to the final action's calibrated mean (finding 4).
            confidence=calibrate_confidence("digest", output.confidence),
            evidence_message_ids=output.evidence_message_ids,
        )

    return output
