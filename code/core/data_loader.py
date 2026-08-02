"""Phase 1 — deterministic data layer.

Loads every CSV in dataset/, parses embedded-newline fields correctly (csv
module), joins history with events, and builds per-message features for the
routing agent. No LLM calls here — this is the single source of structured
context for the rest of the pipeline.

Invariants verified by tests (test_phase1.py): 110 incoming messages, all
referenced users exist, all group senders are group members, events are 1:1
with history (412 ids), every media file exists on disk.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Raw row types (one dataclass per CSV, numeric fields already typed)
# ---------------------------------------------------------------------------

_DT_FMT = "%Y-%m-%d %H:%M"
_DATE_FMT = "%Y-%m-%d"

_MISSING = {"", "none", "NULL", "null"}


def _opt_str(v: str) -> str | None:
    v = v.strip()
    return None if v in _MISSING else v


def _opt_int(v: str) -> int | None:
    v = v.strip()
    if not v or v in _MISSING:
        return None
    return int(float(v))


def _opt_float(v: str) -> float | None:
    v = v.strip()
    if not v or v in _MISSING:
        return None
    return float(v)


def _opt_dt(v: str, fmt: str = _DT_FMT) -> datetime | None:
    v = v.strip()
    if not v or v in _MISSING:
        return None
    return datetime.strptime(v, fmt)


@dataclass(frozen=True)
class Message:
    """One row of messages.csv / message_history.csv (shared shape)."""

    message_id: str
    user_id: str
    conversation_type: str  # personal | group | business
    group_id: str | None
    business_id: str | None
    sender_user_id: str | None
    created_at: datetime
    message_text: str
    media_type: str  # '' | image | voice
    media_id: str | None
    forwarded_count: int


@dataclass(frozen=True)
class HistoryMessage(Message):
    """message_history.csv joined 1:1 with message_events.csv (same 412 ids)."""

    opened: bool | None
    replied: bool | None
    reaction_minutes: float | None
    dismissed: bool | None
    muted_after: bool | None
    reported: bool | None


@dataclass(frozen=True)
class SampleMessage(Message):
    """sample_messages.csv — the 30 solved rows used as validation labels."""

    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    dnd_raw: str
    dnd_start_min: int | None  # minutes since midnight; start>end => window wraps
    dnd_end_min: int | None
    opened_30d: int
    replied_30d: int
    dismissed_30d: int
    reported_30d: int


@dataclass(frozen=True)
class GroupProfile:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: datetime
    messages_30d: int


@dataclass(frozen=True)
class GroupMembership:
    group_id: str
    user_id: str
    role: str  # admin | member | ...
    joined_at: datetime
    sent_30d: int
    read_30d: int
    replied_30d: int
    dismissed_30d: int
    group_muted_by_user: bool


@dataclass(frozen=True)
class BusinessProfile:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int | None
    messages_sent_30d: int | None
    user_reports_30d: int | None
    domain_used_by_sender_age_days: int | None


@dataclass(frozen=True)
class UserBusinessHistory:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: datetime | None
    allows_promotions: bool
    promotions_opted_out_at: datetime | None
    activity_count_180d: int | None
    messages_opened_30d: int | None
    messages_dismissed_30d: int | None
    messages_replied_30d: int | None
    last_reply_at: datetime | None


@dataclass(frozen=True)
class DailyNotif:
    user_id: str
    date: datetime
    notifications_sent: int
    notifications_dismissed: int


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    incoming: list[Message]  # 110 rows — every row must get a prediction
    samples: list[SampleMessage]  # 30 solved rows (validation only)
    history: list[HistoryMessage]  # 412 rows, joined with events
    users: dict[str, UserProfile]  # 54
    groups: dict[str, GroupProfile]  # 23
    memberships: dict[tuple[str, str], GroupMembership]  # (group_id, user_id)
    businesses: dict[str, BusinessProfile]  # 110
    user_business: dict[tuple[str, str], UserBusinessHistory]  # (user_id, biz_id)
    daily: dict[str, list[DailyNotif]]  # user_id -> sorted-by-date list
    images: dict[str, Path]  # image_id -> absolute media path (20)
    voice_notes: dict[str, Path]  # voice_note_id -> absolute media path (13)
    dataset_dir: Path

    @property
    def history_by_id(self) -> dict[str, HistoryMessage]:
        return {m.message_id: m for m in self.history}


def repo_root() -> Path:
    """Repo root = two levels above code/core/."""
    return Path(__file__).resolve().parents[2]


def default_dataset_dir() -> Path:
    return repo_root() / "dataset"


def load_dataset(dataset_dir: str | Path | None = None) -> Dataset:
    """Load every CSV under dataset_dir (default: <repo>/dataset)."""
    ddir = Path(dataset_dir) if dataset_dir else default_dataset_dir()
    ddir = ddir.resolve()
    media_dir = ddir / "media"

    def _rows(name: str) -> list[dict]:
        with (ddir / name).open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def _parse_message(row: dict) -> Message:
        return Message(
            message_id=row["message_id"].strip(),
            user_id=row["user_id"].strip(),
            conversation_type=row["conversation_type"].strip(),
            group_id=_opt_str(row.get("group_id", "")),
            business_id=_opt_str(row.get("business_id", "")),
            sender_user_id=_opt_str(row.get("sender_user_id", "")),
            created_at=datetime.strptime(row["created_at"].strip(), _DT_FMT),
            message_text=row.get("message_text", "") or "",
            media_type=(row.get("media_type", "") or "").strip(),
            media_id=_opt_str(row.get("media_id", "")),
            forwarded_count=_opt_int(row.get("forwarded_count", "0")) or 0,
        )

    # --- incoming messages -------------------------------------------------
    incoming = [_parse_message(r) for r in _rows("messages.csv")]

    # --- sample messages (solved) -----------------------------------------
    samples = []
    for r in _rows("sample_messages.csv"):
        m = _parse_message(r)
        samples.append(
            SampleMessage(
                **{
                    **m.__dict__,
                    "action": r["action"].strip(),
                    "message_type": r["message_type"].strip(),
                    "reason": r["reason"].strip(),
                    "confidence": float(r["confidence"]),
                    "evidence_message_ids": r["evidence_message_ids"].strip(),
                }
            )
        )

    # --- history + events (1:1 join on message_id) --------------------------
    events = {r["message_id"].strip(): r for r in _rows("message_events.csv")}
    history = []
    for r in _rows("message_history.csv"):
        m = _parse_message(r)
        ev = events.get(m.message_id, {})
        history.append(
            HistoryMessage(
                **{
                    **m.__dict__,
                    "opened": _opt_int(ev.get("message_opened", "")) == 1
                    if ev
                    else None,
                    "replied": _opt_int(ev.get("message_replied", "")) == 1
                    if ev
                    else None,
                    "reaction_minutes": _opt_float(ev.get("reaction_time_minutes", ""))
                    if ev
                    else None,
                    "dismissed": _opt_int(ev.get("notification_dismissed", "")) == 1
                    if ev
                    else None,
                    "muted_after": _opt_int(ev.get("muted_after_message", "")) == 1
                    if ev
                    else None,
                    "reported": _opt_int(ev.get("message_reported", "")) == 1
                    if ev
                    else None,
                }
            )
        )

    # --- users ---------------------------------------------------------------
    users = {}
    for r in _rows("users.csv"):
        uid = r["user_id"].strip()
        s, e = parse_dnd_window(r["do_not_disturb_window"])
        users[uid] = UserProfile(
            user_id=uid,
            dnd_raw=r["do_not_disturb_window"].strip(),
            dnd_start_min=s,
            dnd_end_min=e,
            opened_30d=_opt_int(r["messages_opened_30d"]) or 0,
            replied_30d=_opt_int(r["messages_replied_30d"]) or 0,
            dismissed_30d=_opt_int(r["notifications_dismissed_30d"]) or 0,
            reported_30d=_opt_int(r["messages_reported_30d"]) or 0,
        )

    # --- groups + memberships --------------------------------------------------
    groups = {}
    for r in _rows("groups.csv"):
        gid = r["group_id"].strip()
        groups[gid] = GroupProfile(
            group_id=gid,
            group_name=r["group_name"].strip(),
            group_type=r["group_type"].strip(),
            member_count=_opt_int(r["member_count"]) or 0,
            admin_count=_opt_int(r["admin_count"]) or 0,
            created_at=datetime.strptime(r["created_at"].strip(), _DATE_FMT),
            messages_30d=_opt_int(r["messages_30d"]) or 0,
        )

    memberships = {}
    for r in _rows("group_members.csv"):
        key = (r["group_id"].strip(), r["user_id"].strip())
        memberships[key] = GroupMembership(
            group_id=key[0],
            user_id=key[1],
            role=r["role"].strip(),
            joined_at=datetime.strptime(r["joined_at"].strip(), _DATE_FMT),
            sent_30d=_opt_int(r["messages_sent_30d"]) or 0,
            read_30d=_opt_int(r["messages_read_30d"]) or 0,
            replied_30d=_opt_int(r["replies_sent_30d"]) or 0,
            dismissed_30d=_opt_int(r["notifications_dismissed_30d"]) or 0,
            group_muted_by_user=_opt_int(r["group_muted_by_user"]) == 1,
        )

    # --- businesses + user-business history ------------------------------------
    businesses = {}
    for r in _rows("business_accounts.csv"):
        bid = r["business_id"].strip()
        businesses[bid] = BusinessProfile(
            business_id=bid,
            display_name=r["display_name"].strip(),
            brand_name=r["brand_name"].strip(),
            category=r["category"].strip(),
            verified=_opt_int(r["verified"]) == 1,
            official_domain=r["official_domain"].strip(),
            domain_used_by_sender=r["domain_used_by_sender"].strip(),
            account_age_days=_opt_int(r["account_age_days"]),
            messages_sent_30d=_opt_int(r["messages_sent_30d"]),
            user_reports_30d=_opt_int(r["user_reports_30d"]),
            domain_used_by_sender_age_days=_opt_int(
                r["domain_used_by_sender_age_days"]
            ),
        )

    user_business = {}
    for r in _rows("user_business_history.csv"):
        key = (r["user_id"].strip(), r["business_id"].strip())
        user_business[key] = UserBusinessHistory(
            user_id=key[0],
            business_id=key[1],
            why_user_knows_account=r["why_user_knows_account"].strip(),
            last_activity_at=_opt_dt(r.get("last_activity_at", ""), _DT_FMT),
            allows_promotions=_opt_int(r["allows_promotions"]) == 1,
            promotions_opted_out_at=_opt_dt(
                r.get("promotions_opted_out_at", ""), _DT_FMT
            ),
            activity_count_180d=_opt_int(r.get("activity_count_180d", "")),
            messages_opened_30d=_opt_int(r.get("messages_opened_30d", "")),
            messages_dismissed_30d=_opt_int(r.get("messages_dismissed_30d", "")),
            messages_replied_30d=_opt_int(r.get("messages_replied_30d", "")),
            last_reply_at=_opt_dt(r.get("last_reply_at", ""), _DT_FMT),
        )

    # --- daily notification summary -------------------------------------------
    daily: dict[str, list[DailyNotif]] = {}
    for r in _rows("daily_notification_summary.csv"):
        uid = r["user_id"].strip()
        d = DailyNotif(
            user_id=uid,
            date=datetime.strptime(r["date"].strip(), _DATE_FMT),
            notifications_sent=_opt_int(r["notifications_sent"]) or 0,
            notifications_dismissed=_opt_int(r["notifications_dismissed"]) or 0,
        )
        daily.setdefault(uid, []).append(d)
    for lst in daily.values():
        lst.sort(key=lambda d: d.date)

    # --- media ----------------------------------------------------------------
    images = {
        r["image_id"].strip(): (ddir / r["file_path"].strip()).resolve()
        for r in _rows("images.csv")
    }
    voice_notes = {
        r["voice_note_id"].strip(): (ddir / r["file_path"].strip()).resolve()
        for r in _rows("voice_notes.csv")
    }

    return Dataset(
        incoming=incoming,
        samples=samples,
        history=history,
        users=users,
        groups=groups,
        memberships=memberships,
        businesses=businesses,
        user_business=user_business,
        daily=daily,
        images=images,
        voice_notes=voice_notes,
        dataset_dir=ddir,
    )


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------

_DND_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


def parse_dnd_window(raw: str) -> tuple[int | None, int | None]:
    """'22:00-07:00' -> (1320, 420) minutes since midnight.

    start > end means the window wraps past midnight. A zero-length window
    (start == end) is treated as no quiet hours to avoid all-day silence.
    Returns (None, None) for unparseable input.
    """
    m = _DND_RE.match(raw)
    if not m:
        return None, None
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    if not (0 <= h1 <= 23 and 0 <= m1 <= 59 and 0 <= h2 <= 23 and 0 <= m2 <= 59):
        return None, None
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if start == end:
        return None, None
    return start, end


def in_quiet_hours(dt: datetime, dnd_start_min: int | None, dnd_end_min: int | None) -> bool:
    """True if dt falls inside the (possibly wrap-around) DND window."""
    if dnd_start_min is None or dnd_end_min is None:
        return False
    now = dt.hour * 60 + dt.minute
    if dnd_start_min < dnd_end_min:
        return dnd_start_min <= now < dnd_end_min
    return now >= dnd_start_min or now < dnd_end_min


# ---------------------------------------------------------------------------
# Feature builder (deterministic context for the agent; no LLM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SenderHistoryStats:
    """Behavior of this sender (or business) towards this user in history."""

    n: int
    opened: int
    replied: int
    dismissed: int
    muted_after: int
    reported: int
    opened_rate: float
    replied_rate: float
    dismissed_rate: float
    reported_rate: float
    forwarded: int
    last_contact_days: float | None


@dataclass(frozen=True)
class MessageFeatures:
    message: Message
    user: UserProfile
    # type-specific context (exactly one of group/business/sender is set)
    group: GroupProfile | None
    self_membership: GroupMembership | None  # receiving user in the group
    sender_membership: GroupMembership | None  # sender in the group
    business: BusinessProfile | None
    ubh: UserBusinessHistory | None  # user x business history
    personal_sender: UserProfile | None  # sender profile (personal msgs)
    sender_stats: SenderHistoryStats
    hour: int
    weekday: int
    in_quiet_hours: bool
    daily_latest_sent: int | None
    daily_latest_dismissed: int | None
    daily_avg_sent: float | None


def _history_scope(ds: Dataset, msg: Message) -> list[HistoryMessage]:
    """History messages relevant to this conversation."""
    if msg.conversation_type == "group":
        return [
            m
            for m in ds.history
            if m.group_id == msg.group_id
            and m.user_id == msg.user_id
            and m.sender_user_id == msg.sender_user_id
        ]
    if msg.conversation_type == "business":
        return [m for m in ds.history if m.business_id == msg.business_id and m.user_id == msg.user_id]
    return [
        m
        for m in ds.history
        if m.conversation_type == "personal"
        and m.user_id == msg.user_id
        and m.sender_user_id == msg.sender_user_id
    ]


def _sender_stats(scope: list[HistoryMessage], now: datetime) -> SenderHistoryStats:
    n = len(scope)
    opened = sum(1 for m in scope if m.opened)
    replied = sum(1 for m in scope if m.replied)
    dismissed = sum(1 for m in scope if m.dismissed)
    muted_after = sum(1 for m in scope if m.muted_after)
    reported = sum(1 for m in scope if m.reported)
    forwarded = sum(1 for m in scope if m.forwarded_count > 0)
    last_days = None
    if scope:
        last = max(m.created_at for m in scope)
        last_days = (now - last).total_seconds() / 86400.0
    return SenderHistoryStats(
        n=n,
        opened=opened,
        replied=replied,
        dismissed=dismissed,
        muted_after=muted_after,
        reported=reported,
        opened_rate=opened / n if n else 0.0,
        replied_rate=replied / n if n else 0.0,
        dismissed_rate=dismissed / n if n else 0.0,
        reported_rate=reported / n if n else 0.0,
        forwarded=forwarded,
        last_contact_days=last_days,
    )


def build_features(ds: Dataset, msg: Message) -> MessageFeatures:
    """Build the deterministic feature bundle for one incoming message."""
    user = ds.users[msg.user_id]
    group = ds.groups.get(msg.group_id) if msg.group_id else None
    business = ds.businesses.get(msg.business_id) if msg.business_id else None
    self_membership = (
        ds.memberships.get((msg.group_id, msg.user_id)) if msg.group_id else None
    )
    sender_membership = (
        ds.memberships.get((msg.group_id, msg.sender_user_id))
        if msg.group_id and msg.sender_user_id
        else None
    )
    ubh = (
        ds.user_business.get((msg.user_id, msg.business_id))
        if msg.business_id
        else None
    )
    personal_sender = (
        ds.users.get(msg.sender_user_id)
        if msg.conversation_type == "personal" and msg.sender_user_id
        else None
    )

    scope = _history_scope(ds, msg)
    stats = _sender_stats(scope, msg.created_at)

    daily = ds.daily.get(msg.user_id, [])
    latest = daily[-1] if daily else None

    return MessageFeatures(
        message=msg,
        user=user,
        group=group,
        self_membership=self_membership,
        sender_membership=sender_membership,
        business=business,
        ubh=ubh,
        personal_sender=personal_sender,
        sender_stats=stats,
        hour=msg.created_at.hour,
        weekday=msg.created_at.weekday(),
        in_quiet_hours=in_quiet_hours(
            msg.created_at, user.dnd_start_min, user.dnd_end_min
        ),
        daily_latest_sent=latest.notifications_sent if latest else None,
        daily_latest_dismissed=latest.notifications_dismissed if latest else None,
        daily_avg_sent=(
            sum(d.notifications_sent for d in daily) / len(daily) if daily else None
        ),
    )
