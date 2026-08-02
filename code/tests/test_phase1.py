"""Phase 1 invariants: data layer, media manifests, retrieval.

No API calls and no model downloads in these tests — media tests check the
manifest (every image/voice id resolves to an existing file), and retrieval
tests run the deterministic scorer only.
"""

from datetime import datetime

import pytest

from core.data_loader import (
    build_features,
    in_quiet_hours,
    load_dataset,
    parse_dnd_window,
)
from core.retrieval import build_index, choose_evidence_ids, retrieve_evidence

DS = load_dataset()


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


def test_incoming_110_unique_ids():
    ids = [m.message_id for m in DS.incoming]
    assert len(ids) == 110
    assert len(set(ids)) == 110


def test_conv_media_distribution():
    from collections import Counter

    conv = Counter(m.conversation_type for m in DS.incoming)
    assert conv == {"group": 63, "business": 30, "personal": 17}
    media = Counter(m.media_type for m in DS.incoming)
    assert media["image"] == 15 and media["voice"] == 8 and media[""] == 87


def test_all_referenced_users_exist():
    known = set(DS.users)
    for m in DS.incoming:
        assert m.user_id in known
        if m.conversation_type == "personal":
            assert m.sender_user_id in known, m.message_id


def test_group_senders_are_members():
    for m in DS.incoming:
        if m.conversation_type == "group":
            assert (m.group_id, m.sender_user_id) in DS.memberships, m.message_id
            assert m.group_id in DS.groups


def test_business_references_resolve():
    for m in DS.incoming:
        if m.conversation_type == "business":
            assert m.business_id in DS.businesses, m.message_id
            assert m.sender_user_id is None  # sender is the business itself


def test_history_and_events_are_1_to_1():
    hist_ids = {m.message_id for m in DS.history}
    assert len(DS.history) == 412
    assert len(hist_ids) == 412
    # events joined at load time: every history row has event fields
    assert all(m.opened is not None for m in DS.history)
    assert all(m.reported is not None for m in DS.history)


def test_media_manifests_match_disk():
    assert len(DS.images) == 20
    assert len(DS.voice_notes) == 13
    for img_id, path in DS.images.items():
        assert path.exists(), f"{img_id} -> {path}"
    for vn_id, path in DS.voice_notes.items():
        assert path.exists(), f"{vn_id} -> {path}"


def test_sample_set_integrity():
    assert len(DS.samples) == 30
    from collections import Counter

    assert Counter(s.action for s in DS.samples) == {
        "digest": 11,
        "mute": 10,
        "notify": 9,
    }
    for s in DS.samples:
        assert 0.0 <= s.confidence <= 1.0
        assert s.evidence_message_ids != ""


def test_sample_evidence_ids_exist_in_history():
    hist_ids = {m.message_id for m in DS.history}
    for s in DS.samples:
        for eid in s.evidence_message_ids.split(";"):
            if eid and eid != "none":
                assert eid in hist_ids, f"{s.message_id} cites {eid}"


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------


def test_dnd_parse_normal_and_wrap():
    assert parse_dnd_window("22:00-07:00") == (22 * 60, 7 * 60)
    assert parse_dnd_window("09:30-17:15") == (9 * 60 + 30, 17 * 60 + 15)
    assert parse_dnd_window("") == (None, None)
    assert parse_dnd_window("garbage") == (None, None)
    # zero-length window -> no quiet hours (avoid all-day silence)
    assert parse_dnd_window("00:00-00:00") == (None, None)


def test_in_quiet_hours():
    assert in_quiet_hours(datetime(2026, 7, 30, 23, 0), 22 * 60, 7 * 60)  # wrap
    assert in_quiet_hours(datetime(2026, 7, 30, 5, 59), 22 * 60, 7 * 60)
    assert not in_quiet_hours(datetime(2026, 7, 30, 12, 0), 22 * 60, 7 * 60)
    assert in_quiet_hours(datetime(2026, 7, 30, 10, 0), 9 * 60, 17 * 60)
    assert not in_quiet_hours(datetime(2026, 7, 30, 7, 0), 9 * 60, 17 * 60)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def test_features_build_for_all_incoming():
    for m in DS.incoming:
        f = build_features(DS, m)
        assert f.user.user_id == m.user_id
        if m.conversation_type == "group":
            assert f.group is not None and f.sender_membership is not None
            assert f.business is None and f.ubh is None
        elif m.conversation_type == "business":
            assert f.business is not None
            assert f.group is None
        else:
            assert f.personal_sender is not None
        assert isinstance(f.sender_stats.n, int)
        assert 0.0 <= f.sender_stats.opened_rate <= 1.0


def test_quiet_hours_flag_on_known_message():
    # u_002 dnd window is 23:00-08:00; msg_023 is 22:19 -> NOT quiet
    msg = next(m for m in DS.incoming if m.message_id == "msg_023")
    f = build_features(DS, msg)
    assert f.in_quiet_hours is False
    # pick a message that is inside quiet hours for its user and assert True
    found = False
    for m in DS.incoming:
        ff = build_features(DS, m)
        if ff.in_quiet_hours:
            assert ff.user.dnd_raw != ""
            found = True
            break
    assert found, "expected at least one incoming message inside quiet hours"


def test_daily_summary_lookup():
    msg = DS.incoming[0]
    f = build_features(DS, msg)
    # user's daily rows exist (all 32 receiving users are in daily summary)
    assert f.daily_latest_sent is not None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieval_deterministic_and_grounded():
    index = build_index(DS)
    for m in DS.incoming[:10]:
        c1 = retrieve_evidence(DS, m, index)
        c2 = retrieve_evidence(DS, m, index)
        assert [c.message_id for c in c1] == [c.message_id for c in c2]
        for c in c1:
            assert c.message_id in {h.message_id for h in DS.history}
            assert 0.0 <= c.score <= 1.5
            assert ("same_sender" in c.tags or "same_business" in c.tags
                    or "early_anchor" in c.tags)


def test_evidence_ids_valid_format():
    index = build_index(DS)
    for m in DS.incoming:
        cands = retrieve_evidence(DS, m, index, k=3)
        ev = choose_evidence_ids(cands)
        if ev == "none":
            assert not cands or all(c.score < 0.30 for c in cands)
        else:
            for eid in ev.split(";"):
                assert eid in {h.message_id for h in DS.history}


def test_evidence_none_for_unknown_sender():
    # synthetic first-contact personal message (no history from u_099)
    from core.data_loader import Message

    fake = Message(
        message_id="msg_fake",
        user_id="u_001",
        conversation_type="personal",
        group_id=None,
        business_id=None,
        sender_user_id="u_099",
        created_at=datetime(2026, 7, 31, 10, 0),
        message_text="hello, completely new person",
        media_type="",
        media_id=None,
        forwarded_count=0,
    )
    cands = retrieve_evidence(DS, fake)
    assert cands == []
    assert choose_evidence_ids(cands) == "none"


def test_retrieval_sane_top_hit_for_group_notice():
    # sample_msg_001's real incoming twin (group_002, sender u_043) should
    # surface history from the same sender with non-zero similarity.
    msg = next(
        m
        for m in DS.incoming
        if m.conversation_type == "group" and m.group_id == "group_002"
        and m.sender_user_id == "u_043"
    )
    cands = retrieve_evidence(DS, msg, k=3)
    assert cands
    assert all(c.created_at < msg.created_at for c in cands)  # no future leaks
