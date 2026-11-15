import csv
from collections import defaultdict

from ledgerlens.synth import Simulator, load_chain_merchants, write_csv
from ledgerlens.taxonomy import CATEGORIES


def _gen(tmp_path, seed=1, users=8):
    path = tmp_path / "t.csv"
    write_csv(Simulator(seed, users, 20).run(), path)
    with path.open() as f:
        return list(csv.DictReader(f))


def test_merchant_seed_list_is_valid():
    ms = load_chain_merchants()
    assert len(ms) > 250
    assert len({m.name for m in ms}) == len(ms), "duplicate merchant names"
    for m in ms:
        assert m.category in CATEGORIES
        assert 0 < m.amount_min <= m.amount_max
        assert m.rail in ("card", "ach")


def test_deterministic(tmp_path):
    a = _gen(tmp_path / "a", seed=3)
    b = _gen(tmp_path / "b", seed=3)
    assert [r["description"] for r in a] == [r["description"] for r in b]


def test_labels_and_splits(tmp_path):
    rows = _gen(tmp_path)
    assert len(rows) > 8 * 12 * 30
    assert {r["category"] for r in rows} <= set(CATEGORIES)
    split_of = defaultdict(set)
    for r in rows:
        split_of[r["merchant"]].add(r["split"])
        assert r["description"].strip()
    assert all(len(s) == 1 for s in split_of.values()), "a merchant leaked across splits"
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_card_name_field_truncated_to_22(tmp_path):
    for r in _gen(tmp_path):
        if r["description"].startswith("CHECKCARD "):
            # CHECKCARD MMDD <name:22> <city:13> <ST> <23 digits>
            body = r["description"][len("CHECKCARD 0000 "):]
            assert len(body.split(" ")[0]) <= 22


def test_subscriptions_are_monthly_and_fixed_price(tmp_path):
    rows = [r for r in _gen(tmp_path) if r["category"] == "SUBSCRIPTIONS"]
    by = defaultdict(list)
    for r in rows:
        by[(r["user_id"], r["merchant"])].append(float(r["amount"]))
    for amts in by.values():
        assert len(amts) == 12
        assert len(set(amts)) == 1
