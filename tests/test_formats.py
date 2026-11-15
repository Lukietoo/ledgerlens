import csv

import pandas as pd
import pytest

from ledgerlens.boilerplate import learn_boilerplate, strip_boilerplate
from ledgerlens.format_eval import format_rows
from ledgerlens.normalize import FORMAT_RULES, disabled_rules, normalize
from ledgerlens.synth import Simulator, write_csv

WF = "PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BTL COF OAKLAND CA S382074619283746 CARD 4821"


def test_disabled_rules_switch_off_and_restore():
    assert normalize(WF).clean == "BLUE BTL COF"
    with disabled_rules(FORMAT_RULES["card:wf"]):
        off = normalize(WF).clean
    assert "PURCHASE AUTHORIZED" in off and "CARD" in off  # the wrapper survives without its rule
    assert normalize(WF).clean == "BLUE BTL COF"  # restored afterwards
    with pytest.raises(KeyError):
        with disabled_rules(["no_such_rule"]):
            pass


def test_every_format_rule_exists():
    with disabled_rules({r for rules in FORMAT_RULES.values() for r in rules}):
        pass  # raises KeyError if FORMAT_RULES names a rule that doesn't exist


def test_date_survives_url_rule():
    # regression: _URL_PATH used to run first and turn "11/01" into "11"
    with disabled_rules(FORMAT_RULES["card:wf"]):
        assert "11" not in normalize("PURCHASE AUTHORIZED ON 11/01 KUNG FU TEA DUBLIN CA").clean.split()


def _statement():
    # letter-distinct names: learning ignores digits, so "SHOP1".."SHOP30" would be one line
    merchants = [f"SHOP{a}{b} PLACE" for a in "ABCDEF" for b in "GHIJK"]
    lines = [f"PURCHASE AUTHORIZED ON 0{i % 9 + 1}/1{i % 9} {m} OAKLAND CA S38207461928{i:04d} CARD 4821"
             for i, m in enumerate(merchants)]
    lines += [f"PURCHASE AUTHORIZED ON 03/14 STARBUCKS STORE 0{k} OAKLAND CA S3820746192{k:05d} CARD 4821"
              for k in range(3)] * 40  # a heavy Starbucks habit: 120 lines, only 3 distinct
    lines += ["PURCHASE AUTHORIZED ON 03/15 TRADER JOE S OAKLAND CA S382074619280001 CARD 4821"] * 5
    return lines


def test_learns_wrapper_not_favorite_store():
    bp = learn_boilerplate(_statement())
    assert {"PURCHASE", "AUTHORIZED", "ON", "CARD", "OAKLAND"} <= bp
    assert "STARBUCKS" not in bp  # 120 raw lines but only 3 distinct ones
    assert "S" not in bp  # single letters never: "TRADER JOE S" must keep its S
    with disabled_rules(FORMAT_RULES["card:wf"]):
        assert normalize(strip_boilerplate(_statement()[-1], bp)).clean == "TRADER JOE S"


def test_strip_handles_colon_fields_and_small_statements():
    bp = {"DES", "ID", "INDN", "CO", "PPD", "BRIAN", "JOHNSON"}
    line = "APPFOLIO INC DES:PAYMENT ID:SKF4TJC6F INDN:BRIAN JOHNSON CO ID:9222898641 PPD"
    with disabled_rules(FORMAT_RULES["ach:boa"]):
        assert normalize(strip_boilerplate(line, bp)).clean == "APPFOLIO INC"
    assert learn_boilerplate(["A B", "C D"]) == set()  # too few distinct lines to learn anything
    assert strip_boilerplate("PPD", {"PPD"}) == "PPD"  # never strips a line to nothing


def test_synth_records_bank_formats(tmp_path):
    sim = Simulator(3, 6, 10)
    path = tmp_path / "t.csv"
    write_csv(sim.run(), path, sim.user_styles)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    per_user = {(r["user_id"], r["card_style"], r["ach_style"]) for r in rows}
    assert len(per_user) == 6 and all(c and a for _, c, a in per_user)  # one format pair per user
    write_csv(sim.run(), tmp_path / "u.csv")  # styles optional: columns present but empty
    with (tmp_path / "u.csv").open() as f:
        assert next(csv.DictReader(f))["card_style"] == ""


def test_format_rows_split_card_and_ach():
    df = pd.DataFrame({
        "merchant": ["Bilt Rent", "Starbucks", "Golden Fox", "Zelle", "SoFi Technologies", "Starbucks"],
        "merchant_kind": ["chain", "chain", "local", "template", "template", "template"],
        "category": ["HOUSING", "COFFEE", "DINING", "TRANSFER", "INCOME", "INCOME"],
    })
    # rent + payroll are ACH; a Starbucks *purchase* is card-format even though Starbucks is also an employer
    assert list(format_rows(df, "ach")) == [True, False, False, False, True, True]
    assert list(format_rows(df, "card")) == [False, True, True, False, False, False]


def _two_users():
    wf = [(f"u1", line) for line in _statement()]
    plain = [("u2", f"SHOP{a}{b} PLACE OAKLAND CA") for a in "LMNOP" for b in "QRSTU"]
    return pd.DataFrame(wf + plain, columns=["user_id", "description"])


def test_add_clean_learns_per_user_and_keeps_raw():
    from ledgerlens.features import add_clean

    df = _two_users()
    with disabled_rules(FORMAT_RULES["card:wf"]):  # a bank nobody wrote a rule for
        on = add_clean(df)
        off = add_clean(df, boilerplate=None)
    u1_on, u1_off = on[on.user_id == "u1"], off[off.user_id == "u1"]
    assert not u1_on["name_clean"].str.contains("PURCHASE|AUTHORIZED").any()
    assert u1_off["name_clean"].str.contains("PURCHASE AUTHORIZED").all()  # learner off = old behavior
    assert (on["clean"] == off["clean"]).all()  # category features never see the learner
    assert (on["description"] == df["description"]).all()  # raw text untouched for format features
    # u2 never says PURCHASE, and u1's boilerplate must not leak into u2's cleaning
    assert on.loc[on.user_id == "u2", "name_clean"].str.startswith("SHOP").all()


def test_add_clean_small_batch_unchanged_and_dict_mode():
    from ledgerlens.features import add_clean

    one = pd.DataFrame({"user_id": ["me"], "description": [WF]})
    assert add_clean(one)["name_clean"].iloc[0] == add_clean(one, boilerplate=None)["name_clean"].iloc[0] == "BLUE BTL COF"
    with disabled_rules(FORMAT_RULES["card:wf"]):
        given = add_clean(one, boilerplate={"me": {"PURCHASE", "AUTHORIZED", "ON", "CARD", "OAKLAND", "CA"}})
    assert given["name_clean"].iloc[0] == "BLUE BTL COF"  # words learned elsewhere (full statement) apply


def test_processor_prefixes_are_never_stripped():
    bp = {"SQ", "TST", "POS", "DEBIT"}
    assert strip_boilerplate("POS DEBIT SQ *BLUE BTL COF", bp) == "SQ *BLUE BTL COF"
    assert strip_boilerplate("TST* GOLDEN FOX", bp) == "TST* GOLDEN FOX"
    assert normalize(strip_boilerplate("POS DEBIT SQ *BLUE BTL COF", bp)).processor == "SQUARE"


def test_learner_only_removes_words():
    # regression: stripping RAW text removed the words the rules anchor on (a trailing state
    # code is how a city gets stripped), so 15% of rows GAINED words like "SAN DIEGO".
    # The learner now runs on the normalizer's output and can only remove words.
    from ledgerlens.features import add_clean

    lines = [f"{m} STORE 0{i} SAN DIEGO CA" for i, m in enumerate(
        ["LOWES", "TOPGOLF", "GIANT EAGLE", "PEETS COFFEE", "TARGET", "COSTCO WHSE", "SAFEWAY", "CHIPOTLE",
         "PANERA BREAD", "SUBWAY", "SHELL OIL", "CVS PHARMACY", "WALGREENS", "BEST BUY", "IKEA", "REI",
         "UNIQLO", "SEPHORA", "PETCO", "MACYS", "ZARA", "NIKE"])]
    df = pd.DataFrame({"user_id": "u", "description": lines})
    on, off = add_clean(df), add_clean(df, boilerplate=None)
    assert all(set(a.split()) <= set(b.split()) for a, b in zip(on["name_clean"], off["name_clean"]))
    assert on["name_clean"].iloc[0] == "LOWES"
    assert (on["clean"] == off["clean"]).all()


def test_category_features_identical_with_learner(tmp_path):
    # the shipped design: the learner may change merchant NAMES, never the model's inputs
    from ledgerlens.features import add_clean, history_features, struct_amount_features

    sim = Simulator(5, 4, 10)
    path = tmp_path / "t.csv"
    write_csv(sim.run(), path, sim.user_styles)
    df = pd.read_csv(path)
    on, off = add_clean(df), add_clean(df, boilerplate=None)
    for col in ["clean", "processor", "key"]:
        assert (on[col] == off[col]).all(), col
    pd.testing.assert_frame_equal(struct_amount_features(on), struct_amount_features(off))
    pd.testing.assert_frame_equal(history_features(on), history_features(off))
    assert (on["name_clean"] != off["name_clean"]).any()  # ...while names do change
