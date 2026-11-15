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
