import pytest

from ledgerlens.normalize import canon, display_name, normalize


@pytest.mark.parametrize(
    "raw, clean, processor",
    [
        ("PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BOTTLE COF OAKLAND CA S382074619283746 CARD 4821",
         "BLUE BOTTLE COF", "SQUARE"),
        ("CHECKCARD 0413 STARBUCKS STORE 05412 BERKELEY CA 30196686274025326268433", "STARBUCKS", None),
        ("TST* GOLDEN FOX HOUSE PALO ALTO CA", "GOLDEN FOX HOUSE", "TOAST"),
        ("AMZN MKTP US*2K4AB1CD2 AMZN.COM/BILL WA", "AMZN MKTP US AMZN", None),
        ("NETFLIX.COM 866-579-7172 CA", "NETFLIX", None),
        ("SPOTIFY                HELP.SPOTIFY. NY", "SPOTIFY", None),
        ("MACY S #6840           OAKLAND       CA", "MACY S", None),
        ("POS DEBIT LEMONADE INS HELP.LEMONADE WA", "LEMONADE INS", None),
        ("Dd *Doordash Doordash.Com Ca", "DOORDASH", "DOORDASH"),
        ("GUSTO PAYROLL PPD ID: 1234567890", "GUSTO PAYROLL", None),
        ("ZILLOW RENT DES:RENT ID:63YY09OPN INDN:ANDREW ANDERSON CO ID:2369560690 WEB", "ZILLOW RENT", None),
        ("ZELLE TO DANIEL LEE ON 07/08 REF # PP05ZY0GMP", "ZELLE", None),
        ("VENMO PAYMENT 1023456789 WEB ID: 3264681992", "VENMO", None),
        ("GOOGLE *FI 855-836-3987 CA", "GOOGLE FI", None),  # too-short remainder: keep prefix
        ("COSTCO WHSE #8873 SANTA MONICA CA", "COSTCO WHSE", None),
        ("UBER *TRIP HELP.UBER.COM CA", "UBER TRIP", None),
    ],
)
def test_normalize(raw, clean, processor):
    n = normalize(raw)
    assert n.clean == clean
    assert n.processor == processor


def test_unknown_city_is_kept_but_state_dropped():
    # "EL CERRITO" isn't in the gazetteer; we don't guess.
    assert normalize("SAFEWAY #1234 EL CERRITO CA").clean == "SAFEWAY EL CERRITO"


def test_never_empty():
    assert normalize("#12345").clean != ""


def test_canon_and_display():
    assert canon("Trader Joe's") == canon("TRADER JOE S")
    assert display_name("MARIO S GRILL") == "Mario's Grill"
    assert display_name("LUCY S") == "Lucy's"
