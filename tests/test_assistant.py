"""Regression tests for Part 1.   python -m unittest discover -s tests -v   (standard library only)"""
import unittest

from assistant.qa import ask


class HardCases(unittest.TestCase):
    def test_straight_lookup(self):
        a = ask("What's the base price of a 2-bed in Block B?")
        self.assertIn("22,425,000", a.headline)
        self.assertTrue(any("Base Prices (Block B)" in s for s in a.sources))

    def test_stacked_premiums_floor15_corner_margalla(self):
        a = ask("What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?")
        text = a.text()
        self.assertIn("25,340,250", text)      # 2-Bed Standard base 22,425,000 x 1.13
        self.assertIn("30,346,150", text)      # 2-Bed Corner base 26,855,000 x 1.13
        self.assertIn("+13%", text)            # 4 + 3 + 6, cumulative

    def test_discount_stacking_and_booking(self):
        a = ask("An overseas Pakistani buyer pays 50% upfront on a 3-Bed Executive - what's the price after discounts, and what's due at booking?")
        self.assertIn("36,321,600", a.headline)   # 39,480,000 x (1 - 6% - 2%)
        self.assertIn("3,632,160", a.headline)    # 10% token
        self.assertTrue(any("Discounts" in s for s in a.sources) and any("Base Prices (Block B)" in s for s in a.sources))

    def test_transfer_fee_conflict_shows_both(self):
        a = ask("What's the transfer fee?")
        self.assertEqual(a.status, "conflict")
        self.assertIn("2%", a.text())
        self.assertIn("2.5%", a.text())
        self.assertIn("management", a.text())

    def test_transfer_fee_conflict_also_flagged_in_retrieval(self):
        a = ask("can i transfer my unit to someone else?")
        self.assertIn("CONFLICT", a.text())

    def test_rental_yield_refused(self):
        a = ask("What's the rental yield on a 1-bed?")
        self.assertEqual(a.status, "refused")
        self.assertIn("marketing manager", a.text())
        self.assertNotRegex(a.headline, r"\d+(\.\d+)?\s?%")      # no invented number

    def test_anchor_tenant_unconfirmed(self):
        a = ask("Who is the anchor tenant?")
        self.assertEqual(a.status, "unconfirmed")
        self.assertIn("no confirmed anchor tenant", a.headline.lower())


class Guards(unittest.TestCase):
    def test_every_answer_has_a_source(self):
        for q in ["base price of studio block A", "who is the anchor tenant", "what is the transfer fee", "payment plan",
                  "what is the weather", "rental yield", "possession date for block A"]:
            self.assertTrue(ask(q).sources, q)

    def test_off_topic_is_not_answered(self):
        for q in ["what's the weather in Islamabad", "Who is the CEO of MGC?", "what is the capital of France", "tell me a joke"]:
            self.assertEqual(ask(q).status, "not_found", q)

    def test_impossible_configurations(self):
        self.assertEqual(ask("price of a margalla facing studio").status, "refused")
        self.assertEqual(ask("price of a 3-bed on floor 30").status, "refused")
        self.assertEqual(ask("price of a 2-bed on floor 2").status, "refused")      # commercial podium
        self.assertEqual(ask("what is the base price of a studio in block B").status, "not_found")

    def test_missing_block_asks_instead_of_guessing(self):
        a = ask("how much is a 1-bed")
        self.assertEqual(a.status, "needs_info")
        self.assertIn("13,320,000", a.text())
        self.assertIn("13,680,000", a.text())

    def test_penthouse_floor_premium(self):
        self.assertIn("94,588,000", ask("price of a penthouse on floor 21").headline)    # 88.4M x 1.07


if __name__ == "__main__":
    unittest.main()
