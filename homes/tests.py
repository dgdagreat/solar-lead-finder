from unittest.mock import patch

from django.test import TestCase, Client

from homes.models import Home
from homes import services


class HaversineTests(TestCase):
    def test_known_distances(self):
        sf = (37.7793, -122.4193)
        # SF City Hall -> downtown Oakland ~8 mi; -> downtown LA ~347 mi
        self.assertAlmostEqual(services._haversine_miles(*sf, 37.8044, -122.2712), 8.3, delta=0.6)
        self.assertAlmostEqual(services._haversine_miles(*sf, 34.0522, -118.2437), 347, delta=6)

    def test_zero_distance(self):
        self.assertAlmostEqual(services._haversine_miles(40, -100, 40, -100), 0.0, delta=0.01)


class BeyondRadiusTests(TestCase):
    @staticmethod
    def _prop(lat, lon):
        return {"location": {"address": {"coordinate": {"lat": lat, "lon": lon}}}}

    def test_within_and_beyond(self):
        sf = (37.7793, -122.4193)
        oakland = self._prop(37.8044, -122.2712)        # ~8 mi away
        self.assertFalse(services._beyond_radius(oakland, sf, 10))   # keep
        self.assertTrue(services._beyond_radius(oakland, sf, 5))     # drop

    def test_missing_coords_fail_open(self):
        # No coordinates -> don't filter the home out
        self.assertFalse(services._beyond_radius({"location": {}}, (37.7, -122.4), 5))


class ParseRapidApiTests(TestCase):
    def test_single_family_is_parsed(self):
        prop = {
            "location": {"address": {"line": "123 Main St", "city": "Reseda",
                                     "state_code": "CA", "postal_code": "91335"}},
            "description": {"type": "single_family"},
            "last_sold_price": 750000,
            "last_sold_date": "2026-05-01",
        }
        home = services._parse_rapidapi_property(prop, "Reseda", "CA")
        self.assertIsNotNone(home)
        self.assertEqual(home.address, "123 Main St")
        self.assertEqual(home.zip_code, "91335")
        self.assertEqual(int(home.sale_price), 750000)

    def test_condo_is_skipped(self):
        prop = {"location": {"address": {"line": "9 Tower Rd"}},
                "description": {"type": "condos"}}
        self.assertIsNone(services._parse_rapidapi_property(prop, "LA", "CA"))

    def test_missing_street_is_skipped(self):
        prop = {"location": {"address": {"line": ""}},
                "description": {"type": "single_family"}}
        self.assertIsNone(services._parse_rapidapi_property(prop, "LA", "CA"))


class MockDataTests(TestCase):
    def test_mock_creates_homes(self):
        homes = services._mock_home_data("Glendale", "CA")
        self.assertGreaterEqual(len(homes), 1)
        self.assertTrue(all(isinstance(h, Home) for h in homes))


class HomeModelTests(TestCase):
    def test_full_address(self):
        home = Home(address="1 A St", city="LA", state="CA", zip_code="90001")
        self.assertEqual(home.full_address, "1 A St, LA, CA 90001")


class DashboardViewTests(TestCase):
    def test_dashboard_renders(self):
        self.assertEqual(Client().get("/").status_code, 200)

    @patch("homes.views.process_home_sync")            # no detection / network
    @patch("homes.views.fetch_recently_sold_homes")    # no property API call
    def test_search_scopes_session_to_results(self, mock_fetch, _mock_proc):
        homes = services._mock_home_data("Glendale", "CA")
        mock_fetch.return_value = homes

        client = Client()
        resp = client.post("/search/", {"city": "Glendale", "state": "CA", "radius": "15"})

        self.assertEqual(resp.status_code, 302)        # redirects back to dashboard
        self.assertEqual(set(client.session["current_home_ids"]),
                         {h.id for h in homes})
        self.assertIn("Glendale", client.session["current_search_label"])
