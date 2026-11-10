from unittest.mock import patch

from django.test import TestCase, Client, override_settings

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

    def test_missing_coords_fail_closed(self):
        # Without coordinates the radius cannot be verified, so omit the home.
        self.assertTrue(services._beyond_radius({"location": {}}, (37.7, -122.4), 5))


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
        self.assertEqual(homes[0].city, "Glendale")
        self.assertNotEqual(homes[1].city, "Glendale")

    @override_settings(DEBUG=False, RAPIDAPI_KEY="", ATTOM_API_KEY="")
    def test_production_without_property_api_never_returns_demo_homes(self):
        with self.assertRaises(services.PropertySearchUnavailable) as err:
            services.fetch_recently_sold_homes("Sacramento", "CA")
        self.assertIn("not configured", str(err.exception))

    @override_settings(DEBUG=True, RAPIDAPI_KEY="", ATTOM_API_KEY="")
    def test_demo_data_is_not_used_for_radius_searches(self):
        with self.assertRaises(services.PropertySearchUnavailable):
            services.fetch_recently_sold_homes("Sacramento", "CA", radius_miles=5)

    @override_settings(RAPIDAPI_KEY="", ATTOM_API_KEY="test-attom-key")
    def test_attom_fallback_does_not_silently_ignore_radius(self):
        with self.assertRaises(services.PropertySearchUnavailable) as err:
            services.fetch_recently_sold_homes("Sacramento", "CA", radius_miles=5)
        self.assertIn("Radius filtering", str(err.exception))

    @override_settings(RAPIDAPI_KEY="test-rapid-key")
    @patch("homes.services._geocode", return_value=None)
    @patch("homes.services.requests.post")
    def test_rapidapi_radius_fails_if_search_center_cannot_be_geocoded(self, post, _geocode):
        with self.assertRaises(services.PropertySearchUnavailable) as err:
            services._fetch_from_rapidapi("Sacramento", "CA", None, 20, 5)
        self.assertIn("could not be geocoded", str(err.exception))
        post.assert_not_called()


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

    @patch("homes.views.process_home_sync")
    @patch("homes.views.fetch_recently_sold_homes", side_effect=services.PropertySearchUnavailable(
        "Live property search is not configured."
    ))
    def test_live_search_error_is_shown_instead_of_demo_results(self, _fetch, _process):
        client = Client()
        response = client.post("/search/", {"city": "Sacramento", "state": "CA", "radius": "15"})
        self.assertEqual(response.status_code, 302)
        response = client.get("/")
        self.assertContains(response, "Live property search is not configured.")

    def test_dashboard_discloses_unconfigured_live_search(self):
        with override_settings(DEBUG=False, RAPIDAPI_KEY="", ATTOM_API_KEY=""):
            response = Client().get("/")
        self.assertContains(response, "Live property search is not configured")

    def test_production_dashboard_hides_legacy_demo_homes(self):
        Home.objects.create(
            address="3421 Ramona Ave", city="San Francisco", state="CA",
            zip_code="95826", is_demo=True,
        )
        Home.objects.create(
            address="88 Real St", city="Sacramento", state="CA", zip_code="95814",
        )

        with override_settings(DEBUG=False, RAPIDAPI_KEY="", ATTOM_API_KEY=""):
            response = Client().get("/")

        self.assertNotContains(response, "3421 Ramona Ave")
        self.assertContains(response, "88 Real St")
        self.assertEqual(response.context["total"], 1)
        self.assertEqual(response.context["all_count"], 1)

    def test_mock_data_is_marked_demo(self):
        homes = services._mock_home_data("Glendale", "CA")
        self.assertTrue(all(home.is_demo for home in homes))
