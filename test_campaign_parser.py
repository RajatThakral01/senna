import unittest
from unittest.mock import patch
import os
import json

# Add the parent directory to the path so we can import campaign_parser and main
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from campaign import campaign_parser
from campaign.campaign_parser import validate_campaign_config
from main import load_template, merge_config

DUMMY_LOGO = "assets/logos/_test_dummy_logo.png"
DUMMY_MUSIC = "assets/music/_test_dummy_track.mp3"


class TestCampaignParser(unittest.TestCase):

    def setUp(self):
        """Create uniquely-named dummy assets (never touch real files)."""
        os.makedirs("assets/logos", exist_ok=True)
        os.makedirs("assets/music", exist_ok=True)
        with open(DUMMY_LOGO, "w") as f:
            f.write("dummy logo")
        with open(DUMMY_MUSIC, "w") as f:
            f.write("dummy music")

    def tearDown(self):
        """Remove only the dummy files this test created."""
        for p in (DUMMY_LOGO, DUMMY_MUSIC):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    def _force_mock_path(self):
        # Force the internal mock config regardless of .env contents
        return patch.object(campaign_parser, "LLM_API_KEY", "your_groq_api_key_here")

    def test_parse_campaign_with_mock_api(self):
        print("Testing campaign parser with internal mock...")
        description = "Add our logo to the top right, podcast style."
        assets = {"logos": ["_test_dummy_logo.png"], "music": []}

        with self._force_mock_path():
            config = campaign_parser.parse_campaign(description, assets)

        self.assertIn("campaign_id", config)
        self.assertEqual(config["logo"], "assets/logos/logo.png")
        self.assertEqual(config["logo_position"], "top-right")
        self.assertEqual(config["template"], "podcast_clip")
        print("Campaign parser test passed!")

    def test_parse_campaign_without_api_key(self):
        print("Testing campaign parser with internal mock (no API key)...")
        description = "Add a logo and use podcast style"
        assets = {"logos": ["logo.png"], "music": ["track.mp3"]}
        with self._force_mock_path():
            config = campaign_parser.parse_campaign(description, assets)

        self.assertEqual(config["campaign_id"], "mock_id")
        self.assertEqual(config["logo"], "assets/logos/logo.png")
        self.assertEqual(config["template"], "podcast_clip")
        print("Internal mock test passed!")

    def test_validate_layout(self):
        self.assertEqual(validate_campaign_config({"layout": "blur"})["layout"], "auto")
        self.assertEqual(
            validate_campaign_config({"layout": "SPEAKER_CROP"})["layout"], "speaker_crop")
        cfg = validate_campaign_config({"layout": "auto", "split_screen": True})
        self.assertEqual(cfg["layout"], "stacked_split")

    def test_merge_config(self):
        print("Testing config merging logic...")
        template = load_template("podcast_clip")
        campaign_config = {
            "logo": "my_logo.png",
            "min_clip_duration": 60,  # This should override the template's 45
            "music": None  # This should not override the template's null
        }

        merged = merge_config(template, campaign_config)

        self.assertEqual(merged["subtitle_style"], "karaoke")  # from template
        self.assertEqual(merged["logo"], "my_logo.png")  # from campaign
        self.assertEqual(merged["min_clip_duration"], 60)  # from campaign (override)
        self.assertEqual(merged["fade"], True)  # from template
        print("Config merging test passed!")


if __name__ == '__main__':
    unittest.main()
