import unittest
from unittest.mock import patch
import os
import json

# Add the parent directory to the path so we can import campaign_parser and main
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from campaign import campaign_parser
from main import load_template, merge_config

class TestCampaignParser(unittest.TestCase):

    def setUp(self):
        """Create dummy asset and template files for testing."""
        os.makedirs("assets/logos", exist_ok=True)
        os.makedirs("assets/music", exist_ok=True)
        with open("assets/logos/logo.png", "w") as f:
            f.write("dummy logo")
        with open("assets/music/track.mp3", "w") as f:
            f.write("dummy music")

    def tearDown(self):
        """Clean up dummy files."""
        os.remove("assets/logos/logo.png")
        os.remove("assets/music/track.mp3")
        os.rmdir("assets/logos")
        os.rmdir("assets/music")


    @patch('campaign.campaign_parser.requests.post')
    def test_parse_campaign_with_mock_api(self, mock_post):
        print("Testing campaign parser with mocked API...")

        # The internal mock is what runs when no API key is present.
        # Let's test that logic directly.
        description = "Add our logo to the top right, podcast style."
        assets = {"logos": ["logo.png"], "music": []}
        
        # Ensure no API key is set, so the internal mock is used
        with patch.dict(os.environ, {"GROK_API_KEY": "your_grok_api_key_here"}):
            config = campaign_parser.parse_campaign(description, assets)

        self.assertIn("campaign_id", config)
        self.assertEqual(config["logo"], "assets/logos/logo.png")
        self.assertEqual(config["logo_position"], "top-right")
        self.assertEqual(config["template"], "podcast_clip")
        print("✅ Campaign parser test passed!")

    def test_parse_campaign_without_api_key(self):
        print("Testing campaign parser with internal mock (no API key)...")
        description = "Add a logo and use podcast style"
        assets = {"logos": ["logo.png"], "music": ["track.mp3"]}
        config = campaign_parser.parse_campaign(description, assets)

        self.assertEqual(config["campaign_id"], "mock_id")
        self.assertEqual(config["logo"], "assets/logos/logo.png")
        self.assertEqual(config["template"], "podcast_clip")
        print("✅ Internal mock test passed!")


    def test_merge_config(self):
        print("Testing config merging logic...")
        template = load_template("podcast_clip")
        campaign_config = {
            "logo": "my_logo.png",
            "min_clip_duration": 60, # This should override the template's 45
            "music": None # This should not override the template's null
        }

        merged = merge_config(template, campaign_config)

        self.assertEqual(merged["subtitle_style"], "karaoke") # from template
        self.assertEqual(merged["logo"], "my_logo.png") # from campaign
        self.assertEqual(merged["min_clip_duration"], 60) # from campaign (override)
        self.assertEqual(merged["fade"], True) # from template
        print("✅ Config merging test passed!")

if __name__ == '__main__':
    unittest.main()
