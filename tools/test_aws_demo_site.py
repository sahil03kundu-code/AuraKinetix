"""Offline contract tests for the AWS-hostable, truthful static showcase."""
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []
        self.has_title = False
        self.has_description = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a":
            self.links.append(attributes.get("href", ""))
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if tag == "title":
            self.has_title = True
        if tag == "meta" and attributes.get("name") == "description":
            self.has_description = True


class AwsDemoSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = (ROOT / "aws/demo-site/index.html").read_text(encoding="utf-8")
        cls.spec = (ROOT / "amplify.yml").read_text(encoding="utf-8")
        cls.parser = SiteParser()
        cls.parser.feed(cls.page)

    def test_site_is_accessible_static_html(self):
        self.assertIn('<html lang="en">', self.page)
        self.assertIn('name="viewport"', self.page)
        self.assertTrue(self.parser.has_title)
        self.assertTrue(self.parser.has_description)
        self.assertEqual(len(self.parser.ids), len(set(self.parser.ids)))
        self.assertIn("#system", self.parser.links)
        self.assertIn("#limitations", self.parser.links)

    def test_reviewers_get_real_repository_and_evidence_links(self):
        self.assertIn("https://github.com/sahil03kundu-code/AuraKinetix", self.parser.links)
        self.assertIn("https://github.com/sahil03kundu-code/AuraKinetix/actions", self.parser.links)
        self.assertIn("https://github.com/sahil03kundu-code/AuraKinetix/issues/21", self.parser.links)

    def test_discloses_simulation_and_no_hardware_claims(self):
        self.assertIn("This is a public project showcase, not a live control system.", self.page)
        self.assertIn("Sub-millisecond physical recovery", self.page)
        self.assertIn("have not been established", self.page)
        self.assertIn("does not perform inference or control a physical device", self.page)

    def test_no_embedded_secrets_external_js_or_auto_cloud_provisioning(self):
        self.assertNotIn("<script", self.page.lower())
        self.assertNotIn("aws_secret_access_key", self.page.lower())
        self.assertNotIn("api_key=", self.page.lower())
        self.assertNotIn("cloudformation deploy", self.spec)
        self.assertNotIn("npm install", self.spec)

    def test_amplify_builds_only_static_showcase(self):
        self.assertIn("version: 1", self.spec)
        self.assertIn("cp aws/demo-site/index.html dist/index.html", self.spec)
        self.assertIn("tools/build_amplify_manifest.py", self.spec)
        self.assertIn("dist/deployment-manifest.json", self.spec)
        self.assertIn("baseDirectory: dist", self.spec)
        self.assertIn("'**/*'", self.spec)


if __name__ == "__main__":
    unittest.main()
