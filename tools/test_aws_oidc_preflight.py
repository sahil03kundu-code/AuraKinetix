"""Offline safety tests for the manual-only AWS OIDC identity preflight."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AwsOidcPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = json.loads((ROOT / "aws/oidc-readonly-preflight.json").read_text())
        cls.probe = (ROOT / ".github/workflows/aws-identity-probe.yml").read_text()

    def test_template_creates_only_oidc_provider_and_unprivileged_role(self):
        resources = self.template["Resources"]
        self.assertEqual(
            {x["Type"] for x in resources.values()},
            {"AWS::IAM::OIDCProvider", "AWS::IAM::Role"},
        )
        role = resources["GitHubReadOnlyPreflightRole"]["Properties"]
        self.assertFalse("Policies" in role or "ManagedPolicyArns" in role)
        self.assertLessEqual(role["MaxSessionDuration"], 3600)

    def test_oidc_trust_is_exact_repo_main_and_sts_audience(self):
        oidc = self.template["Resources"]["GitHubOIDCProvider"]["Properties"]
        self.assertEqual(oidc["Url"], "https://token.actions.githubusercontent.com")
        self.assertEqual(oidc["ClientIdList"], ["sts.amazonaws.com"])
        stmts = self.template["Resources"]["GitHubReadOnlyPreflightRole"][
            "Properties"]["AssumeRolePolicyDocument"]["Statement"]
        self.assertEqual(len(stmts), 1)
        stmt = stmts[0]
        self.assertEqual(stmt["Action"], "sts:AssumeRoleWithWebIdentity")
        self.assertEqual(
            stmt["Condition"]["StringEquals"]["token.actions.githubusercontent.com:aud"],
            "sts.amazonaws.com",
        )
        self.assertEqual(
            stmt["Condition"]["StringEquals"]["token.actions.githubusercontent.com:sub"],
            {"Fn::Sub": "repo:${GitHubOwner}/${GitHubRepository}:ref:refs/heads/main"},
        )

    def test_probe_is_manual_only_and_read_only(self):
        workflow = self.probe
        self.assertIn("on:\n  workflow_dispatch:", workflow)
        self.assertNotIn("\n  pull_request:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("id-token: write", workflow)
        self.assertIn("AWS_READONLY_ROLE_ARN", workflow)
        self.assertIn("aws sts get-caller-identity", workflow)
        for forbidden in ("aws ec2 run-instances", "aws cloudformation deploy",
                          "aws s3 mb", "aws iam create-", "aws ecr create-"):
            self.assertNotIn(forbidden, workflow)


if __name__ == "__main__":
    unittest.main()
