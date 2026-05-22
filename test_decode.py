import unittest
import json
from decode import (
    validate_tool_call,
    extract_tool_calls,
    tool_eval_report,
    TOOL_REGISTRY,
)


class TestValidateToolCall(unittest.TestCase):

    def setUp(self):
        self.valid_call = json.dumps({"name": "get_weather", "args": {"city": "Tokyo"}})

    def test_valid_call(self):
        r = validate_tool_call(self.valid_call)
        self.assertTrue(r["valid"])
        self.assertEqual(r["name"], "get_weather")
        self.assertEqual(r["args"], {"city": "Tokyo"})
        self.assertIsNone(r["failure_mode"])

    def test_empty_input(self):
        r = validate_tool_call("")
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "parse_error")

    def test_malformed_json(self):
        r = validate_tool_call('{"name": "get_weather" "args": {}}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "parse_error")

    def test_missing_name(self):
        r = validate_tool_call('{"args": {"city": "Tokyo"}}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "missing_name")

    def test_name_not_string(self):
        r = validate_tool_call('{"name": 42, "args": {}}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "missing_name")

    def test_unknown_tool(self):
        r = validate_tool_call('{"name": "fly_to_mars", "args": {}}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "unknown_tool")
        self.assertEqual(r["name"], "fly_to_mars")

    def test_missing_args(self):
        r = validate_tool_call('{"name": "get_weather"}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "missing_args")

    def test_args_not_object(self):
        r = validate_tool_call('{"name": "get_weather", "args": "sunny"}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "missing_args")

    def test_missing_required_param(self):
        r = validate_tool_call('{"name": "get_weather", "args": {}}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "missing_param")
        self.assertIn("city", r["error"])

    def test_type_mismatch_int_for_string(self):
        r = validate_tool_call('{"name": "get_weather", "args": {"city": 42}}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "type_mismatch")

    def test_type_mismatch_string_for_int(self):
        r = validate_tool_call('{"name": "search_arxiv", "args": {"query": "SSM", "days": "seven"}}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "type_mismatch")

    def test_optional_param_omitted(self):
        r = validate_tool_call('{"name": "search_arxiv", "args": {"query": "SSM"}}')
        self.assertTrue(r["valid"])
        self.assertIsNone(r["failure_mode"])

    def test_extra_params_ignored(self):
        r = validate_tool_call('{"name": "get_weather", "args": {"city": "Tokyo", "extra": "stuff"}}')
        self.assertTrue(r["valid"])
        self.assertIsNone(r["failure_mode"])

    def test_strips_trailing_observe_token(self):
        r = validate_tool_call('{"name": "get_weather", "args": {"city": "Tokyo"}}<|observe|>')
        self.assertTrue(r["valid"])
        self.assertIsNone(r["failure_mode"])

    def test_strips_trailing_eos(self):
        r = validate_tool_call('{"name": "get_weather", "args": {"city": "Tokyo"}}<eos>')
        self.assertTrue(r["valid"])

    def test_extract_handles_leading_tool_call_token(self):
        results = extract_tool_calls('<|tool_call|>{"name": "get_weather", "args": {"city": "Tokyo"}}')
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["valid"])

    def test_all_registry_tools_validate(self):
        for name, schema in TOOL_REGISTRY.items():
            args = {}
            for pname, pschema in schema["params"].items():
                ptype = pschema["type"]
                if ptype == "string":
                    args[pname] = "test"
                elif ptype == "integer":
                    args[pname] = 1
                elif ptype == "number":
                    args[pname] = 1.0
                elif ptype == "boolean":
                    args[pname] = True
            call = json.dumps({"name": name, "args": args})
            r = validate_tool_call(call)
            self.assertTrue(r["valid"], f"tool {name} should be valid: {r['error']}")


class TestExtractToolCalls(unittest.TestCase):

    def test_single_call(self):
        text = "<|tool_call|>{\"name\": \"get_weather\", \"args\": {\"city\": \"Tokyo\"}}"
        results = extract_tool_calls(text)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["valid"])

    def test_multiple_calls(self):
        text = (
            "<|tool_call|>{\"name\": \"search_arxiv\", \"args\": {\"query\": \"SSM\"}}"
            "<|observe|>results"
            "<|tool_call|>{\"name\": \"fetch_abstract\", \"args\": {\"id\": \"123\"}}"
        )
        results = extract_tool_calls(text)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["valid"])
        self.assertTrue(results[1]["valid"])
        self.assertEqual(results[0]["name"], "search_arxiv")
        self.assertEqual(results[1]["name"], "fetch_abstract")

    def test_no_tool_call(self):
        text = "Just a normal response without tools."
        results = extract_tool_calls(text)
        self.assertEqual(len(results), 0)

    def test_mixed_valid_and_invalid(self):
        text = (
            "<|tool_call|>{\"name\": \"get_weather\", \"args\": {\"city\": \"Tokyo\"}}"
            "<|tool_call|>invalid json here"
        )
        results = extract_tool_calls(text)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["valid"])
        self.assertFalse(results[1]["valid"])
        self.assertEqual(results[1]["failure_mode"], "parse_error")


class TestToolEvalReport(unittest.TestCase):

    def test_empty_report(self):
        r = tool_eval_report([])
        self.assertEqual(r["total"], 0)
        self.assertEqual(r["valid"], 0)
        self.assertEqual(r["valid_pct"], 0.0)

    def test_all_valid(self):
        results = [
            {"valid": True, "name": "get_weather", "args": {}, "failure_mode": None},
            {"valid": True, "name": "search_arxiv", "args": {}, "failure_mode": None},
        ]
        r = tool_eval_report(results)
        self.assertEqual(r["total"], 2)
        self.assertEqual(r["valid"], 2)
        self.assertEqual(r["valid_pct"], 100.0)
        self.assertEqual(r["breakdown"], {})

    def test_failure_breakdown(self):
        results = [
            {"valid": False, "name": None, "args": None, "failure_mode": "parse_error"},
            {"valid": False, "name": "unknown", "args": None, "failure_mode": "unknown_tool"},
            {"valid": False, "name": "get_weather", "args": {}, "failure_mode": "missing_param"},
            {"valid": True, "name": "get_weather", "args": {}, "failure_mode": None},
        ]
        r = tool_eval_report(results)
        self.assertEqual(r["total"], 4)
        self.assertEqual(r["valid"], 1)
        self.assertEqual(r["valid_pct"], 25.0)
        self.assertEqual(r["breakdown"], {"missing_param": 1, "parse_error": 1, "unknown_tool": 1})

    def test_tool_counts(self):
        results = [
            {"valid": True, "name": "get_weather", "args": {}, "failure_mode": None},
            {"valid": True, "name": "get_weather", "args": {}, "failure_mode": None},
            {"valid": True, "name": "search_arxiv", "args": {}, "failure_mode": None},
        ]
        r = tool_eval_report(results)
        self.assertEqual(r["tool_counts"], {"get_weather": 2, "search_arxiv": 1})


class TestValidateEdgeCases(unittest.TestCase):

    def test_unicode_in_args(self):
        r = validate_tool_call('{"name": "get_weather", "args": {"city": "München"}}')
        self.assertTrue(r["valid"])

    def test_nested_objects_in_args(self):
        r = validate_tool_call('{"name": "send_email", "args": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}}')
        self.assertTrue(r["valid"])

    def test_empty_string_tool_name(self):
        r = validate_tool_call('{"name": "", "args": {}}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "unknown_tool")

    def test_whitespace_only(self):
        r = validate_tool_call("   ")
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "parse_error")

    def test_extra_top_level_keys(self):
        r = validate_tool_call('{"name": "get_weather", "args": {"city": "Tokyo"}, "extra": "field"}')
        self.assertTrue(r["valid"])

    def test_args_as_array(self):
        r = validate_tool_call('{"name": "get_weather", "args": [1, 2, 3]}')
        self.assertFalse(r["valid"])
        self.assertEqual(r["failure_mode"], "missing_args")


if __name__ == "__main__":
    unittest.main()
