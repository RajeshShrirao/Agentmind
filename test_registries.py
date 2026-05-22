import unittest
import json
import os
from decode import TOOL_REGISTRY, validate_tool_call

CEREBRAS_KEY_MISSING = not os.environ.get("CEREBRAS_API_KEY")

ALL_TOOLS_SORTED = sorted([
    "web_search", "read_file", "write_file", "run_python", "get_weather",
    "search_arxiv", "fetch_abstract", "execute_sql", "send_email", "git_commit",
    "list_directory", "get_stock_price", "translate", "summarize",
])

def _parse_generate_synthetic_tools():
    """Parse TOOLS list from generate_synthetic.py via AST (avoids Cerebras import)."""
    import ast
    with open("generate_synthetic.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(t.id == "TOOLS" for t in node.targets if isinstance(t, ast.Name)):
            names = []
            for elem in node.value.elts:
                name_idx = next(i for i, k in enumerate(elem.keys) if k.value == "name")
                names.append(elem.values[name_idx].value)
            return sorted(names)
    raise RuntimeError("Could not find TOOLS in generate_synthetic.py")

class TestToolRegistryParity(unittest.TestCase):

    def test_decode_registry_has_all_14_tools(self):
        names = sorted(TOOL_REGISTRY.keys())
        self.assertEqual(names, ALL_TOOLS_SORTED,
                         f"Missing tools in TOOL_REGISTRY: {set(ALL_TOOLS_SORTED) - set(names)}")

    def test_synthetic_registry_has_all_14_tools(self):
        from data.synthetic import SYNTHETIC_TOOLS
        names = sorted(SYNTHETIC_TOOLS.keys())
        self.assertEqual(names, ALL_TOOLS_SORTED,
                         f"Missing tools in SYNTHETIC_TOOLS: {set(ALL_TOOLS_SORTED) - set(names)}")

    def test_scaled_registry_has_all_14_tools(self):
        from generate_scaled_synthetic import TOOLS
        names = sorted(TOOLS.keys())
        self.assertEqual(names, ALL_TOOLS_SORTED,
                         f"Missing tools in scaled TOOLS: {set(ALL_TOOLS_SORTED) - set(names)}")

    def test_simple_registry_has_all_14_tools(self):
        names = _parse_generate_synthetic_tools()
        self.assertEqual(names, ALL_TOOLS_SORTED,
                         f"Missing tools in simple TOOLS: {set(ALL_TOOLS_SORTED) - set(names)}")

    def test_search_arxiv_days_is_integer_everywhere(self):
        from data.synthetic import SYNTHETIC_TOOLS
        self.assertIs(SYNTHETIC_TOOLS["search_arxiv"]["args"]["days"], int)
        self.assertEqual(TOOL_REGISTRY["search_arxiv"]["params"]["days"]["type"], "integer")

    def test_tool_names_identical_across_all_registries(self):
        from data.synthetic import SYNTHETIC_TOOLS
        from generate_scaled_synthetic import TOOLS as SCALED_TOOLS

        decode_names = set(TOOL_REGISTRY.keys())
        synth_names = set(SYNTHETIC_TOOLS.keys())
        scaled_names = set(SCALED_TOOLS.keys())
        simple_names = set(_parse_generate_synthetic_tools())

        self.assertEqual(decode_names, synth_names,
                         f"Diff decode vs synth: {decode_names ^ synth_names}")
        self.assertEqual(decode_names, scaled_names,
                         f"Diff decode vs scaled: {decode_names ^ scaled_names}")
        self.assertEqual(decode_names, simple_names,
                         f"Diff decode vs simple: {decode_names ^ simple_names}")


class TestSyntheticTypeCorrectness(unittest.TestCase):

    def test_each_tool_constructs_valid_call(self):
        for name, schema in TOOL_REGISTRY.items():
            args = {}
            for pname, pschema in schema["params"].items():
                ptype = pschema["type"]
                if ptype == "string":
                    args[pname] = f"test_{pname}"
                elif ptype == "integer":
                    args[pname] = 42
                elif ptype == "number":
                    args[pname] = 3.14
                elif ptype == "boolean":
                    args[pname] = True
            call = json.dumps({"name": name, "args": args})
            r = validate_tool_call(call)
            self.assertTrue(r["valid"], f"Tool '{name}' should be valid: {r.get('error')}")


if __name__ == "__main__":
    unittest.main()
