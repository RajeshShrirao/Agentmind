"""
Layer 8 — Environment Simulation.

Use Faker + Mimesis to generate realistic world states:
organizations, invoices, logs, emails, CRM records, stack traces,
filesystem trees, fake APIs, terminals, meeting transcripts.
"""

import random
import json
from .core import parse_tool_calls, apply_positional


class EnvironmentGenerator:
    """Layer 8 — Inject realistic world-state data into observations."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.faker = None
        self._init_faker()

    def _init_faker(self):
        try:
            from faker import Faker
            self.faker = Faker()
            self.faker.seed_instance(self.rng.randint(0, 2**32))
        except Exception:
            pass

    def enrich(self, sample, n_variants=1):
        """Replace sterile observations with realistic environment data."""
        asst = sample["messages"][-1]["content"]
        segments = parse_tool_calls(asst)
        if not segments:
            return [sample]

        results = []
        for _ in range(n_variants):
            new_asst = asst
            for seg in reversed(segments):
                if not seg.call_data or seg.is_failure:
                    continue
                tool = seg.call_data["name"]
                args = seg.call_data["args"]
                realistic = self._realistic_observe(tool, args)
                if realistic:
                    obs_str = json.dumps(realistic)
                    new_seg = f"<|tool_call|>{seg.call_str}<|observe|>{obs_str}"
                    new_asst = apply_positional(new_asst, seg.start, seg.end, new_seg)

            results.append({
                "domain": "tool_caller",
                "type": sample["type"],
                "messages": [
                    {"role": "user", "content": sample["messages"][0]["content"]},
                    {"role": "assistant", "content": new_asst},
                ]
            })
        return results

    def _realistic_observe(self, tool, args):
        """Generate realistic environment output for tool."""
        if not self.faker:
            return None

        try:
            if tool == "web_search":
                return self._fake_web_search(args)
            elif tool == "read_file":
                return self._fake_read_file(args)
            elif tool == "list_directory":
                return self._fake_list_dir(args)
            elif tool == "execute_sql":
                return self._fake_sql(args)
            elif tool == "get_weather":
                return self._fake_weather(args)
            elif tool == "get_stock_price":
                return self._fake_stock(args)
            elif tool == "send_email":
                return self._fake_email(args)
            elif tool == "run_python":
                return self._fake_python(args)
            elif tool == "git_commit":
                return self._fake_git(args)
            elif tool == "translate":
                return self._fake_translate(args)
            elif tool == "search_arxiv":
                return self._fake_arxiv(args)
            elif tool == "fetch_abstract":
                return self._fake_abstract(args)
        except Exception:
            pass
        return None

    def _fake_web_search(self, args):
        query = args.get("query", "research")
        return {"results": [
            {"title": f"{query} — Recent Advances", "url": self.faker.url(), "snippet": self.faker.sentence()},
            {"title": f"Understanding {query}", "url": self.faker.url(), "snippet": self.faker.sentence()},
            {"title": f"{query}: A Comprehensive Guide", "url": self.faker.url(), "snippet": self.faker.sentence()},
        ]}

    def _fake_read_file(self, args):
        path = args.get("path", "file.txt")
        if path.endswith(".py") or path.endswith(".js"):
            return {"content": f"# {path}\n{self.faker.text()}\n\ndef main():\n    print('{self.faker.word()}')"}
        elif path.endswith(".json") or path.endswith(".yaml"):
            return {"content": json.dumps({"name": self.faker.company(), "version": "1.0", "config": {"debug": True, "port": 8080}}, indent=2)}
        elif path.endswith(".csv"):
            return {"content": f"id,name,email\n1,{self.faker.name()},{self.faker.email()}\n2,{self.faker.name()},{self.faker.email()}"}
        elif path.endswith(".log"):
            return {"content": "\n".join([f"2024-01-{d:02d} {self.faker.time()} [INFO] {self.faker.sentence()}" for d in range(1, 6)])}
        return {"content": self.faker.text()}

    def _fake_list_dir(self, args):
        path = args.get("path", "/home/user")
        return {"files": [
            self.faker.file_name() for _ in range(self.rng.randint(3, 8))
        ], "path": path}

    def _fake_sql(self, args):
        query = args.get("query", "SELECT * FROM users")
        if "COUNT" in query.upper():
            return {"rows": [{"count": self.rng.randint(100, 50000)}]}
        if "users" in query.lower() or "employees" in query.lower():
            return {"rows": [
                {"name": self.faker.name(), "email": self.faker.email(), "department": self.faker.bs()}
                for _ in range(self.rng.randint(1, 5))
            ]}
        if "orders" in query.lower() or "sales" in query.lower():
            return {"rows": [
                {"id": self.rng.randint(1000, 9999), "amount": round(self.rng.uniform(10, 5000), 2), "status": self.rng.choice(["pending", "shipped", "delivered"])}
                for _ in range(self.rng.randint(1, 5))
            ]}
        return {"rows": [{"result": self.faker.word()}]}

    def _fake_weather(self, args):
        city = args.get("city", "Unknown")
        return {
            "location": city,
            "temp": self.rng.randint(-5, 40),
            "feels_like": self.rng.randint(-8, 42),
            "condition": self.rng.choice(["sunny", "cloudy", "rainy", "foggy", "windy", "snowy"]),
            "humidity": self.rng.randint(20, 95),
            "wind_speed": round(self.rng.uniform(0, 30), 1),
            "forecast": [
                {"day": "Tomorrow", "temp": self.rng.randint(-5, 40), "condition": self.rng.choice(["sunny", "cloudy", "rainy"])},
            ]
        }

    def _fake_stock(self, args):
        ticker = args.get("ticker", "UNKNOWN")
        price = round(self.rng.uniform(20, 2000), 2)
        change = round(self.rng.uniform(-5, 5), 2)
        return {
            "symbol": ticker.upper(),
            "price": price,
            "change": change,
            "change_percent": f"{round(change/price*100, 2)}%",
            "volume": self.rng.randint(1000000, 50000000),
            "market_cap": f"{round(self.rng.uniform(10, 3000), 1)}B",
        }

    def _fake_email(self, args):
        return {
            "success": True,
            "message_id": f"msg_{self.rng.randint(100000, 999999)}",
            "to": args.get("to", "user@example.com"),
            "subject": args.get("subject", "No Subject"),
            "sent_at": self.faker.date_time_this_month().isoformat(),
        }

    def _fake_python(self, args):
        code = args.get("code", "")
        if "error" in code.lower() or "raise" in code.lower():
            return {"stdout": "", "stderr": f"Traceback (most recent call last):\n  File \"<stdin>\", line 1, in <module>\n{self.faker.word()}: {self.faker.sentence()}"}
        return {"stdout": self.faker.sentence() + "\n", "stderr": ""}

    def _fake_git(self, args):
        return {
            "success": True,
            "hash": self.faker.hexify(text="^^^^^^^"),
            "message": args.get("message", "commit"),
            "branch": self.rng.choice(["main", "develop", "feature/new-dataset", "fix/memory-leak"]),
            "files_changed": self.rng.randint(1, 10),
        }

    def _fake_translate(self, args):
        text = args.get("text", "Hello")
        lang = args.get("target_lang", "es")
        fake_translations = {
            "es": f"Traducción de: {text}",
            "fr": f"Traduction de: {text}",
            "de": f"Übersetzung von: {text}",
            "ja": f"{text}の翻訳",
            "zh": f"{text}的翻译",
            "hi": f"{text} का अनुवाद",
        }
        return {"translated": fake_translations.get(lang, f"[{lang}] {text}"), "detected_language": "en"}

    def _fake_arxiv(self, args):
        query = args.get("query", "machine learning")
        return {"results": [
            {"id": f"24{self.rng.randint(1,9)}.{self.rng.randint(10000, 99999)}", "title": f"{query}: {self.faker.catch_phrase()}", "authors": [self.faker.name() for _ in range(self.rng.randint(2, 5))], "date": self.faker.date_this_year().isoformat()},
            {"id": f"24{self.rng.randint(1,9)}.{self.rng.randint(10000, 99999)}", "title": f"Advances in {query}", "authors": [self.faker.name() for _ in range(self.rng.randint(2, 5))], "date": self.faker.date_this_year().isoformat()},
        ]}

    def _fake_abstract(self, args):
        return {"abstract": self.faker.paragraph(nb_sentences=5), "id": args.get("id", "2405.12345")}
