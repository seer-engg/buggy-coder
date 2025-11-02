from src.syntax_tools import fix_python_syntax


def test_fix_python_syntax_adds_missing_colon_inline():
	snippet = "def sum(a, b) return a + b"
	expected = "def sum(a, b): return a + b"
	assert fix_python_syntax(snippet) == expected


def test_fix_python_syntax_adds_missing_colon_multiline():
	snippet = "def greet(name)\n    return f\"Hello, {name}\""
	expected = "def greet(name):\n    return f\"Hello, {name}\""
	assert fix_python_syntax(snippet) == expected


def test_fix_python_syntax_handles_annotations():
	snippet = "def area(radius) -> float return 3.14 * radius * radius"
	expected = "def area(radius) -> float: return 3.14 * radius * radius"
	assert fix_python_syntax(snippet) == expected


def test_fix_python_syntax_no_change_when_syntax_is_valid():
	snippet = "def add(a, b):\n    return a + b"
	assert fix_python_syntax(snippet) == snippet
