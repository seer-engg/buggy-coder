from src.graph import (
	add_import_buggy,
	app,
	apply_fixes,
	bump_indices_off_by_one,
	rename_first_occurrence,
	stub_function_singleline,
)


def test_add_import_inserts_module_once():
	snippet = "def foo():\n    return 1\n"
	result = add_import_buggy(snippet, "math")
	assert result == "import math\ndef foo():\n    return 1"
	# idempotency
	again = add_import_buggy(result, "math")
	assert again == result


def test_rename_preserves_substrings_and_strings():
	snippet = "def foo():\n    bar = 1\n    foobar = 'bar'\n    return bar, foobar\n"
	result = rename_first_occurrence(snippet, "bar", "baz")
	expected = "def foo():\n    baz = 1\n    foobar = 'bar'\n    return baz, foobar"
	assert result == expected


def test_stub_function_keeps_signature():
	snippet = "def compute(value, limit): pass\n"
	result = stub_function_singleline(snippet)
	expected = "def compute(value, limit):\n    raise NotImplementedError(\"Todo: implement function\")"
	assert result == expected


def test_fix_indexing_is_safe_noop():
	snippet = "values = [1, 2, 3]\nresult = values[1]\n"
	assert bump_indices_off_by_one(snippet) == "values = [1, 2, 3]\nresult = values[1]"


def test_type_conversion_fix():
	snippet = "def add(a, b):\n    return a + b\n"
	instructions = "Perform type conversion by casting both inputs to int before addition."
	result = apply_fixes(snippet, instructions)
	assert "a = int(a)" in result
	assert "b = int(b)" in result
	assert result.endswith("return a + b")


def test_reports_syntax_errors():
	snippet = "def broken(:\n    pass\n"
	result = apply_fixes(snippet, "Fix the syntax error.")
	assert result.startswith("SyntaxError:")
	assert "line" in result


def test_division_by_zero_guard():
	snippet = "def divide(a, b):\n    return a / b\n"
	instructions = "Avoid division by zero by returning 'undefined'."
	result = apply_fixes(snippet, instructions)
	assert "if b == 0" in result
	assert "return 'undefined'" in result


def test_empty_list_guard():
	snippet = "def first(items):\n    return items[0]\n"
	instructions = "Handle the empty list case by returning None."
	result = apply_fixes(snippet, instructions)
	assert "if not items" in result
	assert "return None" in result


def test_agent_invoke_uses_apply_fixes():
	snippet = "def add(a, b):\n    return a + b\n"
	instructions = "Perform type conversion to int."
	expected = apply_fixes(snippet, instructions)
	state = {"snippet": snippet, "instructions": instructions}
	response = app.invoke(state)
	assert response["snippet"] == expected
	assert response["messages"][-1].content == expected


def test_type_conversion_respects_docstring_and_existing_casts():
	snippet = (
		"def scale(value, factor, label):\n"
		"    \"\"\"Scale a value.\"\"\"\n"
		"    value = float(value)\n"
		"    return value * factor\n"
	)
	instructions = "Ensure numeric inputs are converted to float before any multiplication."
	result = apply_fixes(snippet, instructions)
	assert result.count("float(value)") == 1
	assert "factor = float(factor)" in result
	assert "label = float(label)" not in result
	lines = result.splitlines()
	docstring_idx = lines.index('    """Scale a value."""')
	assert lines[docstring_idx + 1] == "    factor = float(factor)"


def test_division_guard_uses_default_message_when_unspecified():
	snippet = "def divide(total, divisor):\n    return total / divisor\n"
	instructions = "Guard against division by zero in the calculation."
	result = apply_fixes(snippet, instructions)
	assert "if divisor == 0" in result
	assert "return 'Cannot divide by zero'" in result
	assert result.strip().endswith("return total / divisor")


def test_empty_list_guard_can_return_empty_list_literal():
	snippet = "def head(items):\n    return items[0]\n"
	instructions = "Return [] when an empty list is provided."
	result = apply_fixes(snippet, instructions)
	assert "if not items" in result
	assert "return []" in result
