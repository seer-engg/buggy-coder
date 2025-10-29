from graph import perform_web_search


def test_web_search():
    query = "Python programming tutorials"
    results = perform_web_search(query)
    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Results should not be empty"
    print("Web search test passed.")


test_web_search()
