from swe_agent.retrieval.tokenizer import tokenize_code


def test_snake_case_keeps_whole_name_and_parts():
    assert tokenize_code("get_user_by_email") == [
        "get_user_by_email", "get", "user", "by", "email",
    ]


def test_camel_case_is_split():
    assert tokenize_code("ZeroDivisionError") == [
        "zerodivisionerror", "zero", "division", "error",
    ]
    assert tokenize_code("parseHTTPResponse") == [
        "parsehttpresponse", "parse", "http", "response",
    ]


def test_plain_word_is_one_token():
    assert tokenize_code("average") == ["average"]


def test_stopwords_are_dropped_but_can_be_kept():
    assert tokenize_code("how do we compute the mean") == ["compute", "mean"]
    assert "the" in tokenize_code("the mean", drop_stopwords=False)


def test_dunder_and_lone_underscores():
    assert tokenize_code("__init__") == ["__init__", "init"]
    assert tokenize_code("_ __") == []