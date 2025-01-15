from coffea.nanoevents.schemas.auto import auto_schema


def test_auto_empty():
    "Test an empty incoming form"

    b = auto_schema({"fields": [], "contents": []})

    assert len(b.form["contents"]) == 0
    assert len(b.form["fields"]) == 0


def test_auto_single_no_structure():
    "Test a single item with no underscore"

    b = auto_schema({"fields": ["pt"], "contents": [{}]})

    assert len(b.form["fields"]) == 1
    assert len(b.form["contents"]) == 1


def test_auto_single_with_structure():
    "Test a single item, that does have an underscore"

    b = auto_schema(
        {
            "fields": ["lep_pt"],
            "contents": [
                {
                    "content": [{"class": "RecordArray"}],
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            ],
        }
    )

    assert "lep" in b.form["fields"]
    assert "pt" in b.form["contents"][0]["content"]["fields"]


def test_auto_multi_with_structure():
    "Test a multiple items, that does have an underscore"

    b = auto_schema(
        {
            "fields": ["lep_pt", "lep_eta", "lep_phi"],
            "contents": [
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            ],
        }
    )

    assert "lep" in b.form["fields"]
    assert "pt" in b.form["contents"][0]["content"]["fields"]
    assert "phi" in b.form["contents"][0]["content"]["fields"]
    assert "eta" in b.form["contents"][0]["content"]["fields"]


def test_auto_multi_and_single_with_structure():
    "Test a multiple items along with a non-structure item, that does have an underscore"

    b = auto_schema(
        {
            "fields": ["lep_pt", "lep_eta", "lep_phi", "fork"],
            "contents": [
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            ],
        }
    )

    assert "lep" in b.form["fields"]
    assert "fork" in b.form["fields"]
    assert (
        b.form["contents"][0]["content"]["parameters"]["__record__"] == "NanoCollection"
    )


def test_auto_4vector_mass():
    "Test a multiple items, that does have an underscore"

    b = auto_schema(
        {
            "fields": [
                "lep_pt",
                "lep_eta",
                "lep_phi",
                "lep_mass",
                "lep_charge",
                "lep_extra",
            ],
            "contents": [
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                # Make sure we can handle an extra item besides!
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            ],
        }
    )

    assert "lep" in b.form["fields"]
    assert "extra" in b.form["contents"][0]["content"]["fields"]
    assert (
        b.form["contents"][0]["content"]["parameters"]["__record__"]
        == "PtEtaPhiMCandidate"
    )


def test_auto_4vector_e():
    "Test a multiple items, that does have an underscore"

    b = auto_schema(
        {
            "fields": [
                "lep_pt",
                "lep_eta",
                "lep_phi",
                "lep_energy",
                "lep_charge",
                "lep_extra",
            ],
            "contents": [
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                # Make sure we can handle an extra item besides!
                {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            ],
        }
    )

    assert "lep" in b.form["fields"]
    assert "extra" in b.form["contents"][0]["content"]["fields"]
    assert (
        b.form["contents"][0]["content"]["parameters"]["__record__"]
        == "PtEtaPhiECandidate"
    )
