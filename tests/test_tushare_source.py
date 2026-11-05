"""_call_paginated must not silently truncate results at a single-call page cap."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from helix.config import Config, DataConfig
from helix.data.store import ParquetStore
from helix.data.tushare_source import TushareSource


class FakePro:
    """Mimics a Tushare interface that caps a single call at ``page_size`` rows."""

    def __init__(self, pages: list[pd.DataFrame], page_size: int):
        self.pages = pages
        self.page_size = page_size
        self.calls: list[dict] = []

    def namechange(self, **kwargs):
        self.calls.append(kwargs)
        idx = kwargs["offset"] // self.page_size
        if idx >= len(self.pages):
            return pd.DataFrame()
        return self.pages[idx]


def make_source(tmp_path, pro) -> TushareSource:
    cfg = Config(data=DataConfig(root=tmp_path))
    src = object.__new__(TushareSource)
    src.cfg = cfg
    src.pro = pro
    src.limiter = SimpleNamespace(acquire=lambda: None)
    src.store = ParquetStore(cfg.data.root)
    return src


def _page(n: int, start: int) -> pd.DataFrame:
    return pd.DataFrame({"ts_code": [f"{i:06d}.SZ" for i in range(start, start + n)]})


def test_call_paginated_stitches_multiple_full_pages_and_stops_on_short_page(tmp_path):
    pages = [_page(5000, 0), _page(5000, 5000), _page(4178, 10000)]
    pro = FakePro(pages, page_size=5000)
    src = make_source(tmp_path, pro)

    result = src._call_paginated("namechange", page_size=5000)

    assert len(result) == 14178
    assert [c["offset"] for c in pro.calls] == [0, 5000, 10000]
    assert all(c["limit"] == 5000 for c in pro.calls)


def test_call_paginated_single_short_page_makes_one_call(tmp_path):
    pages = [_page(120, 0)]
    pro = FakePro(pages, page_size=5000)
    src = make_source(tmp_path, pro)

    result = src._call_paginated("namechange", page_size=5000)

    assert len(result) == 120
    assert len(pro.calls) == 1


def test_call_paginated_exact_page_boundary_probes_one_more_empty_page(tmp_path):
    pages = [_page(5000, 0)]
    pro = FakePro(pages, page_size=5000)
    src = make_source(tmp_path, pro)

    result = src._call_paginated("namechange", page_size=5000)

    assert len(result) == 5000
    assert [c["offset"] for c in pro.calls] == [0, 5000]


def test_call_paginated_empty_result_returns_empty_frame(tmp_path):
    pro = FakePro([], page_size=5000)
    src = make_source(tmp_path, pro)

    result = src._call_paginated("namechange", page_size=5000)

    assert result.empty
    assert len(pro.calls) == 1
