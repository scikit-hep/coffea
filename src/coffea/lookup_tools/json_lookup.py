class json_lookup:
    def __init__(self, wrapped_values):
        self.values = wrapped_values

    def __call__(self, run, lumi):
        out = []
        for r, ls in zip(run, lumi):
            table = self.values.get(str(r))
            out.append(None if table is None else table.get(ls))
        return out
