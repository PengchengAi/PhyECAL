def update(base, upd):
    if isinstance(upd, dict):
        for key, val in dict(upd).items():
            if key in base:
                if not isinstance(val, dict) or len(val) == 0:
                    base[key] = val
                else:
                    update(base[key], val)
            else:
                base[key] = val
    return base
