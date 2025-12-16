import importlib
mods = [
    'pandas','numpy','sklearn','plotly','dash','dash_bootstrap_components',
    'requests','bs4','matplotlib','pefile','bitarray','fixedint'
]
have = []
missing = []
for m in mods:
    try:
        importlib.import_module(m)
        have.append(m)
    except Exception as e:
        missing.append((m, str(e)))
print('Environment verification:')
if have:
    print('Present:')
    for m in have:
        print(' -', m)
if missing:
    print('Missing:')
    for m,e in missing:
        print(' -', m, ':', e)
if not missing:
    print('OK: all required imports succeeded')
