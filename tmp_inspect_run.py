from pathlib import Path
p = Path('mlruns/1/39cfb596a60548d89e85564e57789e0c')
print('run dir exists', p.exists())
for f in ['meta.yaml', 'params', 'metrics', 'tags', 'meta.json']:
    fp = p / f
    if fp.exists():
        print(f'---- {f} ----')
        print(fp.read_text()[:2000])
