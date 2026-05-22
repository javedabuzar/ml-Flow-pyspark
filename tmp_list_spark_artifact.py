from pathlib import Path
p = Path('mlruns/2/3e294f8be4e746cf9b4b43042081a49a/artifacts/model')
print('exists', p.exists())
for x in sorted(p.iterdir()):
    print(x.name, 'dir' if x.is_dir() else 'file')
    if x.is_file():
        print('  size', x.stat().st_size)
