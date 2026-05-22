import glob, os
paths = sorted(glob.glob('mlruns/**/artifacts/model', recursive=True), key=os.path.getmtime, reverse=True)
print('spark model candidates:')
for p in paths:
    print(p)
