

def add_gcc_to_dockerfile(model_dir):
    with open(model_dir / 'Dockerfile', 'r') as f:
        lines = f.readlines()
    for ind, line in enumerate(lines):
        if line.startswith('RUN apt-get -y update && apt-get install -y --no-install-recommends '):
            lines[ind] = line.split('\n')[0] + ' -y gcc g++\n'
    with open(model_dir / 'Dockerfile', 'w') as f:
        f.writelines(lines)
