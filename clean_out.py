import os

def clean_files():
    out_dir = 'out'
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    content = f.read()
                marker = "## Benchmarks:"
                if marker in content:
                    start = content.find(marker)
                    new_content = content[start:]
                    with open(path, 'w') as f:
                        f.write(new_content)

if __name__ == "__main__":
    clean_files()
