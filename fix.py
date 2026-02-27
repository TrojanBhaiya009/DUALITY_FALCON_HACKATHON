files = ['check_data.py', 'train.py', 'test.py', 'visualize.py']
for fname in files:
    with open(fname, 'r', encoding='utf-8') as f:
        c = f.read()
    c = c.replace('data/train', 'train').replace('data/val', 'val').replace('data/test', 'test')
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(c)
print('Done! All paths fixed!')