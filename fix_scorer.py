lines = open(r'c:\Users\kriti\OneDrive\MU HALLOCINATION REDUCTION MODEL\train_commonsense_boost.py', encoding='utf-8').readlines()
new_lines = lines[:314] + [
    '    if qa_lr_w is None:\n',
    '        print("No scorer weights found. Run boost_and_score() first.")\n',
    '        return\n',
    '\n',
    '    # Use the actual number of features the scorer was trained on\n',
    '    scorer_dim = len(qa_lr_w)\n',
    '    print(f"  Using scorer with {scorer_dim} features")\n',
    '\n'
] + lines[315:]
open(r'c:\Users\kriti\OneDrive\MU HALLOCINATION REDUCTION MODEL\train_commonsense_boost.py', 'w', encoding='utf-8').writelines(new_lines)
print('Done')
