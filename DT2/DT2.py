from sklearn.tree import DecisionTreeClassifier, export_text

# Dataset
a1 = ['T', 'T', 'T', 'F', 'F', 'F']
a2 = ['T', 'T', 'F', 'F', 'T', 'T']
y  = ['+', '+', '-', '+', '-', '-']

# Convert T/F to numeric values
X = [
    [1 if a1[i] == 'T' else 0,
     1 if a2[i] == 'T' else 0]
    for i in range(len(a1))
]

# Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Export tree rules
tree_rules = export_text(clf, feature_names=['a1', 'a2'])

# Print results
print("=== Dataset Used ===")
for i in range(len(a1)):
    print(f"Instance {i+1}: a1={a1[i]}, a2={a2[i]}, class={y[i]}")

print("\n=== Decision Tree Rules ===")
print(tree_rules)
