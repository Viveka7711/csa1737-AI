from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# -------------------------
#  Dataset
# -------------------------
outlook =    ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny',
              'Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']

temperature = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
               'Cool','Mild','Mild','Mild','Hot','Mild']

humidity =    ['High','High','High','High','Normal','Normal','High','High',
               'Normal','Normal','Normal','High','Normal','High']

windy =       ['False','True','False','False','False','True','True','False',
               'False','False','True','True','False','True']

play =        ['No','No','Yes','Yes','Yes','No','Yes','No',
               'Yes','Yes','Yes','Yes','Yes','No']

# -------------------------
# Label Encoding
# -------------------------
le_outlook = LabelEncoder()
le_temp    = LabelEncoder()
le_humid   = LabelEncoder()
le_windy   = LabelEncoder()
le_play    = LabelEncoder()

o = le_outlook.fit_transform(outlook)
t = le_temp.fit_transform(temperature)
h = le_humid.fit_transform(humidity)
w = le_windy.fit_transform(windy)
p = le_play.fit_transform(play)

# -------------------------
# Combine features
# -------------------------
X = [[o[i], t[i], h[i], w[i]] for i in range(len(o))]

# -------------------------
# Train Decision Tree
# -------------------------
clf = DecisionTreeClassifier()
clf.fit(X, p)

# -------------------------
# Print Dataset (Readable)
# -------------------------
print("=== Original Dataset ===")
for i in range(len(outlook)):
    print(f"{i+1:2d}  Outlook={outlook[i]:8s}  Temp={temperature[i]:4s}  "
          f"Humidity={humidity[i]:6s}  Windy={windy[i]:5s}  Play={play[i]}")

# -------------------------
# Print Encoded Values
# -------------------------
print("\n=== Encoded Values ===")
print("Outlook Encoding:", list(le_outlook.classes_), "->", set(o))
print("Temp Encoding:   ", list(le_temp.classes_), "->", set(t))
print("Humidity Encoding:", list(le_humid.classes_), "->", set(h))
print("Wind Encoding:    ", list(le_windy.classes_), "->", set(w))
print("Play Encoding:    ", list(le_play.classes_), "->", set(p))

# -------------------------
# Print Decision Tree Rules
# -------------------------
print("\n=== Decision Tree Rules ===")
print(export_text(clf, feature_names=['Outlook', 'Temperature', 'Humidity', 'Windy']))
