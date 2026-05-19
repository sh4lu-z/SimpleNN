from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Insert your 100 sentences and Labels here
# Our small dataset (Training Data - 100 items)
texts = [
    # --- Positive Reviews (50) ---
    "Highly recommend this item to everyone",
    "Exceeded my expectations in every way",
    "Great value for the price paid",
    "Five stars, will definitely buy again",
    "The customer service was exceptionally helpful",
    "Beautiful design and perfectly functional",
    "Arrived on time and in perfect condition",
    "This is exactly what I was looking for",
    "Superb quality, feels very premium",
    "Works flawlessly right out of the box",
    "I am completely satisfied with my purchase",
    "Best purchase I have made this year",
    "Incredibly fast shipping and great packaging",
    "Such a wonderful and useful gadget",
    "I cannot stop talking about how good this is",
    "Absolutely brilliant performance",
    "Worth every single penny",
    "A must-have for anyone who needs this",
    "Fits perfectly and looks stunning",
    "Very reliable and easy to use",
    "Delightful experience from start to finish",
    "Top notch quality and excellent support",
    "I am absolutely thrilled with this product",
    "Smooth transaction and fantastic item",
    "Surprisingly good for such a low price",
    "The material feels incredibly soft and durable",
    "Couldn't be happier with this choice",
    "It makes my life so much easier",
    "Outstanding craftsmanship and attention to detail",
    "An absolute game changer for my daily routine",
    "The battery life is phenomenal",
    "Highly efficient and does the job perfectly",
    "Everything was handled with utmost professionalism",
    "A delightful surprise, much better than expected",
    "So glad I decided to purchase this",
    "Gives a very premium vibe",
    "Flawless design and execution",
    "My kids absolutely love playing with this",
    "Transforms the whole room, looks beautiful",
    "Very user-friendly and intuitive",
    "Solid build quality that will last for years",
    "A fantastic addition to my collection",
    "Packs a lot of features for a small price",
    "I get compliments on it all the time",
    "Truly a magical experience using this",
    "Consistently delivers great results",
    "Fast, responsive, and incredibly reliable",
    "The setup was a breeze, very straightforward",
    "I'm extremely impressed by the durability",
    "Perfect gift, my friend loved it",

    # --- Negative Reviews (50) ---
    "Absolutely terrible, do not waste your time",
    "Broke within the first five minutes of use",
    "Customer service was incredibly rude and unhelpful",
    "Looks nothing like the picture online",
    "Completely useless, does not work at all",
    "A huge disappointment, returning it immediately",
    "Save your money and buy something else",
    "The quality is shockingly bad",
    "Arrived late and the box was completely damaged",
    "Horrible smell that won't go away",
    "Stopped working after just two days",
    "Very cheap material, feels like it will break easily",
    "Missing several important parts from the package",
    "The worst online shopping experience I've had",
    "Total garbage, going straight to the trash",
    "Way overpriced for what you actually get",
    "Doesn't fit properly, sizing is completely off",
    "Instructions were confusing and unhelpful",
    "I regret buying this so much",
    "Painfully slow and constantly freezing",
    "Fell apart while I was trying to assemble it",
    "Terrible battery life, dies almost instantly",
    "Not worth a fraction of the price",
    "Absolutely awful sound quality",
    "The color faded after just one wash",
    "Scratches easily and looks terrible now",
    "Very frustrating to use and poorly designed",
    "I would give it zero stars if I could",
    "Seller refused to issue a refund for a broken item",
    "An absolute nightmare to deal with",
    "Nothing but problems since the day I got it",
    "Feels very flimsy and incredibly cheap",
    "It overheats constantly and shuts down",
    "Completely misleading product description",
    "Ruined my clothes, totally unacceptable",
    "The app keeps crashing, literally unusable",
    "Horrendous lag and terrible performance",
    "Makes a very loud and annoying noise",
    "Not compatible with my devices despite claims",
    "The stitching came undone immediately",
    "I feel completely ripped off",
    "Unsafe to use, sparked when I plugged it in",
    "Utterly pointless and badly engineered",
    "I'm thoroughly disgusted with the lack of quality control",
    "Such a letdown after reading positive reviews",
    "Hard to clean and stains permanently",
    "It's a complete scam, stay away",
    "Failed to live up to the hype in every way",
    "Gave me a terrible allergic reaction",
    "Worst customer support I have ever encountered"
]

labels = (
    ["Positive"] * 50 + 
    ["Negative"] * 50
)

print("--- 1. How the AI recognizes words (Vocabulary) ---")
# 1. Word vectorizer tool
vectorizer = TfidfVectorizer()

# The matrix created when training with data (X_train)
X_train = vectorizer.fit_transform(texts)

# The "word dictionary" created by the AI (ID assigned to each word)
vocabulary = vectorizer.vocabulary_
print(f"Total number of words: {len(vocabulary)}")
print("IDs assigned to some words:", dict(list(vocabulary.items())[:10])) 
print("\n")

print("--- 2. How the AI sees the first sentence (Vector Representation) ---")
# First sentence: "Highly recommend this item to everyone"
# The AI sees this as a long array of 0s and decimals.
first_sentence_vector = X_train.toarray()[0]
print(f"Original sentence: '{texts[0]}'")
print("Number grid seen by the AI (Array):")
print(np.round(first_sentence_vector, 2)) 
print("\n")

print("--- 3. Training the Model ---")
model = MultinomialNB()
model.fit(X_train, labels)
print("Training successful! The AI has learned the patterns.\n")

print("--- 4. How the AI evaluates a new sentence (Probabilities) ---")
test_sentences = ["I really love this product, it is good"]

# Convert the new sentence to numbers
X_test = vectorizer.transform(test_sentences)

# Check the probability of this sentence being Positive/Negative
probabilities = model.predict_proba(X_test)[0]
classes = model.classes_

print(f"Sentence to test: '{test_sentences[0]}'")

# Show the result as a percentage
for i in range(len(classes)):
    percent = round(probabilities[i] * 100, 2)
    print(f"  Probability of being {classes[i]}: {percent}%")

# Final decision
prediction = model.predict(X_test)[0]
print(f"\nFinal Decision: This is a {prediction} sentence!")