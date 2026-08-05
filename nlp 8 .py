import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

reviews = []
n = int(input("Enter number of reviews: "))

for i in range(n):
    reviews.append(input("Enter review: "))

# Accepts words and numbers
vectorizer = CountVectorizer(
    stop_words="english",
    token_pattern=r"(?u)\b\w+\b"
)

X = vectorizer.fit_transform(reviews)

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\nTopics:")
for i, topic in enumerate(lda.components_):
    print("\nTopic", i + 1)
    for j in topic.argsort()[-5:][::-1]:
        print(words[j])

# Assign each review to the most likely topic
topic_dist = lda.transform(X)
topic_labels = topic_dist.argmax(axis=1)

counts = [sum(topic_labels == 0), sum(topic_labels == 1)]

print("\nReviews per Topic:")
print("Topic 1:", counts[0])
print("Topic 2:", counts[1])

# Pie Chart
plt.figure(figsize=(6,6))
plt.pie(
    counts,
    labels=["Topic 1", "Topic 2"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Customer Reviews by Topic")
plt.show()