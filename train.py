import json
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("Классификатор жанров детских книг")
print("="*60)

with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

genres = [item['genre'] for item in data]
descriptions = [item['description'] for item in data]

print(f"\nВсего записей: {len(data)}")
print(f"Уникальных жанров: {len(set(genres))}")

MIN_SAMPLES = 10
genre_counts = {}
for g in genres:
    genre_counts[g] = genre_counts.get(g, 0) + 1

valid_genres = {g for g, count in genre_counts.items() if count >= MIN_SAMPLES}

filtered_data = [(desc, genre) for desc, genre in zip(descriptions, genres) if genre in valid_genres]
descriptions = [d[0] for d in filtered_data]
genres = [d[1] for d in filtered_data]

print(f"После фильтрации: {len(descriptions)} записей, {len(set(genres))} жанров")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_descriptions = [clean_text(desc) for desc in descriptions]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(genres)
num_classes = len(set(genres))
labels_one_hot = to_categorical(encoded_labels, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_descriptions, 
    labels_one_hot, 
    test_size=0.2, 
    random_state=42,
    stratify=encoded_labels
)

print(f"\nОбучающая выборка: {len(X_train)}")
print(f"Тестовая выборка: {len(X_test)}")

VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 64

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

print(f"Форма данных: {X_train_padded.shape}")

print("\nСоздание модели...")

model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 20
BATCH_SIZE = 32

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(encoded_labels),
    y=encoded_labels
)
class_weight_dict = dict(enumerate(class_weights))

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print(f"\nОбучение модели (до {EPOCHS} эпох, EarlyStopping patience=3)...")

history = model.fit(
    X_train_padded,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"\nТочность на тесте: {test_accuracy*100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], 'b-', label='Обучение', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 'r-', label='Валидация', linewidth=2)
axes[0].set_title('Точность модели')
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Точность')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], 'b-', label='Обучение', linewidth=2)
axes[1].plot(history.history['val_loss'], 'r-', label='Валидация', linewidth=2)
axes[1].set_title('Функция потерь')
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Потери')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("\nГрафики сохранены")

model.save('model.keras')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
    
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

params = {
    'vocab_size': VOCAB_SIZE,
    'max_length': MAX_LENGTH,
    'embedding_dim': EMBEDDING_DIM,
    'num_classes': num_classes,
    'genres': list(label_encoder.classes_)
}

with open('model_params.pkl', 'wb') as f:
    pickle.dump(params, f)

print("Модель и параметры сохранены")

def load_and_predict(text, loaded_model, loaded_tokenizer, loaded_label_encoder, loaded_params):
    cleaned = clean_text(text)
    sequence = loaded_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=loaded_params['max_length'], padding='post')
    prediction = loaded_model.predict(padded, verbose=0)

    predicted_index = np.argmax(prediction[0])
    predicted_genre = loaded_label_encoder.inverse_transform([predicted_index])[0]

    return predicted_genre, prediction[0], loaded_params['genres']

test_descriptions = [
    "Увлекательная история о маленьком мальчике, который отправился в далёкое путешествие.",
    "Сборник стихов для детей о временах года и природе.",
    "Волшебная история о принцессе в заколдованном замке.",
    "Рассказы о школьной жизни и дружбе между подростками."
]

print("\nПримеры предсказаний:")

loaded_model = load_model('model.keras')

with open('tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    loaded_label_encoder = pickle.load(f)

with open('model_params.pkl', 'rb') as f:
    loaded_params = pickle.load(f)

for i, desc in enumerate(test_descriptions, 1):
    predicted_genre, probabilities, genre_names = load_and_predict(
        desc, loaded_model, loaded_tokenizer, loaded_label_encoder, loaded_params
    )
    print(f"\n{i}. {desc[:60]}...")
    print(f"   Жанр: {predicted_genre} ({probabilities[np.argmax(probabilities)]*100:.1f}%)")

print("\n" + "="*60)
print(f"Завершено. Точность: {test_accuracy*100:.2f}%")
print("="*60)