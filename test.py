import pickle
import re
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_classifier():
    print("Загрузка модели...")

    required_files = [
        'model.keras',
        'tokenizer.pkl',
        'label_encoder.pkl',
        'model_params.pkl'
    ]

    for f in required_files:
        if not os.path.exists(f):
            print(f"Ошибка: файл '{f}' не найден!")
            print("Сначала запустите обучение: python train.py")
            return None, None, None, None

    model = load_model('model.keras')

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('model_params.pkl', 'rb') as f:
        params = pickle.load(f)

    print("Модель загружена")
    print(f"  Классы: {params['genres']}")
    print(f"  Словарь: {params['vocab_size']}, макс. длина: {params['max_length']}")

    return model, tokenizer, label_encoder, params


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_genre(text, model, tokenizer, label_encoder, params):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=params['max_length'], padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)

    predicted_index = np.argmax(prediction[0])
    predicted_genre = label_encoder.inverse_transform([predicted_index])[0]
    confidence = prediction[0][predicted_index]

    all_probs = {
        label_encoder.inverse_transform([i])[0]: float(prediction[0][i])
        for i in range(len(prediction[0]))
    }

    return predicted_genre, confidence, all_probs


def print_prediction(text, genre, confidence, all_probs):
    print(f"\n{'─'*60}")
    print(f"Текст: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"{'─'*60}")
    print(f"\nЖанр: {genre} ({confidence*100:.1f}%)")
    print("\nРаспределение:")

    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

    for g, prob in sorted_probs:
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " <-" if g == genre else ""
        print(f"  {g:25} [{bar}] {prob*100:5.1f}%{marker}")


TEST_EXAMPLES = [
    {
        "text": "Увлекательная история о пиратах, которые отправились на поиски сокровищ на далёкий остров. Отважный капитан и его команда преодолевают множество опасностей.",
        "expected": "Приключения"
    },
    {
        "text": "Сборник стихотворений для детей о природе, временах года и животных. Весёлые и добрые рифмы для самых маленьких.",
        "expected": "Стихи"
    },
    {
        "text": "Волшебная история о маленькой принцессе, которая жила в заколдованном замке. Добрая фея помогла ей найти дорогу домой.",
        "expected": "Сказка"
    },
    {
        "text": "Повесть о школьной жизни, о дружбе между одноклассниками, о первой любви и взрослении современных подростков.",
        "expected": "Проза"
    },
    {
        "text": "Учебное пособие по математике для начальной школы. Задачи и примеры для развития логического мышления.",
        "expected": "Образовательная литература"
    },
    {
        "text": "Детектив про юных сыщиков, которые раскрывают загадочное преступление в маленьком городке.",
        "expected": "Приключения"
    },
    {
        "text": "Жили-были дед да баба, и была у них курочка Ряба. Снесла курочка яичко не простое, а золотое.",
        "expected": "Сказка"
    },
    {
        "text": "Автобиографическая повесть о детстве автора в деревне. Воспоминания о бабушке, играх с друзьями и первых жизненных уроках.",
        "expected": "Проза"
    },
]


def main():
    print("=" * 60)
    print("  ТЕСТИРОВАНИЕ КЛАССИФИКАТОРА ЖАНРОВ КНИГ")
    print("=" * 60)

    model, tokenizer, label_encoder, params = load_classifier()
    if model is None:
        return

    print(f"\nТестирование на {len(TEST_EXAMPLES)} примерах:")

    correct = 0
    total = len(TEST_EXAMPLES)

    for i, example in enumerate(TEST_EXAMPLES, 1):
        print(f"\n--- Пример {i}/{total} ---")

        genre, confidence, all_probs = predict_genre(
            example["text"], model, tokenizer, label_encoder, params
        )
        print_prediction(example["text"], genre, confidence, all_probs)

        if genre == example["expected"]:
            correct += 1
            print(f"\nВерно (ожидалось: {example['expected']})")
        else:
            print(f"\nНеверно (ожидалось: {example['expected']})")

    print(f"\n{'=' * 60}")
    print(f"  Результат: {correct}/{total} ({correct / total * 100:.0f}%)")
    print(f"{'=' * 60}")

    print("\nИнтерактивный режим — введите описание книги.")
    print("Для выхода: 'выход' или 'exit'\n")

    while True:
        try:
            user_input = input("Описание книги: ").strip()

            if user_input.lower() in ['выход', 'exit', 'quit', 'q', '']:
                print("До свидания!")
                break

            if len(user_input) < 10:
                print("Слишком короткое описание, попробуйте ещё раз.")
                continue

            genre, confidence, all_probs = predict_genre(
                user_input, model, tokenizer, label_encoder, params
            )
            print_prediction(user_input, genre, confidence, all_probs)

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
