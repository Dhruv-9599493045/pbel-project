from transformers import pipeline

def main():
    print("Loading the emotion recognition model... (This may take a moment to download on first run)")
    try:
        # We use a model specifically fine-tuned for emotion detection
        # It detects emotions like anger, disgust, fear, joy, neutral, sadness, and surprise
        emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Failed to load the model. Error: {e}")
        print("\nPlease make sure you have the required libraries installed.")
        print("Run the following command in your terminal:")
        print("pip install transformers torch")
        return

    while True:
        print("-" * 50)
        user_input = input("Enter a sentence to analyze its emotion (or type 'quit' to exit):\n> ")
        
        if user_input.strip().lower() in ['quit', 'exit']:
            print("Exiting. Goodbye!")
            break
            
        if not user_input.strip():
            print("Please enter some text.")
            continue

        try:
            # Perform emotion classification
            results = emotion_classifier(user_input)[0]
            
            # Sort the results by score in descending order
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            print("\nEmotion Analysis Results:")
            for result in sorted_results:
                label = result['label'].capitalize()
                score = result['score'] * 100
                print(f" - {label}: {score:.2f}%")
                
            dominant_emotion = sorted_results[0]
            print(f"\n=> Dominant Emotion: {dominant_emotion['label'].capitalize()} ({dominant_emotion['score']*100:.2f}%)")
            
        except Exception as e:
            print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
