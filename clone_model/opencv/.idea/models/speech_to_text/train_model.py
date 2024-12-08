from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Tokenizer

custom_model_path = "models\\speech_to_text"

model_name = "facebook/wav2vec2-large-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

model.save_pretrained(custom_model_path)
tokenizer.save_pretrained(custom_model_path)

stt_pipeline = pipeline(task="automatic-speech-recognition", model=custom_model_path)

audio_path = "data\\converted_data2\\500k\\audio1.wav"
output_text = stt_pipeline(audio_path)

print(f"Detected Text: {output_text['text']}")