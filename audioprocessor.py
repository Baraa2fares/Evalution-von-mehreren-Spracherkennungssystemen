import speech_recognition as sr
from pydub import AudioSegment
import pyttsx3
from vosk import Model, KaldiRecognizer
import wave
import json
import deepspeech
import numpy as np


class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def recognize_audio(self, file_paths, library):
        if library == 'vosk':
           return self.recognize_audio_vosk(file_paths)
        elif library == 'speech_recognition':
           return self.recognize_audio_speech_recognition(file_paths)
       
        elif library == 'deepspeech_mozilla':
           return self.recognize_audio_deepspeech_mozilla(file_paths)
        else:
           raise ValueError("Invalid library name")
    #speech_recognition
    def recognize_audio_speech_recognition(self,files):
        results = []
        for i, file_path in enumerate(files):
          
        # Check the file format and convert to WAV if needed
            if not file_path.endswith(".wav"):
                audio = AudioSegment.from_file(file_path, format="mp3")
                file_path = "temp.wav"
                audio.export(file_path, format="wav")

            # Load the audio file
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
   
            language = "de-DE" if i == 0 else "en-US"     
            try:
                # Use the recognize_google function to convert the audio to text
                text = self.recognizer.recognize_google(audio_data, language=language)
                results.append(text)
            except sr.UnknownValueError:
                results.append("Speech recognition could not understand the audio content.")
            except sr.RequestError:
                results.append("There was an issue with the API request.")

        return results
 
    def recognize_audio_vosk(self,files):
        results=[]
        for i, file_path in enumerate(files):
            model = Model("models_vosk/vosk-model-small-de-0.15") if i == 0 else Model("models_vosk/vosk-model-small-en-us-0.15")
            kaldi_recognizer = KaldiRecognizer(model, 16000)
           
            with wave.open(file_path, 'rb') as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                    return "Audio file must be mono PCM."
                
                recognized_text = ""
                while True:
                    data = wf.readframes(8192)
                    if len(data) == 0:
                        break

                    if kaldi_recognizer.AcceptWaveform(data):
                        
                       
                       
                        result_text = kaldi_recognizer.FinalResult()
                        data_dict = json.loads(result_text)

                      
                        text = data_dict["text"]
      
                       

                        recognized_text += text + " "
                    
                results.append(recognized_text)        

        return results
    
    def recognize_audio_deepspeech_mozilla(self,files):
        results=[]
        for i, file_path in enumerate(files):
            model_file_path = "models_deep_speech_mozilla/output_graph_de.pbmm" if i == 0 else "models_deep_speech_mozilla/deepspeech-0.9.3-models.pbmm"
            model = deepspeech.Model(model_file_path)

        
            
            stream = model.createStream()

            buf = bytearray(1024)
            with open(file_path, 'rb') as audio:
                while audio.readinto(buf):
                    data16 = np.frombuffer(buf, dtype=np.int16)
                    stream.feedAudioContent(data16)

            text = stream.finishStream()
            results.append(text)    

        return results    

    def write_results_to_file(self, results, output_file):
        with open(output_file, "w") as file:
            for i, result in enumerate(results):
                file.write(f"Result {i + 1}:\n\n")
                file.write(result + "\n")

    def convert_output_to_dict(self,results):
        libraries = {
       "vosk_library": {'Deutch' : results[0], 'english' : results[1]},
       "speech_recognition_library": {'Deutch' : results[2], 'english' : results[3]}}
        
        return libraries
   
# Test the AudioProcessor class and save results to a file
if __name__ == "__main__":
    processor = AudioProcessor()

    # Test speech recognition with Vosk
    file_paths = ["audio/Zitieren_wav.wav", "audio/231023-fructose-and-obesity wav.wav"]
    libraries = ['vosk', 'speech_recognition','deepspeech_mozilla']

    results = []
    for library in libraries:
        results.extend(processor.recognize_audio(file_paths, library))

    output_file = "recognized_results.txt"
    processor.write_results_to_file(results, output_file)

    print("Results saved to", output_file)
