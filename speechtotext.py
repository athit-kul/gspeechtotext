from google.cloud import speech
import os
import io
from dotenv import load_dotenv
import gradio as gr
from pyannote.audio import Pipeline
from pyannote.audio import Model


load_dotenv()

model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token=os.getenv("Pyanotate_Token"))
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", 
                            use_auth_token=os.getenv("Pyanotate_Token"))

#asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
#classifier = pipeline("text-classification")
#input_str = "สังคล ผู้สูงอายุ เคย เป็นเรื่อง ยาว ไกล นะ ครับ"
#str_array = input_str.split(" ")
#result = ""
#for txt in str_array:
#    result += correct(txt)
#print(result)


#setting Google credential
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= os.getenv("GOOGLE_APPLICATION_CREDENTIALS")# create client instance 
client = speech.SpeechClient()

def find_time_interval(speech_start, speech_end, time_to_check):
    time_to_check_sec = time_to_check.total_seconds() 
    for idx, (start, end) in enumerate(zip(speech_start, speech_end)):
        if start <= time_to_check_sec <= end:
            return idx
    return None

def speech_to_text(audio):
    global client
    text = ""
    try:
        ##sampling = librosa.get_samplerate(speech)
        ##audio, sr = librosa.load(audio, sr=16000)
        #with io.open("./sound/1.wav", "rb") as audio_file:
        #    #diarization = pipeline(audio_file, min_speakers=2, max_speakers=5)
        #    diarization = pipeline("./sound/1.wav")
        #    text += diarization
        diarization = pipeline(audio)     
        check_speeker = ""
        speeker_id = []
        speech_start = []
        speech_end = []
        last_turn_end = ""
        for turn, _, speaker in diarization.itertracks(yield_label=True):

            speeker_id.append(speaker)
            speech_start.append(turn.start)
            speech_end.append(turn.end)


             #print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

        with io.open(audio, "rb") as audio_file:           

            content = audio_file.read()
            audioout = speech.RecognitionAudio(content=content)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                enable_automatic_punctuation=True,
                audio_channel_count=2,
                language_code="th-TH",
                enable_word_time_offsets=True,


                #diarization_config=diarization_config,

            )
            # Sends the request to google to transcribe the audio
            response = client.long_running_recognize(request={"config": config, "audio": audioout})
            result = response.result(timeout=90)
            last_speeker = ""
            for result in result.results:
                alternative = result.alternatives[0]
                #print(f"Transcript: {alternative.transcript}")
                #print(f"Confidence: {alternative.confidence}")
                for word_info in alternative.words:
                    word = word_info.word
                    start_time = word_info.start_time
                    end_time = word_info.end_time
                    interval_index = find_time_interval(speech_start, speech_end, word_info.end_time)
                    #print(
                    #    f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}"
                    #)
                    if(last_speeker != interval_index):
                        last_speeker = interval_index
                        if(text != ""):
                            text += "\n"
                        text += str(speeker_id[interval_index]) + ":"+word
                    else:
                        text += word

       

        

    except Exception as exc:
        print(str(exc))
    return text


#def text_to_sentiment(text):
#    return classifier(text)[0]["label"]


demo = gr.Blocks()

with demo:
    audio_file = gr.Audio(type="filepath")
    b1 = gr.Button("Recognize Speech")
    #label = gr.Label(label="เลือกไฟล์ที่ต้องการเป็น .mp3, .wav ความยาวไม่เกิน 1 นาที จากนั้นกดที่ Recognize Speech 1 ครั้ง แล้วรอผลข้อความ จะใช้เวลาค่อนข้างนาน 5-10 นาทีในการประมวลผล")
    label = gr.Markdown("เลือกไฟล์ที่ต้องการเป็น .mp3, .wav ความยาวไม่เกิน 1 นาที จากนั้นกดที่ Recognize Speech 1 ครั้ง แล้วรอผลข้อความ จะใช้เวลาค่อนข้างนาน ประมาณ 3 นาทีในการประมวลผล")
    text = gr.Textbox(label="Text result:")
        
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    #b2.click(text_to_sentiment, inputs=text, outputs=label)

if __name__ == "__main__":
    demo.launch(share=True)

