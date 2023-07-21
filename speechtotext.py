#from pythaiasr import asr
#import gradio as gr
#import sys
#import os
#import shutil

#def upload_file(files):
#    print("Speech to text")
#    file_paths = [file.name for file in files]    
#    for file_to_copy in file_paths:
#        path = "./sound/" + os.path.basename(file_to_copy)
#        shutil.copyfile(file_to_copy, path)        
#        sound_file = file_to_copy
#        text_result = asr(sound_file)
#    return text_result

#with gr.Blocks() as demo:
#    file_output = gr.File()
#    upload_button = gr.UploadButton("Click to Upload a File", file_types=[".wav", ".mp3"], file_count="1")
#    upload_button.upload(upload_file, upload_button, file_output)

#demo.launch()

from logging import exception
from pickle import NONE
import queue
from pythaiasr import asr
import librosa 
import soundfile as sf
import numpy as np
import gradio as gr
from pythainlp.spell import correct

#asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
#classifier = pipeline("text-classification")
#input_str = "สังคล ผู้สูงอายุ เคย เป็นเรื่อง ยาว ไกล นะ ครับ"
#str_array = input_str.split(" ")
#result = ""
#for txt in str_array:
#    result += correct(txt)
#print(result)

def speech_to_text(speech):
    segment_dur_secs = 15  
    text = ""
    try:
        #sampling = librosa.get_samplerate(speech)
        speech, sr = librosa.load(speech, sr=16000)
        segment_length = sr * segment_dur_secs
        num_sections = int(np.ceil(len(speech) / segment_length))

        split = []

        for i in range(num_sections):
            t = speech[i * segment_length: (i + 1) * segment_length]
            split.append(t)

        for i in range(num_sections):
            #sf.write("./sound/dest_audio_"+str(i)+".wav", split[i], sr)
            #convert_result = asr(split[i], model="wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm", lm=True)
            convert_result = asr(split[i])
            str_array = convert_result.split(" ")
            for word in str_array:
                if(word == ""):
                    text += " "
                else:
                    text += word
                #text += correct(word)
            #text += correct(convert_result)
            #text += asr(split[i],device="cuda")
        
        
        
        #for i in range(0, len(speech),5 * sr):
        #    y = speech[5 * sr * i: 5 * sr *(i+1)]
        #    sf.write("./sound/dest_audio_"+str(i)+".wav", y, sr)
        #sf.write('./sound/00.mp3',speech,samplerate=16000)
        #text =  asr(speech)
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
    label = gr.Markdown("เลือกไฟล์ที่ต้องการเป็น .mp3, .wav ความยาวไม่เกิน 1 นาที จากนั้นกดที่ Recognize Speech 1 ครั้ง แล้วรอผลข้อความ จะใช้เวลาค่อนข้างนาน 3 นาทีในการประมวลผล")
    text = gr.Textbox(label="Text result:")
        
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    #b2.click(text_to_sentiment, inputs=text, outputs=label)

if __name__ == "__main__":
    demo.launch(share=True)

