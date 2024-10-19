import sys
from data import featurelen
from UModel import UModel
from umsc import UgMultiScriptConverter
source_script = 'UAS'
target_script = 'ULS'
converter = UgMultiScriptConverter(source_script, target_script)


if __name__ == '__main__':
    model = UModel(featurelen)
    
    if len(sys.argv)<2:
        print("Using \n\tpython tonu.py audiofile | audiolist.txt | folder")
    else:
        device = 'cpu'
        model.to(device)
        audiofile = sys.argv[1]
        txt = model.predict(audiofile,device)
        script = converter(txt)
        print(script)
