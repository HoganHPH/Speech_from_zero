import soundata

data_home = "./DATA"
dataset = soundata.initialize('urbansound8k', data_home=data_home)
dataset.download()
dataset.validate()

example_clip = dataset.choice_clip()
print(example_clip)

print("\nDOWNLOAD DATA SUCCESSFULLY !")