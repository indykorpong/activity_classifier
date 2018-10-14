from keras.models import Model
from keras.layers import Input, LSTM, Dense
from os import listdir
from os.path import isfile, join

mypath = "Dataset/AI/"
file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
label_count = 4 	# 4 different activities in filelist
input_dim = 15
# print(file_list)

# create denses for input and output
inputs = Input(shape=(input_dim,len(file_list)))

x = Dense(input_dim, activation='relu')(inputs)
x = Dense(input_dim, activation='relu')(x)
predictions = Dense(label_count, activation='softmax')(x)

# data and label
data = np.zeros(
	(input_dim, len(file_list)),
    dtype='float32')
label = np.zeros(
    (label_count, len(file_list)),
    dtype='float32')

for file in file_list:
	f = open(file,"r")
	

# define and fit the final model
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(data,label)
