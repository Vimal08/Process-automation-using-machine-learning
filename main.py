import pandas as pd
import numpy as np
import tensorflow as tf
import normal
import com

#reading the smaple data
df = pd.read_csv('trail.csv')
print(df.head())
print(np.mean(df))

x_value = df['Value'].values

#input and output array
input = []
output =[]

for i in range(0,len(x_value)):
    input.append(normal.norms(x_value[i]))
for j in range(0,len(x_value)):
    output.append(normal.fun(x_value[j]))

input = np.array(input)
output = np.array(output)

#randamization
num = len(x_value)
ran = np.arange(num)
np.random.shuffle(ran)

input = input[ran]
output = output[ran]

#splitting the data for train and split
TRAIN_SPLIT = int(0.6*num)
TEST_SPLIT = int(0.2*num+TRAIN_SPLIT)

x_train,x_validation,x_test = np.split(input,[TRAIN_SPLIT,TEST_SPLIT])
y_train,y_validation,y_test = np.split(output,[TRAIN_SPLIT,TEST_SPLIT])

#model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50,activation='relu'))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(len(output),activation='softmax'))
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
history = model.fit(x_train,y_train,epochs=10,batch_size=1,validation_data=(x_validation,y_validation))

#prediction
pr =[]
while(1):
    pr.append(com.Com())
    prediction = model.predict(pr)
    print('prediction')
    print(np.round(prediction,decimals=5))
    print('Acutal',normal.fun(com.Com()))
