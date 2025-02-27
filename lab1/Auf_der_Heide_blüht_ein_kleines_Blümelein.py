import tensorflow
from tensorflow import keras
from keras.Layers import Dense,LSTM
from keras.models import Sequential





def lstm_model():
    model = Sequential()
    
    model.add(LSTM(10,activation = 'relu', name = 'lstm'))
    model.add(Dense(10,activation = 'relu',name = 'dense1'))
    model.add(Dense(1,activation = 'softmax', name  = 'last'))
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model



def absolute_meaningless():

    print("                              ⣀⣴⣶⣶⣶⣤         ")                     
    print("                             ⢰⣿⣿⣿⣿⣿⣿⣷⡀                           ") 
    print("                            ⣼⣿⣿⣿⣿⣿⣿⣿⣇                           ") 
    print("                            ⢻⣿⣿⣿⣿⣿⣿⣿⣿                             ") 
    print("                            ⠈⢿⣿⣿⣿⣿⣿⣿⣿                             ") 
    print("                             ⠈⣿⣿⣿⣿⣿⣾⡋                             ") 
    print("                            ⢀⡴⠻⠿⠻⠿⠻⢤⡇                             ") 
    print("                          ⡀⢀⢤⣒⣽⢍⣸⣷⣶⣉⣁⣀⠑⠤⠤⢀⡀                         ") 
    print("                      ⢀⢎⣨⡟⣉⣄⣴⡎⣳⡏⡏⣡⡦⣼⠋⣷⢦⣠⣫⠳⣦                       ") 
    print("                     ⣰⢿⣅⠈⢹⠟⠸⡟⠓⢟⡷⠥⢷⢿⢻⢾⠇⣴⢮⡏⠆⢹⣧                      ") 
    print("                    ⡸⣾⣟⢬⣍⣊⠠⡐⢶⠨⢀⢠⢡⢸ ⠸⢸ ⢠⢸⣶⡳⡿⣻⣃                     ") 
    print("                  ⣠⡎⣧⡾⣿⣏⢯⢳⢶⣿⡟⢿⠟⣿⣾⢠⣮⣘⣴⠎⣬⣾⣯⣮⡞⢯⣿⡄                    ") 
    print("                ⢀⣾⢯⣍⣶⡽⣿⡿⢦⠻⢸⢘⣿⣿⣾⣿⣿⣿⡏⢿⠬⢿⡸⢻⣛⣹⡴⢮⣻⣿⡄                   ") 
    print("              ⣠⣴⠟⣜⣿⣿⢿⡽⠋ ⠘⣆⠤⡧⡃         ⢿⠶⣿⣿⣋⣩⣶⣿⣷⡀                  ") 
    print("            ⣠⣾⠟⣌⢦⣿⢡⡿⠊    ⠸⣆⡧⣈⢷⣤⣤⢲⣶   ⢀⣿⣰⣷⢇⢻⣥⣿⣻⣮⢿⣄                 ") 
    print("          ⢀⡴⡛⣡⣿⣿⣿⡿⠋       ⢻⣿⣻⣾⣿⣧⣿⡏  ⢀⣾⣷⣿⡯⣿⡀⠹⣿⣧⡈⢷⣿⣷⣄               ") 
    print("        ⢀⣴⣟⣽⣿⣿⣿⠟⠉         ⢸⠳⣻⡿⢟⣓⡿   ⢸⣽⣾⣿⢿⣹⣵ ⠘⢷⣮⡱⣿⣮⣿⣦              ") 
    print("      ⢀⣴⣿⢻⣽⣿⠿⠋            ⢸ ⠙⠶⣶⣿⠇  ⢀⣯⣾⣯⣯⣈⡇⢷⣧  ⠙⠻⣾⣿⡻⣿⣷⣄            ") 
    print("    ⢀⣴⣶⡿⣯⢿⡿⠚⠁              ⠸⡶⡒⣾⣿⠃   ⣼⣟⣍⢳⡻⡯⣯⣩⣯⡆   ⠈⠻⣝⢫⣻⣿⢷⣦⣠⠤⣴⡶⠶     ") 
    print("    ⣰⣿⣿⣿⣿⠞⠉                 ⢀⢿ ⣿⠃   ⣜⢯⣹⣯⡏⣟⣿⡧⠿⢧⠁     ⠈⠙⢾⣿⣟⣭⣥⣔⣧⣀      ") 
    print("  ⣸⣿⡿⣿⠏⠁                   ⡘⠻⡾⣿⡀  ⣰⣿⣺⣟⣯⠟⢥⡿⢟⣷⣮⡇         ⠹⣷⣻⣯⣿⡻⠿⣦⡄   ") 
    print("  ⠻⠿⢱⠟                    ⢀⠦⣵⣿⣿⡻⣿⣿⣷⣯⡾⢷⣯⣏⣣⣶⣧⢷⣮⣿⡄         ⠈⠻⣿⣻⣿⣶⣄    ")
    print("                            ⢺⣶⣿⣷⣾⣿⣿⣿⣿⢿⣿⡿⣿⣿⣿⣞⣶⠿⣽⣽⣿⡀          ⠙⠇⠙⠋⠁   ") 
    print("                          ⠠⡗⢿⣿⣧⡷⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣽⣿⣯⣵⢹⣷⡀                  ") 
    print("                        ⢠⣿⣇⣷⡿⣿⣿⣿⣮⣿⣿⣿⣿⣿⣿⣿⣿⣟⡫⣻⡿⣷⣾⣻⣧                  ") 
    print("                       ⢀⣿⣿⣿⣿⣻⣷⣿⣿⣯⣿⣿⣾⣿⣿⢿⣿⣶⣾⣏⣼⣞⣿⣿⣥⢿⡄                 ") 
    print("                       ⣾⣿⣯⣿⣿⣷⣿⣷⣿⡭⡿⣽⠏⠉⠉⠉⠲⣤⣽⣼⣿⣿⣥⣧⣧⣦⣷                 ") 
    print("                       ⢀⢿⡷⣓⣿⣿⣿⣻⣿⣻⣻⣯⠋     ⢹⣾⣟⣿⣿⣿⣿⡿⡛⢻⣇                ") 
    print("                     ⣸  ⠘⣿⣿⣭⡽⠟⠛⠛⠛⠋      ⢘⡟⣿⣿⣏⣍⣖⣹⢿⡿⡞⠃               ") 
    print("                    ⣴⠋ ⢈⣿⣿⡟⠁            ⠹⠛⠛⠻⡉⠉ ⢈⡖⢸⡀               ") 
    print("                  ⢀⣼⣥⣀⠠⣿⣿⠏                  ⢳⣀⣴⣶⣷⣶⣧               ") 
    print("                 ⣠⠗⠈⠑⢮⣿⡾⠃                   ⠸⡟⠋⠁⠘⡌⠙⢆              ") 
    print("               ⢀⣼⢋ ⠄⢱⡿⡟⠁                     ⢩⠁ ⢛⣱⡀⠈⢆             ") 
    print("             ⢀⡮⣤⣦⡀⢀⡾⡝                       ⠈⡆⠖⠋⠁⢣ ⠈⡆            ") 
    print("             ⣾⠃⡰⠉⣳⣾⡝                         ⠘⡾⠋⠠⠈⣆ ⠸⡀           ") 
    print("            ⡜  ⢠⣾⡿⠋                           ⠈⠣⡀ ⠈  ⢱           ") 
    print("           ⢰⢁⢤⣰⣿⠟                               ⠙⣄    ⢇          ") 
    print("          ⢀⠃⣮⣿⡿⠁                                 ⠈⢆  ⣀⠘⡄         ") 
    print("         ⢠⠎⢤⡿⡟                                    ⠈⢣⢊⠬⠆⠑⣄        ") 
    print("        ⣰⣿⣿⡿⣥⠁                                     ⠈⢦⣶⡒⢶⡞⠢⣄      ") 
    print("      ⣠⣾⣛⡏⡻⣳⠃                                       ⢸⢿⣿⣿⣾⢷⣿⣿⣶⣯⡿⢆⡀") 
    print("     ⠉⠉⠚⠚⠛⠋                                          ⠸⠦⠿⠿⠿⠿⠿⠿⠿⠿⠷⠷⠾") 
