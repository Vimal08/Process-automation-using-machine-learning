#serial communicate with board
import serial

board = serial.Serial('COM3',115200)
#data calling function
def Com():
    var = board.readline()
    var_n = var.decode().rstrip()
    str = int(var_n)
    return str
