#encoding: utf-8
'''
tic tac toe のSarsa λ実行クラス
'''
from tic_tac_toe_sarsa_r_com import SarsaRCom

from mark import Mark
from maru_mark import Maru
from batsu_mark import Batsu
from tic_tac_toe_game import Game

import dill

com_1 = SarsaRCom(Mark(Maru()), 0.1, 0.1, 0.6)
com_2 = SarsaRCom(Mark(Batsu()), 0.1, 0.1, 0.6)

iterations = 100000
print("Input the number of iterations (%d):" % (iterations))
while(True):
    input_line = raw_input()
    if input_line.isdigit():
        iterations = int(input_line)
        break
    elif input_line == '':
        break
    else:
        print("Input number:")

# 学習
for i in xrange(iterations):
    game = Game(com_1, com_2)
    if i % 1000 == 0:
        print("training iterations: No.%d" % (i))
        game.start(True)
    else:
        game.start(False)

# com同士のデモンストレーション
com_1.training = False
com_1.verbose = True
com_2.training = False
com_2.verbose = True

game = Game(com_1, com_2)
game.start(True)

# モデルの保存
with open('tic_tac_toe_com_1_sarsa_r.pkl', 'wb') as f:
    dill.dump(com_1, f)

with open('tic_tac_toe_com_2_sarsa_r.pkl', 'wb') as f:
    dill.dump(com_2, f)


