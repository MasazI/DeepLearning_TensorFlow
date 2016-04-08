# encoding: utf-8
'''
tic tac toe プレイヤークラス
'''
class Player:
    def __init__(self, mark):
        self.mark = mark

    def select_index(self, state):
        '''
        インデックスの選択
        '''
        print("Player: %s" % (self.mark.to_string()))
        actions = state.get_valid_actions()
        while(True):
            print("Please select index %s" % (actions))
            input_line = raw_input()

            if input_line == '':
                continue
            elif not input_line.isdigit():
                continue
            if int(input_line) in actions:
                return int(input_line)

    def learn(self, reward, finished=False):
        '''
        学習(AIにオーバーライドさせる）
        '''
        pass
