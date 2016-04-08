#encoding: utf-8
'''
tic tac toeゲームを対戦するクラス
'''
from mark import Mark
from maru_mark import Maru
from batsu_mark import Batsu
from empty_mark import Empty

from tic_tac_toe_state import State
from tic_tac_toe_player import Player

class Game:
    '''
    対戦クラス
    '''
    def __init__(self, maru_player, batsu_player):
        '''
        初期化
        arguments:
            Maru Player
            Batsu Player
        '''
        self.players = {1: maru_player, -1: batsu_player}

    def start(self, verbose=False):
        '''
        対戦の開始
        '''
        state = State()
        current_player_mark = 1
        result = None
        while(True):
            #print("="*30)
            current_player = self.players[current_player_mark]
            if verbose:
                print("%s" % (state.to_array()))
                print state.output()
                print("-"*5)
            # プレイヤーの行動の選択
            index = current_player.select_index(state)
            #print("%s selected %i" % (self.players[current_player_mark].mark.to_string(), index))
            state = state.set(index, self.players[current_player_mark].mark)

            # この時点のstateで報酬が発生する場合はここでrewardを判定して学習できる
            # tic_tac_toeでは勝負が決まるまで報酬は0
            current_player.learn(0)

            if state.is_win(self.players[current_player_mark].mark):
                result = self.players[current_player_mark].mark
                # 勝者の報酬
                current_player.learn(1, True)
                # 敗者の報酬
                self.players[result.opponent().to_int()].learn(-1, True)
                if verbose:
                    print("%s" % (state.to_array()))
                    print("-"*5)
                    state.output()
                    print("-"*5)
                    print("%s win!!!" % (self.players[current_player_mark].mark.to_string()))
                break
            elif state.is_draw():
                result = Mark(Empty())
                for player in self.players.itervalues():
                    player.learn(0, True)
                if verbose:
                    state.output()
                    print("draw.")
                break
            current_player_mark = self.players[current_player_mark].mark.opponent().to_int()
            #print("="*30)

if __name__ == '__main__':
    player1 = Player(Mark(Maru()))
    player2 = Player(Mark(Batsu()))
    game = Game(player1, player2)
    game.start(verbose=True)
