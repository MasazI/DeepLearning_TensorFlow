#encoding: utf-8
'''
state of tic tac toe game
'''
from mark import Mark
from maru_mark import Maru
from batsu_mark import Batsu
from empty_mark import Empty

import copy

class State:
    win_state = ([0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6])

    '''
    状態クラス：状態を系列で保持するため、状態変更関数は新しい状態を返す
    '''
    def __init__(self):
        # 盤面を配列で保持
        self.state = [Mark(Empty())]*9

    def __hash__(self):
        value = 0
        for i in xrange(9):
            value += 2 ** i * self.state[i].to_int()
        return hash(value)

    def __eq__(self, other):
        for (a, b) in zip(self.state, other.state):
            if a.to_int() != b.to_int():
                return False
        return True

    def get_valid_actions(self):
        '''
        選択可能な行動の取得
        return:
            空のマスを示すindex
        '''
        valid_actions = []
        for i in xrange(9):
            if self.state[i].is_empty():
                valid_actions.append(i)
        return valid_actions

    def get(self, index):
        '''
        指定したindexのマスのMarkを返す
        arguments:
            index: インデックス
        return:
            インデックスのマスのMark
        '''
        return self.state[index]
    
    def set(self, index, mark):
        '''
        現在の状態を変えずに新しい状態を取得
        arguments:
            index: 状態を変えるマスのインデックス
            mark: 状態を変えるマスのMark
        return:
            新しい状態
        '''
        new_state = State()
        new_state.state = copy.deepcopy(self.state)
        new_state.state[index] = mark
        return new_state

    def to_array(self):
        '''
        整数型の状態配列
        '''
        int_array = []
        for mark in self.state:
            int_array.append(mark.to_int())
        return int_array


    def is_win(self, mark):
        '''
        指定したマークの手が勝利かどうか
        arguments:
            mark:マーク
        return:
            True:指定したマークが勝利
            False:指定したマークは勝利していない（勝負がついていない）
        '''
        mark_int = mark.to_int()
        for i, j, k in self.win_state:
            if self.state[i].to_int() == mark_int and self.state[j].to_int() == mark_int and self.state[k].to_int() == mark_int:
                return True
        return False

    def is_draw(self):
        '''
        盤面の状態が引き分けかどうか
        return:
            True:引き分け
            False:引き分けではない（勝負がついていない）
        '''
        if len(self.get_valid_actions()) == 0 and not self.is_win(Mark(Maru())) and not self.is_win(Mark(Batsu())):
            return True
        return False

    def is_end(self):
        '''
        盤面の状態がゲーム修了かどうか
        return:
            True:修了
            False:修了していない（勝負がついていない）
        '''
        if self.is_win(Mark(Maru())) or self.is_win(Mark(Batsu())) or len(self.get_valid_actions()) == 0:
            return True
        return False

    def output(self):
        for x, y, z in zip(*[iter(self.state)]*3):
            print("%s %s %s" % (x.to_string(), y.to_string(), z.to_string()))

if __name__ == '__main__':
    state1 = State()
    print state1.state
    print state1.get_valid_actions()
    print state1.get(0).to_int()
    new_state = state1.set(0, Mark(Maru()))
    print new_state
    print new_state.get(0).to_int()
    print state1.to_array()
    print new_state.to_array()
    print state1.win_state
    print state1.is_win(Mark(Maru()))
    maru1 = Mark(Maru())
    maru2 = Mark(Maru())
    print maru1.to_int() == maru2.to_int()
    new_state2 = new_state.set(1, Mark(Maru()))
    new_state3 = new_state2.set(2, Mark(Maru()))
    print new_state3.to_array()
    print new_state3.is_win(Mark(Maru()))
    print new_state2.is_draw()
    print new_state2.is_end()
    print new_state3.is_end()
    new_state = state1.set(0, Mark(Batsu()))
    new_state2 = new_state.set(1, Mark(Maru()))
    new_state4 = new_state2.set(2, Mark(Batsu()))
    new_state5 = new_state4.set(3, Mark(Maru()))
    new_state6 = new_state5.set(4, Mark(Batsu()))
    new_state7 = new_state6.set(5, Mark(Batsu()))
    new_state8 = new_state7.set(6, Mark(Maru()))
    new_state9 = new_state8.set(7, Mark(Batsu()))
    new_state10 = new_state9.set(8, Mark(Maru()))
    print new_state10.to_array()
    print new_state10.is_draw()

    new_state10.output()

    state11 = State()
    state12 = State()
    print state11 == state12
    print new_state == state11
