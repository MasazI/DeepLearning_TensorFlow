# encoding: utf-8 
from mark_base import MarkBase

class Batsu(MarkBase):
    def is_empty(self):
        return False

    def opponent(self):
        return 1

    def to_int(self):
        return -1

    def to_string(self):
        return '×'

if __name__ == '__main__':
    mark = Batsu()
    print mark.is_empty()
    print mark.opponent()
    print mark.to_int()
    print mark.to_string()
