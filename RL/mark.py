# encoding: utf-8

from maru_mark import Maru
from batsu_mark import Batsu
from empty_mark import Empty

class Mark:
    def __init__(self, mark):
        self.mark = mark

    def is_empty(self):
        return self.mark.is_empty()

    def opponent(self):
        if self.mark.opponent() == 1:
            return Maru()
        elif self.mark.opponent() == 0:
            return Empty()
        elif self.mark.opponent() == -1:
            return Batsu()

    def to_int(self):
        return self.mark.to_int()

    def to_string(self):
        return self.mark.to_string()

if __name__ == '__main__':
    maru = Mark(Maru())
    print maru.to_string()
    print maru.to_int()
    print maru.opponent().to_string()
    print '=' * 10
    batsu = Mark(Batsu())
    print batsu.to_string()
    print batsu.to_int()
    print batsu.opponent().to_string()
    print '=' * 10
    empty = Mark(Empty())
    print empty.to_string()
    print empty.to_int()
    print empty.opponent().to_string()
